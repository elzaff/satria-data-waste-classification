"""Full ConvNeXtV2-L fine-tuning for BDC2026 on Modal.

Run training and inference:
  modal run modal_convnextv2_pipeline.py --action all

Resume inference from the saved final checkpoint:
  modal run modal_convnextv2_pipeline.py --action infer
"""

from pathlib import Path

import modal

from modal_backbone_app import (
    CACHE_ROOT,
    DATA_ROOT,
    SEED,
    cache_volume,
    data_volume,
    hf_secret,
    image,
)

# Modal only mounts this entrypoint by default; the shared image definition is
# also imported when the entrypoint starts inside the remote container.
shared_app = Path(__file__).with_name("modal_backbone_app.py")
if not shared_app.exists():
    shared_app = Path.cwd() / "modal_backbone_app.py"
image = image.add_local_file(shared_app, "/root/modal_backbone_app.py")


APP_NAME = "bdc2026-convnextv2-finetune"
GPU = "A100-80GB"
MODEL_NAME = "facebook/convnextv2-large-22k-224"
MODEL_REVISION = "e58a79c331e6c9acd20e3ba2de0e934c546f0eea"
RUN_NAME = "convnextv2_large_full_finetune_224_seed2026"
RUN_ROOT = f"{CACHE_ROOT}/runs/{RUN_NAME}"

NUM_CLASSES = 3
IMAGE_SIZE = 224
BATCH_SIZE = 32
GRAD_ACCUMULATION = 2
MAX_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 3
BACKBONE_LR = 2e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
NUM_WORKERS = 8

app = modal.App(APP_NAME)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    volumes={"/data": data_volume},
)
def prepare_manifest():
    import hashlib

    import pandas as pd

    data_root = Path(DATA_ROOT)
    manifest_path = data_root / "train_manifest.csv"
    frame = None
    if manifest_path.exists():
        try:
            candidate = pd.read_csv(manifest_path)
            if (
                len(candidate) == 26_527
                and set(candidate.columns) == {"path", "label", "group"}
                and set(candidate["label"].astype(int)) == {0, 1, 2}
            ):
                frame = candidate
        except (OSError, ValueError):
            pass
    if frame is None:
        rows = []
        folders = {
            0: data_root / "train" / "0_Recyclable",
            1: data_root / "train" / "1_Electronic",
            2: data_root / "train" / "2_Organic",
        }
        for label, folder in folders.items():
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing dataset folder: {folder}")
            for path in sorted(p for p in folder.iterdir() if p.is_file()):
                rows.append(
                    {
                        "path": str(path),
                        "label": label,
                        "group": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        frame = pd.DataFrame(rows)
        frame.to_csv(manifest_path, index=False)
        data_volume.commit()
    result = {
        "rows": len(frame),
        "duplicate_extras": int(frame.duplicated("group").sum()),
    }
    print(result, flush=True)
    return result


@app.function(
    image=image,
    gpu=GPU,
    timeout=12 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def train(force: bool = False):
    import hashlib
    import json
    import math
    import os
    import random

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageFile, ImageOps
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import (
        AutoModelForImageClassification,
        get_cosine_schedule_with_warmup,
    )

    run_root = Path(RUN_ROOT)
    final_path = run_root / "final_model.pt"
    metrics_path = run_root / "training_metrics.json"
    if final_path.exists() and metrics_path.exists() and not force:
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        result["cached"] = True
        print(json.dumps(result, indent=2), flush=True)
        return result

    os.environ["PYTHONHASHSEED"] = str(SEED)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    def seed_worker(worker_id):
        worker_seed = (torch.initial_seed() + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    class ResizePad:
        def __call__(self, source):
            resized = ImageOps.contain(
                source,
                (IMAGE_SIZE, IMAGE_SIZE),
                method=Image.Resampling.BICUBIC,
            )
            canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (124, 116, 104))
            offset = (
                (IMAGE_SIZE - resized.width) // 2,
                (IMAGE_SIZE - resized.height) // 2,
            )
            canvas.paste(resized, offset)
            return canvas

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.15, 0.15, 0.15, 0.05)], p=0.5
            ),
            transforms.RandomRotation(8, fill=(124, 116, 104)),
            ResizePad(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08), value="random"),
        ]
    )
    eval_transform = transforms.Compose(
        [ResizePad(), transforms.ToTensor(), normalize]
    )

    class WasteDataset(Dataset):
        def __init__(self, frame, transform):
            self.paths = frame["path"].astype(str).tolist()
            self.labels = frame["label"].astype(int).tolist()
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            with Image.open(self.paths[index]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                return self.transform(image), self.labels[index]

    def make_model():
        return AutoModelForImageClassification.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
            id2label={0: "Recyclable", 1: "Electronic", 2: "Organic"},
            label2id={"Recyclable": 0, "Electronic": 1, "Organic": 2},
        )

    def make_loader(frame, transform, shuffle, seed, batch_size=BATCH_SIZE):
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            WasteDataset(frame, transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def train_epochs(model, train_loader, epochs, class_weights, device, valid_loader=None):
        backbone_parameters, head_parameters = [], []
        for name, parameter in model.named_parameters():
            (head_parameters if name.startswith("classifier") else backbone_parameters).append(
                parameter
            )
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": BACKBONE_LR},
                {"params": head_parameters, "lr": HEAD_LR},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=LABEL_SMOOTHING,
        )
        updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUMULATION)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(0.05 * updates_per_epoch * epochs)),
            num_training_steps=updates_per_epoch * epochs,
        )
        history = []
        best = None
        stale_epochs = 0
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for step, (images, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(pixel_values=images).logits
                    raw_loss = criterion(logits, targets)
                    loss = raw_loss / GRAD_ACCUMULATION
                loss.backward()
                total_loss += raw_loss.detach().item()
                should_update = (step + 1) % GRAD_ACCUMULATION == 0 or (
                    step + 1 == len(train_loader)
                )
                if should_update:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            row = {"epoch": epoch, "train_loss": total_loss / len(train_loader)}
            if valid_loader is not None:
                metrics = evaluate(model, valid_loader, device)
                row.update(metrics)
                if best is None or metrics["macro_f1"] > best["macro_f1"] + 1e-6:
                    best = dict(row)
                    stale_epochs = 0
                    torch.save({"model": model.state_dict(), "metrics": best}, run_root / "best_validation.pt")
                    cache_volume.commit()
                else:
                    stale_epochs += 1
            history.append(row)
            print(json.dumps(row), flush=True)
            if valid_loader is not None and stale_epochs >= EARLY_STOPPING_PATIENCE:
                break
        return history, best

    @torch.inference_mode()
    def evaluate(model, loader, device):
        model.eval()
        labels, predictions = [], []
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pixel_values=images).logits
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(targets.tolist())
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
        }

    seed_everything(SEED)
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(Path(DATA_ROOT) / "train_manifest.csv")
    manifest["label"] = manifest["label"].astype(int)
    if len(manifest) != 26_527 or set(manifest["label"]) != {0, 1, 2}:
        raise ValueError("Invalid training manifest")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, valid_idx = next(
        splitter.split(manifest, manifest["label"], manifest["group"])
    )
    train_frame = manifest.iloc[train_idx].reset_index(drop=True)
    valid_frame = manifest.iloc[valid_idx].reset_index(drop=True)
    counts = np.bincount(train_frame["label"], minlength=NUM_CLASSES)
    class_weights = len(train_frame) / (NUM_CLASSES * counts)

    device = torch.device("cuda")
    validation_model = make_model().to(device)
    validation_history, best = train_epochs(
        validation_model,
        make_loader(train_frame, train_transform, True, SEED),
        MAX_EPOCHS,
        class_weights,
        device,
        make_loader(valid_frame, eval_transform, False, SEED),
    )
    if best is None:
        raise RuntimeError("Validation did not produce a checkpoint")
    best_epoch = int(best["epoch"])
    del validation_model
    torch.cuda.empty_cache()

    seed_everything(SEED + 100)
    full_counts = np.bincount(manifest["label"], minlength=NUM_CLASSES)
    full_class_weights = len(manifest) / (NUM_CLASSES * full_counts)
    final_model = make_model().to(device)
    final_history, _ = train_epochs(
        final_model,
        make_loader(manifest, train_transform, True, SEED + 100),
        best_epoch,
        full_class_weights,
        device,
    )
    torch.save(
        {
            "model": final_model.state_dict(),
            "best_epoch": best_epoch,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
        },
        final_path,
    )
    dataset_fingerprint = hashlib.sha256(
        "".join(
            f"{row.path}|{row.label}|{row.group}\n" for row in manifest.itertuples()
        ).encode()
    ).hexdigest()
    result = {
        "cached": False,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dataset_fingerprint": dataset_fingerprint,
        "seed": SEED,
        "best_epoch": best_epoch,
        "validation_best": best,
        "validation_history": validation_history,
        "final_history": final_history,
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cache_volume.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.function(
    image=image,
    gpu=GPU,
    timeout=2 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def infer():
    import io
    import json
    import os
    import random

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageFile, ImageOps
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import AutoModelForImageClassification

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    class ResizePad:
        def __call__(self, source):
            resized = ImageOps.contain(
                source,
                (IMAGE_SIZE, IMAGE_SIZE),
                method=Image.Resampling.BICUBIC,
            )
            canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (124, 116, 104))
            offset = (
                (IMAGE_SIZE - resized.width) // 2,
                (IMAGE_SIZE - resized.height) // 2,
            )
            canvas.paste(resized, offset)
            return canvas

    transform = transforms.Compose(
        [
            ResizePad(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    test_paths = sorted(
        (path for path in (Path(DATA_ROOT) / "test").iterdir() if path.is_file()),
        key=lambda path: int(path.stem),
    )
    if len(test_paths) != 1_458:
        raise ValueError(f"Expected 1458 test images, found {len(test_paths)}")

    class TestDataset(Dataset):
        def __init__(self, flip=False):
            self.flip = flip

        def __len__(self):
            return len(test_paths)

        def __getitem__(self, index):
            with Image.open(test_paths[index]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                if self.flip:
                    image = ImageOps.mirror(image)
                return transform(image)

    device = torch.device("cuda")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)
    checkpoint = torch.load(
        Path(RUN_ROOT) / "final_model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    @torch.inference_mode()
    def predict(flip):
        loader = DataLoader(
            TestDataset(flip),
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )
        outputs = []
        for images in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pixel_values=images).logits
            outputs.append(logits.float().cpu().numpy())
        return np.concatenate(outputs)

    logits = (predict(False) + predict(True)) / 2
    predictions = logits.argmax(axis=1).astype(int)
    ids = [int(path.stem) for path in test_paths]
    prediction_map = dict(zip(ids, predictions, strict=True))
    submission = pd.read_csv(Path(DATA_ROOT) / "submission.csv")[["id"]]
    submission["predicted"] = submission["id"].astype(int).map(prediction_map)
    if submission["predicted"].isna().any() or len(submission) != 1_458:
        raise ValueError("Invalid submission IDs")
    submission["predicted"] = submission["predicted"].astype(int)

    output_path = Path(RUN_ROOT) / "submission_convnextv2_finetuned.csv"
    submission.to_csv(output_path, index=False)
    metadata = {
        "rows": len(submission),
        "best_epoch": int(checkpoint["best_epoch"]),
        "class_counts": {
            str(label): int(count)
            for label, count in submission["predicted"].value_counts().sort_index().items()
        },
    }
    (Path(RUN_ROOT) / "inference.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    cache_volume.commit()
    buffer = io.StringIO()
    submission.to_csv(buffer, index=False)
    print(json.dumps(metadata, indent=2), flush=True)
    return buffer.getvalue()


@app.local_entrypoint()
def main(action: str = "all", output_dir: str = ".", force: bool = False):
    if action not in {"train", "infer", "all"}:
        raise ValueError("action must be train, infer, or all")
    if action in {"train", "all"}:
        manifest = prepare_manifest.remote()
        print(f"Manifest: {manifest['rows']} rows")
        metrics = train.remote(force=force)
        print(
            f"Best epoch: {metrics['best_epoch']}; "
            f"validation Macro-F1: {metrics['validation_best']['macro_f1']:.6f}"
        )
    if action in {"infer", "all"}:
        csv_text = infer.remote()
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        output_path = destination / "submission_convnextv2_finetuned.csv"
        output_path.write_text(csv_text, encoding="utf-8")
        print(f"Wrote {output_path.resolve()}")
