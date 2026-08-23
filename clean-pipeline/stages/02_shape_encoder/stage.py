"""Full DINOv3 ConvNeXt-L fine-tuning for BDC2026 on Modal.

Run training and inference:
  modal run modal_dinov3_convnext_finetune_pipeline.py --action all
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


shared_app = Path(__file__).with_name("modal_backbone_app.py")
if not shared_app.exists():
    shared_app = Path.cwd() / "modal_backbone_app.py"
image = image.add_local_file(shared_app, "/root/modal_backbone_app.py")

APP_NAME = "bdc2026-dinov3-convnext-finetune"
GPU = "A100-80GB"
MODEL_NAME = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
MODEL_REVISION = "e959efa74c867491dcfe3ec3e4f97382e39025b3"
RUN_NAME = "dinov3_convnext_large_full_finetune_224_seed2026"
RUN_ROOT = f"{CACHE_ROOT}/runs/{RUN_NAME}"

NUM_CLASSES = 3
IMAGE_SIZE = 224
BATCH_SIZE = 32
GRAD_ACCUMULATION = 2
MAX_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 3
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
LAYER_DECAY = 0.8
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
EMA_DECAY = 0.999
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

    root = Path(DATA_ROOT)
    path = root / "train_manifest.csv"
    frame = None
    if path.exists():
        try:
            candidate = pd.read_csv(path)
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
            0: root / "train" / "0_Recyclable",
            1: root / "train" / "1_Electronic",
            2: root / "train" / "2_Organic",
        }
        for label, folder in folders.items():
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing dataset folder: {folder}")
            for image_path in sorted(p for p in folder.iterdir() if p.is_file()):
                rows.append(
                    {
                        "path": str(image_path),
                        "label": label,
                        "group": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    }
                )
        frame = pd.DataFrame(rows)
        frame.to_csv(path, index=False)
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
    import re
    import time

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageFile, ImageOps
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from torch import nn
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModel, get_cosine_schedule_with_warmup

    run_root = Path(RUN_ROOT)
    final_path = run_root / "final_model.pt"
    metrics_path = run_root / "training_metrics.json"
    if final_path.exists() and metrics_path.exists() and not force:
        result = json.loads(metrics_path.read_text(encoding="utf-8"))
        result["cached"] = True
        print(json.dumps(result, indent=2), flush=True)
        return result

    started_at = time.monotonic()

    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    device = torch.device("cuda")
    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION
    )
    normalize = transforms.Normalize(
        mean=processor.image_mean, std=processor.image_std
    )

    class ResizePad:
        def __call__(self, source):
            resized = ImageOps.contain(
                source,
                (IMAGE_SIZE, IMAGE_SIZE),
                method=Image.Resampling.BICUBIC,
            )
            canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (124, 116, 104))
            canvas.paste(
                resized,
                ((IMAGE_SIZE - resized.width) // 2, (IMAGE_SIZE - resized.height) // 2),
            )
            return canvas

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
                image_value = ImageOps.exif_transpose(source).convert("RGB")
                return self.transform(image_value), self.labels[index]

    class DINOConvNextClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(
                MODEL_NAME, revision=MODEL_REVISION
            )
            hidden = self.backbone.config.hidden_sizes[-1]
            self.classifier = nn.Linear(hidden, NUM_CLASSES)

        def forward(self, pixel_values):
            pooled = self.backbone(pixel_values=pixel_values).pooler_output
            return self.classifier(pooled)

    def make_loader(frame, transform, shuffle, seed):
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            WasteDataset(frame, transform),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            generator=generator,
        )

    def optimizer_for(model):
        depth = model.backbone.config.num_stages
        grouped = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("classifier"):
                lr = HEAD_LR
            else:
                match = re.search(r"backbone\.stages\.(\d+)\.", name)
                if match:
                    lr = BACKBONE_LR * LAYER_DECAY ** (
                        depth - 1 - int(match.group(1))
                    )
                elif "backbone.embeddings" in name:
                    lr = BACKBONE_LR * LAYER_DECAY**depth
                else:
                    lr = BACKBONE_LR
            decay = 0.0 if parameter.ndim == 1 or name.endswith(".bias") else WEIGHT_DECAY
            grouped.setdefault((lr, decay), []).append(parameter)
        return torch.optim.AdamW(
            [
                {"params": parameters, "lr": lr, "weight_decay": decay}
                for (lr, decay), parameters in grouped.items()
            ]
        )

    @torch.inference_mode()
    def evaluate(model, loader):
        model.eval()
        labels, predictions = [], []
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
            labels.extend(targets.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
        per_class = f1_score(labels, predictions, average=None, labels=[0, 1, 2])
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "class_f1": [float(value) for value in per_class],
        }

    def fit(model, loader, epochs, class_weights, valid_loader=None, checkpoint=None):
        model = model.to(device)
        ema = AveragedModel(
            model,
            device=device,
            multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY),
            use_buffers=True,
        )
        optimizer = optimizer_for(model)
        updates_per_epoch = math.ceil(len(loader) / GRAD_ACCUMULATION)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, math.ceil(updates_per_epoch * epochs * 0.05)),
            num_training_steps=updates_per_epoch * epochs,
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=LABEL_SMOOTHING,
        )
        history = []
        best_epoch, best_f1, stale = epochs, -1.0, 0

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step, (images, targets) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw_loss = criterion(model(images), targets)
                    loss = raw_loss / GRAD_ACCUMULATION
                loss.backward()
                total_loss += raw_loss.item()
                should_update = (
                    (step + 1) % GRAD_ACCUMULATION == 0
                    or step + 1 == len(loader)
                )
                if should_update:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    ema.update_parameters(model)
                    optimizer.zero_grad(set_to_none=True)

            row = {"epoch": epoch, "train_loss": total_loss / len(loader)}
            if valid_loader is not None:
                row.update(evaluate(ema.module, valid_loader))
                if row["macro_f1"] > best_f1:
                    best_f1, best_epoch, stale = row["macro_f1"], epoch, 0
                    torch.save(ema.module.state_dict(), checkpoint)
                else:
                    stale += 1
            history.append(row)
            print(json.dumps(row), flush=True)
            if valid_loader is not None and stale >= EARLY_STOPPING_PATIENCE:
                break

        if valid_loader is None:
            torch.save(ema.module.state_dict(), checkpoint)
        del ema, optimizer, scheduler
        return history, best_epoch

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

    run_root.mkdir(parents=True, exist_ok=True)
    validation_path = run_root / "best_validation_model.pt"
    validation_history, best_epoch = fit(
        DINOConvNextClassifier(),
        make_loader(train_frame, train_transform, True, SEED),
        MAX_EPOCHS,
        class_weights,
        make_loader(valid_frame, eval_transform, False, SEED),
        validation_path,
    )

    torch.cuda.empty_cache()
    full_counts = np.bincount(manifest["label"], minlength=NUM_CLASSES)
    full_weights = len(manifest) / (NUM_CLASSES * full_counts)
    final_history, _ = fit(
        DINOConvNextClassifier(),
        make_loader(manifest, train_transform, True, SEED + 100),
        best_epoch,
        full_weights,
        checkpoint=final_path,
    )

    summary = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUMULATION,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
        "max_epochs": MAX_EPOCHS,
        "backbone_lr": BACKBONE_LR,
        "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": max(
            row["macro_f1"] for row in validation_history
        ),
        "best_validation": max(
            validation_history, key=lambda row: row["macro_f1"]
        ),
        "validation_history": validation_history,
        "full_training_history": final_history,
        "dataset_fingerprint": hashlib.sha256(
            "".join(
                f"{row.path}|{row.label}|{row.group}\n"
                for row in manifest.itertuples()
            ).encode()
        ).hexdigest(),
        "class_weighting": "inverse_frequency",
        "ema_decay": EMA_DECAY,
        "layer_decay": LAYER_DECAY,
        "training_seconds": time.monotonic() - started_at,
        "cached": False,
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cache_volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(
    image=image,
    gpu=GPU,
    timeout=3 * 60 * 60,
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
    from PIL import Image, ImageOps
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModel

    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    class DINOConvNextClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(
                MODEL_NAME, revision=MODEL_REVISION
            )
            hidden = self.backbone.config.hidden_sizes[-1]
            self.classifier = nn.Linear(hidden, NUM_CLASSES)

        def forward(self, pixel_values):
            pooled = self.backbone(pixel_values=pixel_values).pooler_output
            return self.classifier(pooled)

    class ResizePad:
        def __call__(self, source):
            resized = ImageOps.contain(
                source,
                (IMAGE_SIZE, IMAGE_SIZE),
                method=Image.Resampling.BICUBIC,
            )
            canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (124, 116, 104))
            canvas.paste(
                resized,
                ((IMAGE_SIZE - resized.width) // 2, (IMAGE_SIZE - resized.height) // 2),
            )
            return canvas

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION
    )
    transform = transforms.Compose(
        [
            ResizePad(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=processor.image_mean, std=processor.image_std
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
        def __len__(self):
            return len(test_paths)

        def __getitem__(self, index):
            with Image.open(test_paths[index]) as source:
                return transform(ImageOps.exif_transpose(source).convert("RGB"))

    final_path = Path(RUN_ROOT) / "final_model.pt"
    if not final_path.exists():
        raise FileNotFoundError("No final checkpoint; run --action train first")
    device = torch.device("cuda")
    model = DINOConvNextClassifier().to(device)
    model.load_state_dict(torch.load(final_path, map_location=device, weights_only=True))
    model.eval()
    loader = DataLoader(
        TestDataset(),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    probabilities = []
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = 0.5 * (model(images) + model(torch.flip(images, dims=[3])))
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    predictions = probabilities.argmax(axis=1).astype(int)

    prediction_map = dict(
        zip((int(path.stem) for path in test_paths), predictions, strict=True)
    )
    submission = pd.read_csv(Path(DATA_ROOT) / "submission.csv")[["id"]]
    submission["predicted"] = submission["id"].astype(int).map(prediction_map)
    if submission["predicted"].isna().any() or len(submission) != 1_458:
        raise ValueError("Invalid submission IDs")
    submission["predicted"] = submission["predicted"].astype(int)

    run_root = Path(RUN_ROOT)
    output_path = run_root / "submission_dinov3_convnext_finetuned.csv"
    submission.to_csv(output_path, index=False)
    np.savez(
        run_root / "test_probabilities.npz",
        ids=submission["id"].astype(int).to_numpy(),
        probabilities=probabilities,
    )
    metadata = {
        "rows": len(submission),
        "class_counts": {
            str(label): int(count)
            for label, count in submission["predicted"].value_counts().sort_index().items()
        },
    }
    (run_root / "inference.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    cache_volume.commit()
    buffer = io.StringIO()
    submission.to_csv(buffer, index=False)
    print(metadata, flush=True)
    return buffer.getvalue()


@app.local_entrypoint()
def main(
    action: str = "all",
    output_dir: str = "artifacts/shape_encoder",
    force: bool = False,
):
    if action not in {"train", "infer", "all"}:
        raise ValueError("action must be train, infer, or all")
    if action in {"train", "all"}:
        manifest = prepare_manifest.remote()
        print(f"Manifest: {manifest['rows']} rows")
        metrics = train.remote(force=force)
        print(f"Validation Macro-F1: {metrics['best_validation_macro_f1']:.6f}")
    if action in {"infer", "all"}:
        csv_text = infer.remote()
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        output_path = destination / "submission_dinov3_convnext_finetuned.csv"
        output_path.write_text(csv_text, encoding="utf-8")
        print(f"Wrote {output_path.resolve()}")
