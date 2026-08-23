"""Unified DINOv3 ConvNeXt-L run 06 for BDC2026 on Modal.

One command trains, infers, runs every ablation, and downloads all artifacts:
  modal run artifacts/augmented_shape_encoder/modal_run06_pipeline.py --action all
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

APP_NAME = "bdc2026-dinov3-convnext-bdc-aug-all"
GPU = "A100-80GB"
MODEL_NAME = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
MODEL_REVISION = "e959efa74c867491dcfe3ec3e4f97382e39025b3"
RUN_NAME = "dinov3_convnext_large_bdc_augmentation_224_seed2026"
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
LABEL_SMOOTHING = 0.10
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 1.0
MIX_PROBABILITY = 0.8
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
    from torchvision.transforms import v2
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
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.55, 1.0),
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4)], p=0.5
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=0.1,
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"
            ),
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
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float32, device=device
        )
        mixup = v2.MixUp(alpha=MIXUP_ALPHA, num_classes=NUM_CLASSES)
        cutmix = v2.CutMix(alpha=CUTMIX_ALPHA, num_classes=NUM_CLASSES)

        def mix_batch(images, targets):
            if torch.rand((), device=device).item() < MIX_PROBABILITY:
                transform = (
                    cutmix if torch.rand((), device=device).item() < 0.5 else mixup
                )
                images, targets = transform(images, targets)
            else:
                targets = torch.nn.functional.one_hot(
                    targets, NUM_CLASSES
                ).float()
            targets = (
                targets * (1.0 - LABEL_SMOOTHING)
                + LABEL_SMOOTHING / NUM_CLASSES
            )
            return images, targets

        def criterion(logits, targets):
            log_probabilities = torch.nn.functional.log_softmax(logits, dim=1)
            return -(
                targets * class_weights_tensor * log_probabilities
            ).sum(dim=1).mean()
        history = []
        best_epoch, best_f1, stale = epochs, -1.0, 0

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step, (images, targets) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                images, targets = mix_batch(images, targets)
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
        "augmentation": {
            "profile": "bdc-convnextv2-base.ipynb",
            "random_resized_crop_scale": [0.55, 1.0],
            "random_resized_crop_ratio": [0.75, 1.33],
            "horizontal_flip": 0.5,
            "vertical_flip": 0.2,
            "color_jitter": 0.4,
            "grayscale": 0.1,
            "gaussian_blur": 0.1,
            "randaugment": "rand-m9",
            "random_erasing": 0.25,
            "mixup_alpha": MIXUP_ALPHA,
            "cutmix_alpha": CUTMIX_ALPHA,
            "mix_probability": MIX_PROBABILITY,
        },
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
    import time

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageOps
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModel

    started_at = time.perf_counter()
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
    forward_started_at = time.perf_counter()
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = 0.5 * (model(images) + model(torch.flip(images, dims=[3])))
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    forward_seconds = time.perf_counter() - forward_started_at
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
    output_path = run_root / "submission_dinov3_convnext_bdc_aug.csv"
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
        "forward_seconds": forward_seconds,
        "end_to_end_seconds": time.perf_counter() - started_at,
    }
    (run_root / "inference.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    cache_volume.commit()
    buffer = io.StringIO()
    submission.to_csv(buffer, index=False)
    print(metadata, flush=True)
    return {
        "submission": buffer.getvalue(),
        "probabilities": (run_root / "test_probabilities.npz").read_bytes(),
        "metadata": metadata,
    }


@app.function(
    image=image,
    gpu=GPU,
    timeout=12 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def run_ablation(method: str, force: bool = False):
    import io
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
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import transforms
    from torchvision.transforms import v2
    from transformers import AutoImageProcessor, AutoModel, get_cosine_schedule_with_warmup

    if method not in {"tta", "knn", "hard-negative"}:
        raise ValueError("method must be tta, knn, or hard-negative")

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
    run_root = Path(RUN_ROOT)
    output_root = run_root / "ablations" / method.replace("-", "_")
    metrics_path = output_root / "metrics.json"
    submission_path = output_root / f"submission_{method.replace('-', '_')}.csv"
    probability_path = output_root / "test_probabilities.npz"

    def payload():
        return {
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
            "submission": submission_path.read_text(encoding="utf-8"),
            "probabilities": probability_path.read_bytes(),
        }

    if (
        metrics_path.exists()
        and submission_path.exists()
        and probability_path.exists()
        and not force
    ):
        cached_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if {"inference_seconds", "runtime_seconds"} <= cached_metrics.keys():
            return payload()

    started_at = time.perf_counter()
    manifest = pd.read_csv(Path(DATA_ROOT) / "train_manifest.csv")
    manifest["label"] = manifest["label"].astype(int)
    if len(manifest) != 26_527 or set(manifest["label"]) != {0, 1, 2}:
        raise ValueError("Invalid training manifest; run the base pipeline first")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, valid_idx = next(splitter.split(manifest, manifest["label"], manifest["group"]))
    train_frame = manifest.iloc[train_idx].reset_index(drop=True)
    valid_frame = manifest.iloc[valid_idx].reset_index(drop=True)
    test_paths = sorted(
        (path for path in (Path(DATA_ROOT) / "test").iterdir() if path.is_file()),
        key=lambda path: int(path.stem),
    )
    if len(test_paths) != 1_458:
        raise ValueError(f"Expected 1458 test images, found {len(test_paths)}")
    test_frame = pd.DataFrame({"path": [str(path) for path in test_paths], "label": -1})

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)

    class ResizePad:
        def __init__(self, size):
            self.size = size

        def __call__(self, source):
            resized = ImageOps.contain(
                source, (self.size, self.size), method=Image.Resampling.BICUBIC
            )
            canvas = Image.new("RGB", (self.size, self.size), (124, 116, 104))
            canvas.paste(
                resized,
                ((self.size - resized.width) // 2, (self.size - resized.height) // 2),
            )
            return canvas

    def eval_transform(size, flip=False):
        items = [ResizePad(size)]
        if flip:
            items.append(transforms.RandomHorizontalFlip(p=1.0))
        items.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(items)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.55, 1.0),
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4)], p=0.5
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=0.1,
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"
            ),
        ]
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
                value = ImageOps.exif_transpose(source).convert("RGB")
                return self.transform(value), self.labels[index]

    class DINOConvNextClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
            self.classifier = nn.Linear(self.backbone.config.hidden_sizes[-1], NUM_CLASSES)

        def forward(self, pixel_values):
            features = self.backbone(pixel_values=pixel_values).pooler_output
            return self.classifier(features)

    def load_model(checkpoint_name):
        checkpoint = run_root / checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing {checkpoint}; run the base pipeline first")
        model = DINOConvNextClassifier().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        return model

    def loader(frame, transform, sampler=None):
        return DataLoader(
            WasteDataset(frame, transform),
            batch_size=BATCH_SIZE,
            shuffle=False,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )

    @torch.inference_mode()
    def predict(
        model,
        frame,
        size=IMAGE_SIZE,
        flip=False,
        embeddings=False,
        logits_only=False,
    ):
        model.eval()
        probabilities, features = [], []
        batch_size = max(8, int(BATCH_SIZE * (IMAGE_SIZE / size) ** 2))
        data_loader = DataLoader(
            WasteDataset(frame, eval_transform(size, flip)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )
        for images, _ in data_loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pooled = model.backbone(pixel_values=images).pooler_output
                logits = model.classifier(pooled)
            values = logits.float() if logits_only else torch.softmax(logits.float(), dim=1)
            probabilities.append(values.cpu().numpy())
            if embeddings:
                features.append(torch.nn.functional.normalize(pooled.float(), dim=1).cpu().numpy())
        result = np.concatenate(probabilities)
        return (result, np.concatenate(features)) if embeddings else result

    def metrics(labels, probabilities):
        predictions = probabilities.argmax(axis=1)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "class_f1": [
                float(value)
                for value in f1_score(labels, predictions, average=None, labels=[0, 1, 2])
            ],
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        }

    def save_result(probabilities, result_metrics):
        result_metrics["runtime_seconds"] = time.perf_counter() - started_at
        source_ids = [int(path.stem) for path in test_paths]
        source_index = {value: index for index, value in enumerate(source_ids)}
        submission = pd.read_csv(Path(DATA_ROOT) / "submission.csv")[["id"]]
        ids = submission["id"].astype(int).to_numpy()
        ordered = np.stack([probabilities[source_index[value]] for value in ids])
        submission["predicted"] = ordered.argmax(axis=1).astype(int)
        if len(submission) != 1_458 or submission["predicted"].isna().any():
            raise ValueError("Invalid submission")
        output_root.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_path, index=False)
        np.savez(probability_path, ids=ids, probabilities=ordered)
        metrics_path.write_text(json.dumps(result_metrics, indent=2), encoding="utf-8")
        cache_volume.commit()
        return payload()

    if method == "tta":
        validation_model = load_model("best_validation_model.pt")
        components = {}
        for size in (224, 288, 320):
            for flip in (False, True):
                components[(size, flip)] = predict(validation_model, valid_frame, size, flip)
        variants = {
            "baseline_224": [(224, False)],
            "tta_224": [(224, False), (224, True)],
            "tta_224_288": [(224, False), (224, True), (288, False), (288, True)],
            "tta_224_288_320": [
                (224, False), (224, True), (288, False),
                (288, True), (320, False), (320, True),
            ],
        }
        labels = valid_frame["label"].to_numpy()
        grid = {}
        for name, keys in variants.items():
            average = np.mean([components[key] for key in keys], axis=0)
            grid[name] = metrics(labels, average)
        best_name = max(grid, key=lambda name: grid[name]["macro_f1"])
        final_model = load_model("final_model.pt")
        inference_started_at = time.perf_counter()
        test_probabilities = np.mean(
            [predict(final_model, test_frame, *key) for key in variants[best_name]], axis=0
        )
        inference_seconds = time.perf_counter() - inference_started_at
        result_metrics = {
            "method": method,
            "selected_variant": best_name,
            "validation": grid,
            "selection_source": "official train validation split only",
            "inference_seconds": inference_seconds,
        }
        return save_result(test_probabilities, result_metrics)

    def neighbor_probabilities(reference_features, reference_labels, query_features, k, temperature=0.07):
        reference = torch.from_numpy(reference_features).to(device)
        reference_labels = torch.from_numpy(reference_labels.astype(np.int64)).to(device)
        outputs = []
        for start in range(0, len(query_features), 256):
            query = torch.from_numpy(query_features[start : start + 256]).to(device)
            similarities, indices = (query @ reference.T).topk(k, dim=1)
            weights = torch.softmax(similarities / temperature, dim=1)
            labels = reference_labels[indices]
            probabilities = (
                torch.nn.functional.one_hot(labels, NUM_CLASSES)
                * weights.unsqueeze(-1)
            ).sum(dim=1)
            outputs.append(probabilities.cpu().numpy())
        return np.concatenate(outputs)

    if method == "knn":
        validation_model = load_model("best_validation_model.pt")
        train_probabilities, train_features = predict(
            validation_model, train_frame, embeddings=True
        )
        valid_probabilities, valid_features = predict(
            validation_model, valid_frame, embeddings=True
        )
        del train_probabilities
        labels = valid_frame["label"].to_numpy()
        reference_labels = train_frame["label"].to_numpy()
        grid = []
        best = {"macro_f1": -1.0}
        for k in (5, 11, 21, 31):
            neighbor = neighbor_probabilities(train_features, reference_labels, valid_features, k)
            for alpha in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
                blended = (1.0 - alpha) * valid_probabilities + alpha * neighbor
                row = {"k": k, "alpha": alpha, **metrics(labels, blended)}
                grid.append(row)
                if row["macro_f1"] > best["macro_f1"]:
                    best = row

        final_model = load_model("final_model.pt")
        _, full_features = predict(final_model, manifest, embeddings=True)
        inference_started_at = time.perf_counter()
        test_probabilities, test_features = predict(final_model, test_frame, embeddings=True)
        neighbor = neighbor_probabilities(
            full_features,
            manifest["label"].to_numpy(),
            test_features,
            int(best["k"]),
        )
        test_probabilities = (
            (1.0 - best["alpha"]) * test_probabilities + best["alpha"] * neighbor
        )
        inference_seconds = time.perf_counter() - inference_started_at
        result_metrics = {
            "method": method,
            "selected": best,
            "validation_grid": grid,
            "selection_source": "official train validation split only",
            "inference_seconds": inference_seconds,
        }
        return save_result(test_probabilities, result_metrics)

    def hard_indices(labels, probabilities):
        labels = np.asarray(labels, dtype=np.int64)
        true_probability = probabilities[np.arange(len(labels)), labels]
        masked = probabilities.copy()
        masked[np.arange(len(labels)), labels] = -1.0
        margin = true_probability - masked.max(axis=1)
        candidates = np.flatnonzero(np.isin(labels, [0, 2]))
        count = max(1, round(0.10 * len(candidates)))
        return candidates[np.argsort(margin[candidates])[:count]]

    def optimizer_for(model):
        grouped = {}
        depth = model.backbone.config.num_stages
        for name, parameter in model.named_parameters():
            if name.startswith("classifier"):
                learning_rate = 2e-5
            else:
                match = re.search(r"backbone\.stages\.(\d+)\.", name)
                learning_rate = 2e-6 * (0.8 ** (depth - 1 - int(match.group(1)))) if match else 2e-6
            decay = 0.0 if parameter.ndim == 1 or name.endswith(".bias") else 0.05
            grouped.setdefault((learning_rate, decay), []).append(parameter)
        return torch.optim.AdamW(
            [
                {"params": values, "lr": learning_rate, "weight_decay": decay}
                for (learning_rate, decay), values in grouped.items()
            ]
        )

    def fine_tune(model, frame, sample_weights, epochs, valid=None, checkpoint=None):
        generator = torch.Generator().manual_seed(SEED + 404)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        train_loader = loader(frame, train_transform, sampler)
        optimizer = optimizer_for(model)
        updates_per_epoch = math.ceil(len(train_loader) / 2)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, round(updates_per_epoch * epochs * 0.05)),
            num_training_steps=updates_per_epoch * epochs,
        )
        counts = np.bincount(frame["label"], minlength=NUM_CLASSES)
        class_weights = len(frame) / (NUM_CLASSES * counts)
        class_weights_tensor = torch.tensor(
            class_weights, dtype=torch.float32, device=device
        )
        mixup = v2.MixUp(alpha=MIXUP_ALPHA, num_classes=NUM_CLASSES)
        cutmix = v2.CutMix(alpha=CUTMIX_ALPHA, num_classes=NUM_CLASSES)

        def mix_batch(images, targets):
            if torch.rand((), device=device).item() < MIX_PROBABILITY:
                transform = (
                    cutmix if torch.rand((), device=device).item() < 0.5 else mixup
                )
                images, targets = transform(images, targets)
            else:
                targets = torch.nn.functional.one_hot(
                    targets, NUM_CLASSES
                ).float()
            targets = (
                targets * (1.0 - LABEL_SMOOTHING)
                + LABEL_SMOOTHING / NUM_CLASSES
            )
            return images, targets

        def criterion(logits, targets):
            log_probabilities = torch.nn.functional.log_softmax(logits, dim=1)
            return -(
                targets * class_weights_tensor * log_probabilities
            ).sum(dim=1).mean()
        history = []
        best = None
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            for step, (images, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                images, targets = mix_batch(images, targets)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw_loss = criterion(model(images), targets)
                (raw_loss / 2).backward()
                loss_sum += raw_loss.item()
                if (step + 1) % 2 == 0 or step + 1 == len(train_loader):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            row = {"epoch": epoch, "train_loss": loss_sum / len(train_loader)}
            if valid is not None:
                valid_probabilities = predict(model, valid)
                row.update(metrics(valid["label"].to_numpy(), valid_probabilities))
                if best is None or row["macro_f1"] > best["macro_f1"]:
                    best = row.copy()
                    torch.save(model.state_dict(), checkpoint)
            history.append(row)
            print(json.dumps(row), flush=True)
        return history, best

    validation_model = load_model("best_validation_model.pt")
    train_probabilities = predict(validation_model, train_frame)
    valid_probabilities = predict(validation_model, valid_frame)
    baseline = metrics(valid_frame["label"].to_numpy(), valid_probabilities)
    selected = hard_indices(train_frame["label"].to_numpy(), train_probabilities)
    sample_weights = np.ones(len(train_frame), dtype=np.float64)
    sample_weights[selected] = 2.0
    output_root.mkdir(parents=True, exist_ok=True)
    validation_checkpoint = output_root / "best_validation_model.pt"
    history, best = fine_tune(
        validation_model,
        train_frame,
        sample_weights,
        2,
        valid_frame,
        validation_checkpoint,
    )
    selected_epoch = int(best["epoch"]) if best["macro_f1"] > baseline["macro_f1"] else 0

    final_model = load_model("final_model.pt")
    full_probabilities = predict(final_model, manifest)
    full_selected = hard_indices(manifest["label"].to_numpy(), full_probabilities)
    full_weights = np.ones(len(manifest), dtype=np.float64)
    full_weights[full_selected] = 2.0
    if selected_epoch:
        fine_tune(final_model, manifest, full_weights, selected_epoch)
        torch.save(final_model.state_dict(), output_root / "final_model.pt")
    inference_started_at = time.perf_counter()
    test_logits = 0.5 * (
        predict(final_model, test_frame, flip=False, logits_only=True)
        + predict(final_model, test_frame, flip=True, logits_only=True)
    )
    test_probabilities = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    inference_seconds = time.perf_counter() - inference_started_at
    result_metrics = {
        "method": method,
        "baseline_validation": baseline,
        "validation_history": history,
        "selected_epoch": selected_epoch,
        "hard_fraction": 0.10,
        "hard_weight": 2.0,
        "hard_train_samples": int(len(selected)),
        "hard_full_samples": int(len(full_selected)),
        "augmentation_profile": "bdc-convnextv2-base.ipynb",
        "mixup_alpha": MIXUP_ALPHA,
        "cutmix_alpha": CUTMIX_ALPHA,
        "mix_probability": MIX_PROBABILITY,
        "selection_source": "official train validation split only",
        "inference_seconds": inference_seconds,
    }
    return save_result(test_probabilities, result_metrics)


def write_remote_result(result, destination):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(
        __import__("json").dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    method = result["metrics"]["method"].replace("-", "_")
    (destination / f"submission_{method}.csv").write_text(
        result["submission"], encoding="utf-8"
    )
    (destination / "test_probabilities.npz").write_bytes(result["probabilities"])


def ensemble_local(output_root):
    import json
    import time

    import numpy as np
    import pandas as pd

    started_at = time.perf_counter()
    root = Path.cwd()
    sources = {
        "dinov3_convnext": root / "artifacts/augmented_shape_encoder/test_probabilities.npz",
        "convnextv2_global_local": root / "run/02_convnextv2-L-global-local/test_probabilities.npz",
        "dinov3_vitb": root / "run/03_dinov3vitB/test_probabilities.npz",
        "tta": output_root.parent / "01_tta/test_probabilities.npz",
        "knn": output_root.parent / "02_knn/test_probabilities.npz",
    }
    loaded = {}
    ids = None
    for name, path in sources.items():
        if not path.exists():
            continue
        with np.load(path) as values:
            current_ids = values["ids"].astype(int)
            probabilities = values["probabilities"].astype(np.float64)
        if ids is None:
            ids = current_ids
        elif not np.array_equal(ids, current_ids):
            raise ValueError(f"ID order differs in {path}")
        loaded[name] = probabilities
    required = {"dinov3_convnext", "convnextv2_global_local", "dinov3_vitb"}
    missing = required - loaded.keys()
    if missing:
        raise FileNotFoundError(f"Missing probability files: {sorted(missing)}")

    variants = {
        "three_models": {
            "dinov3_convnext": 0.50,
            "convnextv2_global_local": 0.30,
            "dinov3_vitb": 0.20,
        }
    }
    if {"tta", "knn"} <= loaded.keys():
        variants["tta_knn_models"] = {
            "tta": 0.40,
            "knn": 0.20,
            "convnextv2_global_local": 0.25,
            "dinov3_vitb": 0.15,
        }

    output_root.mkdir(parents=True, exist_ok=True)
    template = pd.read_csv(root / "BDC2026/submission.csv")[["id"]]
    if not np.array_equal(template["id"].astype(int).to_numpy(), ids):
        raise ValueError("Submission template ID order differs from probability files")
    metadata = {"selection": "fixed weights; no test labels used", "variants": {}}
    for name, weights in variants.items():
        probabilities = sum(loaded[source] * weight for source, weight in weights.items())
        submission = template.copy()
        submission["predicted"] = probabilities.argmax(axis=1).astype(int)
        submission.to_csv(output_root / f"submission_ensemble_{name}.csv", index=False)
        np.savez(
            output_root / f"test_probabilities_{name}.npz",
            ids=ids,
            probabilities=probabilities,
        )
        metadata["variants"][name] = {"weights": weights}
    metadata["inference_seconds"] = time.perf_counter() - started_at
    (output_root / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


@app.local_entrypoint()
def main(
    action: str = "all",
    output_dir: str = "artifacts/augmented_shape_encoder",
    force: bool = False,
):
    import json

    if action not in {"train", "infer", "ablations", "all"}:
        raise ValueError("action must be train, infer, ablations, or all")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = {"action": action, "inference_seconds": {}}

    if action in {"train", "all"}:
        manifest = prepare_manifest.remote()
        print(f"Manifest: {manifest['rows']} rows")
        metrics = train.remote(force=force)
        (destination / "training_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        summary["training_seconds"] = metrics["training_seconds"]
        print(f"Validation Macro-F1: {metrics['best_validation_macro_f1']:.6f}")

    if action in {"infer", "all"}:
        result = infer.remote()
        (destination / "submission_dinov3_convnext_bdc_aug.csv").write_text(
            result["submission"], encoding="utf-8"
        )
        (destination / "test_probabilities.npz").write_bytes(
            result["probabilities"]
        )
        (destination / "inference.json").write_text(
            json.dumps(result["metadata"], indent=2), encoding="utf-8"
        )
        summary["inference_seconds"]["base_forward"] = result["metadata"][
            "forward_seconds"
        ]
        summary["inference_seconds"]["base_end_to_end"] = result["metadata"][
            "end_to_end_seconds"
        ]
        print(f"Base inference: {result['metadata']['forward_seconds']:.2f}s")

    if action in {"ablations", "all"}:
        ablation_root = destination / "ablation"
        folders = {
            "tta": ablation_root / "01_tta",
            "knn": ablation_root / "02_knn",
            "hard-negative": ablation_root / "03_hard_negative",
        }
        for method, folder in folders.items():
            result = run_ablation.remote(method, force=force)
            write_remote_result(result, folder)
            seconds = result["metrics"]["inference_seconds"]
            summary["inference_seconds"][method] = seconds
            print(f"{method} inference: {seconds:.2f}s")
        ensemble_metrics = ensemble_local(ablation_root / "04_ensemble")
        summary["inference_seconds"]["ensemble"] = ensemble_metrics[
            "inference_seconds"
        ]
        print(
            f"ensemble inference: "
            f"{ensemble_metrics['inference_seconds']:.2f}s"
        )

    (destination / "stage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"All requested artifacts saved in {destination.resolve()}")
