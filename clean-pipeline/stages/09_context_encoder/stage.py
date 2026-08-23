"""Partial SigLIP fine-tune + conservative OOF calibrated blend.

Run from project root:
  modal run artifacts/context_encoder/modal_run13_pipeline.py

Use --force only when retraining is intentional.
"""

from pathlib import Path
import sys

import modal


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (
        candidate
        for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents, Path.cwd())
        if (candidate / "modal_backbone_app.py").is_file()
    ),
    Path.cwd(),
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modal_backbone_app import (  # noqa: E402
    CACHE_ROOT,
    DATA_ROOT,
    HF_CACHE,
    SEED,
    cache_volume,
    data_volume,
    hf_secret,
    image,
)


image = image.pip_install("timm==1.0.27").add_local_file(
    PROJECT_ROOT / "modal_backbone_app.py", "/root/modal_backbone_app.py"
)

APP_NAME = "bdc2026-context-encoder"
GPU = "A100-80GB"
MODEL_REPO = "timm/vit_so400m_patch14_siglip_378.webli_ft_in1k"
MODEL_ARCH = "vit_so400m_patch14_siglip_378.webli_ft_in1k"
MODEL_REVISION = "efb07b5711bda4e3eb24630db9879e0832615ad4"
RUN11_NAME = "class_boundary_calibration_seed2026"
RUN11_ROOT = f"{CACHE_ROOT}/runs/{RUN11_NAME}"
RUN_NAME = "context_encoder_seed2026"
RUN_ROOT = f"{CACHE_ROOT}/runs/{RUN_NAME}"

NUM_CLASSES = 3
IMAGE_SIZE = 378
BATCH_SIZE = 24
GRAD_ACCUMULATION = 2
HEAD_WARMUP_EPOCHS = 1
MAX_PARTIAL_EPOCHS = 4
EARLY_STOPPING_PATIENCE = 2
UNFROZEN_BLOCKS = 4
HEAD_LR = 3e-5
BACKBONE_LR = 2e-6
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
NUM_WORKERS = 8
ALPHAS = tuple(round(index * 0.025, 3) for index in range(21))
METHOD_VERSION = 1

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=49152,
    timeout=6 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def run_pipeline(force: bool = False):
    import hashlib
    import io
    import json
    import math
    import os
    import random
    import time

    import numpy as np
    import pandas as pd
    import timm
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image, ImageFile, ImageOps
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    started_at = time.perf_counter()
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TIMM_FUSED_ATTN"] = "0"
    timm.layers.set_fused_attn(False)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    seed_all(SEED)
    device = torch.device("cuda")
    run_root = Path(RUN_ROOT)
    metrics_path = run_root / "metrics.json"
    submission_path = run_root / "submission_context_ensemble.csv"
    test_probability_path = run_root / "test_probabilities.npz"
    validation_csv_path = run_root / "validation_predictions.csv"
    validation_probability_path = run_root / "validation_probabilities.npz"

    def payload(metrics):
        return {
            "metrics": metrics,
            "submission": submission_path.read_text(encoding="utf-8"),
            "test_probabilities": test_probability_path.read_bytes(),
            "validation_csv": validation_csv_path.read_text(encoding="utf-8"),
            "validation_probabilities": validation_probability_path.read_bytes(),
        }

    required = (
        metrics_path,
        submission_path,
        test_probability_path,
        validation_csv_path,
        validation_probability_path,
    )
    if all(path.exists() for path in required) and not force:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("method_version") == METHOD_VERSION:
            metrics["cached"] = True
            return payload(metrics)

    run11_validation_path = Path(RUN11_ROOT) / "validation_probabilities.npz"
    run11_test_path = Path(RUN11_ROOT) / "test_probabilities.npz"
    for path in (run11_validation_path, run11_test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Run11 artifact: {path}. Run Run11 first.")

    manifest = pd.read_csv(Path(DATA_ROOT) / "train_manifest.csv")
    manifest["label"] = manifest["label"].astype(int)
    if (
        len(manifest) != 26_527
        or set(manifest.columns) != {"path", "label", "group"}
        or set(manifest["label"]) != {0, 1, 2}
    ):
        raise ValueError("Invalid training manifest")
    fingerprint = hashlib.sha256(
        "".join(
            f"{row.path}|{row.label}|{row.group}\n" for row in manifest.itertuples()
        ).encode()
    ).hexdigest()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_indices, valid_indices = next(
        splitter.split(manifest, manifest["label"], manifest["group"])
    )
    train_frame = manifest.iloc[train_indices].reset_index(drop=True)
    valid_frame = manifest.iloc[valid_indices].reset_index(drop=True)
    valid_labels = valid_frame["label"].to_numpy(dtype=np.int64)

    checkpoint = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="model.safetensors",
        revision=MODEL_REVISION,
        cache_dir=HF_CACHE,
    )

    def new_model():
        model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=1000)
        timm.models.load_checkpoint(model, checkpoint)
        model.reset_classifier(NUM_CLASSES)
        return model

    probe = new_model()
    data_config = timm.data.resolve_model_data_config(probe)
    mean, std = data_config["mean"], data_config["std"]
    del probe

    class ResizePad:
        def __call__(self, source):
            resized = ImageOps.contain(
                source, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.BICUBIC
            )
            canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (124, 116, 104))
            canvas.paste(
                resized,
                ((IMAGE_SIZE - resized.width) // 2, (IMAGE_SIZE - resized.height) // 2),
            )
            return canvas

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.70, 1.0),
                ratio=(0.80, 1.25),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], 0.35),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(
                p=0.10, scale=(0.02, 0.15), ratio=(0.5, 2.0), value="random"
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [ResizePad(), transforms.ToTensor(), normalize]
    )

    class WasteDataset(Dataset):
        def __init__(self, frame, transform, with_labels=True):
            self.paths = frame["path"].astype(str).tolist()
            self.labels = frame["label"].astype(int).tolist() if with_labels else None
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            with Image.open(self.paths[index]) as source:
                value = self.transform(ImageOps.exif_transpose(source).convert("RGB"))
            return (value, self.labels[index]) if self.labels is not None else value

    def make_loader(frame, transform, shuffle, seed, with_labels=True):
        return DataLoader(
            WasteDataset(frame, transform, with_labels),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            generator=torch.Generator().manual_seed(seed),
        )

    def freeze_for_head(model):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.get_classifier().parameters():
            parameter.requires_grad = True

    def unfreeze_tail(model):
        freeze_for_head(model)
        if not hasattr(model, "blocks") or len(model.blocks) < UNFROZEN_BLOCKS:
            raise ValueError("Unexpected timm model: transformer blocks not found")
        for block in model.blocks[-UNFROZEN_BLOCKS:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for name in ("norm", "fc_norm"):
            module = getattr(model, name, None)
            if isinstance(module, nn.Module):
                for parameter in module.parameters():
                    parameter.requires_grad = True

    def parameter_groups(model, partial):
        if not partial:
            return [{"params": model.get_classifier().parameters(), "lr": HEAD_LR}]
        head_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
        backbone = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids
        ]
        return [
            {"params": backbone, "lr": BACKBONE_LR},
            {"params": model.get_classifier().parameters(), "lr": HEAD_LR},
        ]

    @torch.inference_mode()
    def predict(model, loader, tta=False):
        model.eval()
        probabilities = []
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                if tta:
                    logits = 0.5 * (logits + model(torch.flip(images, dims=[3])))
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(probabilities)

    def score(labels, probabilities):
        predictions = probabilities.argmax(axis=1)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "class_f1": [
                float(value)
                for value in f1_score(labels, predictions, average=None, labels=[0, 1, 2])
            ],
            "confusion_matrix": confusion_matrix(
                labels, predictions, labels=[0, 1, 2]
            ).tolist(),
            "errors": int((predictions != labels).sum()),
        }

    def train_epochs(
        model,
        loader,
        class_weights,
        epochs,
        partial,
        valid_loader=None,
        checkpoint_path=None,
        best_macro_f1=-1.0,
        patience=None,
    ):
        if epochs <= 0:
            return []
        optimizer = torch.optim.AdamW(
            parameter_groups(model, partial), weight_decay=WEIGHT_DECAY
        )
        updates = math.ceil(len(loader) / GRAD_ACCUMULATION) * epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, updates)
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=LABEL_SMOOTHING,
        )
        history = []
        global_step = 0
        stale = 0
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for step, (images, targets) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw_loss = criterion(model(images), targets)
                (raw_loss / GRAD_ACCUMULATION).backward()
                total_loss += raw_loss.item()
                if (step + 1) % GRAD_ACCUMULATION == 0 or step + 1 == len(loader):
                    nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad), 1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
            row = {
                "phase": "partial" if partial else "head",
                "epoch": epoch,
                "train_loss": total_loss / len(loader),
            }
            if valid_loader is not None:
                row.update(score(valid_labels, predict(model, valid_loader)))
                if row["macro_f1"] > best_macro_f1:
                    best_macro_f1 = row["macro_f1"]
                    stale = 0
                    if checkpoint_path is not None:
                        torch.save(model.state_dict(), checkpoint_path)
                else:
                    stale += 1
            history.append(row)
            print(json.dumps(row), flush=True)
            if patience is not None and stale >= patience:
                break
        del optimizer, scheduler
        return history

    train_counts = np.bincount(train_frame["label"], minlength=NUM_CLASSES)
    train_weights = len(train_frame) / (NUM_CLASSES * train_counts)
    train_loader = make_loader(train_frame, train_transform, True, SEED)
    valid_loader = make_loader(valid_frame, eval_transform, False, SEED)

    training_started_at = time.perf_counter()
    validation_model = new_model().to(device)
    freeze_for_head(validation_model)
    head_history = train_epochs(
        validation_model,
        train_loader,
        train_weights,
        HEAD_WARMUP_EPOCHS,
        partial=False,
        valid_loader=valid_loader,
    )
    best_metrics = head_history[-1]
    best_partial_epoch = 0
    validation_checkpoint = run_root / "best_validation_model.pt"
    run_root.mkdir(parents=True, exist_ok=True)
    torch.save(validation_model.state_dict(), validation_checkpoint)

    unfreeze_tail(validation_model)
    partial_history = train_epochs(
        validation_model,
        train_loader,
        train_weights,
        MAX_PARTIAL_EPOCHS,
        partial=True,
        valid_loader=valid_loader,
        checkpoint_path=validation_checkpoint,
        best_macro_f1=best_metrics["macro_f1"],
        patience=EARLY_STOPPING_PATIENCE,
    )
    improvements = [
        row for row in partial_history if row["macro_f1"] > best_metrics["macro_f1"]
    ]
    if improvements:
        best_metrics = max(improvements, key=lambda row: row["macro_f1"])
        best_partial_epoch = best_metrics["epoch"]

    validation_model.load_state_dict(
        torch.load(validation_checkpoint, map_location=device, weights_only=True)
    )
    siglip_validation_probabilities = predict(validation_model, valid_loader, tta=True)
    siglip_validation_metrics = score(valid_labels, siglip_validation_probabilities)
    trainable_parameters = sum(
        parameter.numel() for parameter in validation_model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in validation_model.parameters())
    del validation_model, train_loader, valid_loader
    torch.cuda.empty_cache()
    cache_volume.commit()

    seed_all(SEED + 100)
    full_counts = np.bincount(manifest["label"], minlength=NUM_CLASSES)
    full_weights = len(manifest) / (NUM_CLASSES * full_counts)
    full_loader = make_loader(manifest, train_transform, True, SEED + 100)
    final_model = new_model().to(device)
    freeze_for_head(final_model)
    full_head_history = train_epochs(
        final_model,
        full_loader,
        full_weights,
        HEAD_WARMUP_EPOCHS,
        partial=False,
    )
    full_partial_history = []
    if best_partial_epoch:
        unfreeze_tail(final_model)
        full_partial_history = train_epochs(
            final_model,
            full_loader,
            full_weights,
            best_partial_epoch,
            partial=True,
        )
    final_checkpoint = run_root / "final_model.pt"
    torch.save(final_model.state_dict(), final_checkpoint)
    cache_volume.commit()
    training_seconds = time.perf_counter() - training_started_at

    test_paths = sorted(
        (path for path in (Path(DATA_ROOT) / "test").iterdir() if path.is_file()),
        key=lambda path: int(path.stem),
    )
    if len(test_paths) != 1_458:
        raise ValueError(f"Expected 1458 test images, found {len(test_paths)}")
    test_ids = np.asarray([int(path.stem) for path in test_paths], dtype=np.int64)
    test_frame = pd.DataFrame({"path": [str(path) for path in test_paths]})
    test_loader = make_loader(
        test_frame, eval_transform, False, SEED, with_labels=False
    )
    inference_started_at = time.perf_counter()
    siglip_test_probabilities = predict(final_model, test_loader, tta=True)
    inference_seconds = time.perf_counter() - inference_started_at
    del final_model, full_loader, test_loader
    torch.cuda.empty_cache()

    run11_validation = np.load(run11_validation_path, allow_pickle=False)
    run11_validation_probabilities = run11_validation[
        "calibrated_probabilities"
    ].astype(np.float64)
    inner_folds = run11_validation["inner_folds"].astype(np.int64)
    if not np.array_equal(run11_validation["labels"], valid_labels):
        raise ValueError("Run11 validation split does not match Run13")

    run11_test = np.load(run11_test_path, allow_pickle=False)
    run11_test_probabilities = run11_test["calibrated_probabilities"].astype(np.float64)
    if not np.array_equal(run11_test["ids"], test_ids):
        raise ValueError("Run11 test IDs do not match Run13")

    def blend(first, second, alpha):
        output = (1.0 - alpha) * first + alpha * second
        return output / output.sum(axis=1, keepdims=True)

    baseline_fold_scores = [
        f1_score(
            valid_labels[inner_folds == fold],
            run11_validation_probabilities[inner_folds == fold].argmax(axis=1),
            average="macro",
        )
        for fold in sorted(np.unique(inner_folds))
    ]
    grid = []
    for alpha in ALPHAS:
        probabilities = blend(
            run11_validation_probabilities,
            siglip_validation_probabilities,
            alpha,
        )
        row = score(valid_labels, probabilities)
        fold_scores = [
            f1_score(
                valid_labels[inner_folds == fold],
                probabilities[inner_folds == fold].argmax(axis=1),
                average="macro",
            )
            for fold in sorted(np.unique(inner_folds))
        ]
        row.update(
            {
                "alpha_siglip": alpha,
                "fold_macro_f1": [float(value) for value in fold_scores],
                "mean_fold_macro_f1": float(np.mean(fold_scores)),
                "non_degrading_folds": int(
                    sum(new >= old - 1e-12 for new, old in zip(fold_scores, baseline_fold_scores))
                ),
            }
        )
        grid.append(row)

    baseline_validation_metrics = grid[0]
    eligible = [
        row
        for row in grid[1:]
        if row["macro_f1"] > baseline_validation_metrics["macro_f1"]
        and row["mean_fold_macro_f1"] >= baseline_validation_metrics["mean_fold_macro_f1"]
        and row["non_degrading_folds"] >= 3
    ]
    selected = max(
        eligible,
        key=lambda row: (
            row["macro_f1"],
            row["mean_fold_macro_f1"],
            -row["alpha_siglip"],
        ),
        default=baseline_validation_metrics,
    )
    alpha = selected["alpha_siglip"]
    blended_validation_probabilities = blend(
        run11_validation_probabilities, siglip_validation_probabilities, alpha
    )
    blended_test_probabilities = blend(
        run11_test_probabilities, siglip_test_probabilities, alpha
    )
    test_predictions = blended_test_probabilities.argmax(axis=1).astype(int)

    template = pd.read_csv(Path(DATA_ROOT) / "submission.csv")[["id"]]
    prediction_map = dict(zip(test_ids, test_predictions, strict=True))
    template["predicted"] = template["id"].astype(int).map(prediction_map)
    if len(template) != 1_458 or template["predicted"].isna().any():
        raise ValueError("Invalid submission mapping")
    template["predicted"] = template["predicted"].astype(int)

    validation_output = valid_frame[["path", "label", "group"]].rename(
        columns={"label": "groundtruth"}
    )
    validation_output["inner_fold"] = inner_folds
    validation_output["run11_predicted"] = run11_validation_probabilities.argmax(axis=1)
    validation_output["siglip_predicted"] = siglip_validation_probabilities.argmax(axis=1)
    validation_output["blended_predicted"] = blended_validation_probabilities.argmax(axis=1)
    for label in range(NUM_CLASSES):
        validation_output[f"run11_p{label}"] = run11_validation_probabilities[:, label]
        validation_output[f"siglip_p{label}"] = siglip_validation_probabilities[:, label]
        validation_output[f"blended_p{label}"] = blended_validation_probabilities[:, label]

    metrics = {
        "method_version": METHOD_VERSION,
        "method": "partial SigLIP fine-tune + validation-selected Run11 probability blend",
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
        "head_warmup_epochs": HEAD_WARMUP_EPOCHS,
        "best_partial_epochs": best_partial_epoch,
        "unfrozen_blocks": UNFROZEN_BLOCKS,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "head_lr": HEAD_LR,
        "backbone_lr": BACKBONE_LR,
        "dataset_fingerprint": fingerprint,
        "validation_history": head_history + partial_history,
        "full_training_history": full_head_history + full_partial_history,
        "siglip_validation": siglip_validation_metrics,
        "run11_validation": score(valid_labels, run11_validation_probabilities),
        "selected_validation": score(valid_labels, blended_validation_probabilities),
        "selected_alpha_siglip": alpha,
        "selection_guard": "aggregate gain, mean-fold non-loss, >=3/5 non-degrading folds",
        "top_blends": sorted(
            grid,
            key=lambda row: (row["macro_f1"], row["mean_fold_macro_f1"], -row["alpha_siglip"]),
            reverse=True,
        )[:10],
        "test_changed_rows_vs_run11": int(
            (
                test_predictions
                != run11_test_probabilities.argmax(axis=1)
            ).sum()
        ),
        "test_class_counts": {
            str(label): int((test_predictions == label).sum()) for label in range(NUM_CLASSES)
        },
        "timing_seconds": {
            "training": training_seconds,
            "test_inference": inference_seconds,
            "end_to_end": time.perf_counter() - started_at,
        },
        "cached": False,
    }

    run_root.mkdir(parents=True, exist_ok=True)
    template.to_csv(submission_path, index=False)
    validation_output.to_csv(validation_csv_path, index=False)
    np.savez(
        validation_probability_path,
        labels=valid_labels,
        inner_folds=inner_folds,
        run11_probabilities=run11_validation_probabilities.astype(np.float32),
        siglip_probabilities=siglip_validation_probabilities.astype(np.float32),
        blended_probabilities=blended_validation_probabilities.astype(np.float32),
    )
    np.savez(
        test_probability_path,
        ids=test_ids,
        run11_probabilities=run11_test_probabilities.astype(np.float32),
        siglip_probabilities=siglip_test_probabilities.astype(np.float32),
        blended_probabilities=blended_test_probabilities.astype(np.float32),
        predictions=test_predictions,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cache_volume.commit()
    return payload(metrics)


@app.local_entrypoint()
def main(
    output_dir: str = "artifacts/context_encoder",
    force: bool = False,
):
    import json

    result = run_pipeline.remote(force=force)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "submission_context_ensemble.csv").write_text(
        result["submission"], encoding="utf-8"
    )
    (destination / "test_probabilities.npz").write_bytes(result["test_probabilities"])
    (destination / "validation_predictions.csv").write_text(
        result["validation_csv"], encoding="utf-8"
    )
    (destination / "validation_probabilities.npz").write_bytes(
        result["validation_probabilities"]
    )
    (destination / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    metrics = result["metrics"]
    print("SigLIP validation Macro-F1:", metrics["siglip_validation"]["macro_f1"])
    print("Selected alpha:", metrics["selected_alpha_siglip"])
    print("Blended validation Macro-F1:", metrics["selected_validation"]["macro_f1"])
    print("Changed test rows vs Run11:", metrics["test_changed_rows_vs_run11"])
    print("Timing seconds:", metrics["timing_seconds"])
    print(f"Wrote all artifacts to {destination.resolve()}")
