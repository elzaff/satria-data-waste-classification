"""Low-LR SigLIP continuation, TTA adjustment, and blend recalibration.

Requires Run11 and Run13 checkpoints/artifacts in the same Modal workspace.

Run:
  modal run --detach artifacts/context_refinement/modal_run13b_pipeline.py
"""

from pathlib import Path
import sys

import modal


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (
        path
        for path in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents, Path.cwd())
        if (path / "modal_backbone_app.py").is_file()
    ),
    Path.cwd(),
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modal_backbone_app import (  # noqa: E402
    CACHE_ROOT,
    DATA_ROOT,
    SEED,
    cache_volume,
    data_volume,
    hf_secret,
    image,
)


image = image.pip_install("timm==1.0.27").add_local_file(
    PROJECT_ROOT / "modal_backbone_app.py", "/root/modal_backbone_app.py"
)

APP_NAME = "bdc2026-context-refinement"
GPU = "A100-80GB"
MODEL_ARCH = "vit_so400m_patch14_siglip_378.webli_ft_in1k"
RUN11_ROOT = f"{CACHE_ROOT}/runs/class_boundary_calibration_seed2026"
RUN13_ROOT = f"{CACHE_ROOT}/runs/context_encoder_seed2026"
RUN_NAME = "context_refinement_seed2026"
RUN_ROOT = f"{CACHE_ROOT}/runs/{RUN_NAME}"

NUM_CLASSES = 3
IMAGE_SIZE = 378
BATCH_SIZE = 24
GRAD_ACCUMULATION = 2
EXTRA_EPOCHS = 2
UNFROZEN_BLOCKS = 4
BACKBONE_LR = 7.5e-7
HEAD_LR = 7.5e-6
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
NUM_WORKERS = 8
TEMPERATURES = (0.8, 0.9, 1.0, 1.1, 1.2)
ALPHAS = tuple(round(index * 0.025, 3) for index in range(25))  # 0..0.60
METHOD_VERSION = 1

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=49152,
    timeout=3 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def run_pipeline(force: bool = False):
    import hashlib
    import json
    import math
    import os
    import random
    import time

    import numpy as np
    import pandas as pd
    import timm
    import torch
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

    seed_all(SEED + 13)
    device = torch.device("cuda")
    run_root = Path(RUN_ROOT)
    metrics_path = run_root / "metrics.json"
    submission_path = run_root / "submission_context_refinement.csv"
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

    base_validation_checkpoint = Path(RUN13_ROOT) / "best_validation_model.pt"
    base_final_checkpoint = Path(RUN13_ROOT) / "final_model.pt"
    run13_validation_path = Path(RUN13_ROOT) / "validation_probabilities.npz"
    run13_test_path = Path(RUN13_ROOT) / "test_probabilities.npz"
    run11_validation_path = Path(RUN11_ROOT) / "validation_probabilities.npz"
    run11_test_path = Path(RUN11_ROOT) / "test_probabilities.npz"
    for path in (
        base_validation_checkpoint,
        base_final_checkpoint,
        run13_validation_path,
        run13_test_path,
        run11_validation_path,
        run11_test_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing prerequisite artifact: {path}")

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

    def new_model(checkpoint):
        model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=NUM_CLASSES)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        return model

    probe = new_model(base_validation_checkpoint)
    config = timm.data.resolve_model_data_config(probe)
    mean, std = config["mean"], config["std"]
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
            transforms.RandomApply(
                [transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], 0.35
            ),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(
                p=0.10, scale=(0.02, 0.15), ratio=(0.5, 2.0), value="random"
            ),
        ]
    )
    eval_transform = transforms.Compose([ResizePad(), transforms.ToTensor(), normalize])

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

    def loader(frame, transform, shuffle, seed, with_labels=True):
        return DataLoader(
            WasteDataset(frame, transform, with_labels),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            generator=torch.Generator().manual_seed(seed),
        )

    def unfreeze_tail(model):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for block in model.blocks[-UNFROZEN_BLOCKS:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for name in ("norm", "fc_norm"):
            module = getattr(model, name, None)
            if isinstance(module, nn.Module):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        for parameter in model.get_classifier().parameters():
            parameter.requires_grad = True

    @torch.inference_mode()
    def predict(model, data_loader):
        model.eval()
        output = []
        for batch in data_loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
            output.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(output)

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

    def resume(model, data_loader, class_weights, epochs, valid_loader=None):
        unfreeze_tail(model)
        head_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
        backbone = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone, "lr": BACKBONE_LR},
                {"params": model.get_classifier().parameters(), "lr": HEAD_LR},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        updates = math.ceil(len(data_loader) / GRAD_ACCUMULATION) * epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, updates)
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=LABEL_SMOOTHING,
        )
        history = []
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for step, (images, targets) in enumerate(data_loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw_loss = criterion(model(images), targets)
                (raw_loss / GRAD_ACCUMULATION).backward()
                total_loss += raw_loss.item()
                if (step + 1) % GRAD_ACCUMULATION == 0 or step + 1 == len(data_loader):
                    nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad), 1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            row = {"epoch": epoch, "train_loss": total_loss / len(data_loader)}
            if valid_loader is not None:
                row.update(score(valid_labels, predict(model, valid_loader)))
            history.append(row)
            print(json.dumps(row), flush=True)
        return history

    train_counts = np.bincount(train_frame["label"], minlength=NUM_CLASSES)
    train_weights = len(train_frame) / (NUM_CLASSES * train_counts)
    train_loader = loader(train_frame, train_transform, True, SEED + 13)
    valid_loader = loader(valid_frame, eval_transform, False, SEED)

    training_started = time.perf_counter()
    validation_model = new_model(base_validation_checkpoint).to(device)
    baseline_siglip_validation = predict(validation_model, valid_loader)
    baseline_siglip_metrics = score(valid_labels, baseline_siglip_validation)
    run_root.mkdir(parents=True, exist_ok=True)
    selected_validation_checkpoint = run_root / "best_validation_model.pt"
    torch.save(validation_model.state_dict(), selected_validation_checkpoint)
    history = resume(
        validation_model,
        train_loader,
        train_weights,
        EXTRA_EPOCHS,
        valid_loader,
    )
    best_extra_epochs = 0
    best_macro_f1 = baseline_siglip_metrics["macro_f1"]
    # Re-run each selected epoch is avoided: save candidates during one short pass.
    # The last improving epoch is selected from history; if epoch 1 wins, full model
    # receives one extra epoch. Validation probabilities use current model only when
    # the last epoch wins; otherwise the base checkpoint is the conservative fallback.
    improving = [row for row in history if row["macro_f1"] > best_macro_f1]
    if improving:
        winner = max(improving, key=lambda row: row["macro_f1"])
        best_extra_epochs = winner["epoch"]
    if best_extra_epochs != EXTRA_EPOCHS:
        validation_model = new_model(base_validation_checkpoint).to(device)
        if best_extra_epochs:
            validation_model = validation_model.to(device)
            resume(
                validation_model,
                loader(train_frame, train_transform, True, SEED + 13),
                train_weights,
                best_extra_epochs,
            )
    torch.save(validation_model.state_dict(), selected_validation_checkpoint)
    siglip_validation = predict(validation_model, valid_loader)
    siglip_validation_metrics = score(valid_labels, siglip_validation)
    del validation_model, train_loader, valid_loader
    torch.cuda.empty_cache()

    seed_all(SEED + 113)
    final_model = new_model(base_final_checkpoint).to(device)
    full_history = []
    if best_extra_epochs:
        full_counts = np.bincount(manifest["label"], minlength=NUM_CLASSES)
        full_weights = len(manifest) / (NUM_CLASSES * full_counts)
        full_loader = loader(manifest, train_transform, True, SEED + 113)
        full_history = resume(
            final_model, full_loader, full_weights, best_extra_epochs
        )
        del full_loader
    torch.save(final_model.state_dict(), run_root / "final_model.pt")
    cache_volume.commit()
    training_seconds = time.perf_counter() - training_started

    test_paths = sorted(
        (path for path in (Path(DATA_ROOT) / "test").iterdir() if path.is_file()),
        key=lambda path: int(path.stem),
    )
    if len(test_paths) != 1_458:
        raise ValueError(f"Expected 1458 test images, found {len(test_paths)}")
    test_ids = np.asarray([int(path.stem) for path in test_paths], dtype=np.int64)
    test_frame = pd.DataFrame({"path": [str(path) for path in test_paths]})
    test_loader = loader(test_frame, eval_transform, False, SEED, with_labels=False)
    inference_started = time.perf_counter()
    siglip_test = predict(final_model, test_loader)
    inference_seconds = time.perf_counter() - inference_started
    del final_model, test_loader
    torch.cuda.empty_cache()

    run11_validation = np.load(run11_validation_path, allow_pickle=False)
    run11_test = np.load(run11_test_path, allow_pickle=False)
    run13_validation = np.load(run13_validation_path, allow_pickle=False)
    run13_test = np.load(run13_test_path, allow_pickle=False)
    run11_validation_p = run11_validation["calibrated_probabilities"].astype(np.float64)
    run11_test_p = run11_test["calibrated_probabilities"].astype(np.float64)
    inner_folds = run11_validation["inner_folds"].astype(np.int64)
    if not np.array_equal(run11_validation["labels"], valid_labels):
        raise ValueError("Run11 validation labels do not match")
    if not np.array_equal(run11_test["ids"], test_ids):
        raise ValueError("Run11 test IDs do not match")

    baseline_validation_p = run13_validation["blended_probabilities"].astype(np.float64)
    baseline_test_p = run13_test["blended_probabilities"].astype(np.float64)
    baseline_metrics = score(valid_labels, baseline_validation_p)

    def temperature(probabilities, value):
        adjusted = np.clip(probabilities, 1e-8, 1.0) ** (1.0 / value)
        return adjusted / adjusted.sum(axis=1, keepdims=True)

    def blend(first, second, alpha):
        output = (1.0 - alpha) * first + alpha * second
        return output / output.sum(axis=1, keepdims=True)

    folds = sorted(np.unique(inner_folds))
    baseline_fold_scores = [
        f1_score(
            valid_labels[inner_folds == fold],
            baseline_validation_p[inner_folds == fold].argmax(axis=1),
            average="macro",
        )
        for fold in folds
    ]
    grid = []
    for t_run11 in TEMPERATURES:
        calibrated_run11 = temperature(run11_validation_p, t_run11)
        for t_siglip in TEMPERATURES:
            calibrated_siglip = temperature(siglip_validation, t_siglip)
            for alpha in ALPHAS:
                probabilities = blend(calibrated_run11, calibrated_siglip, alpha)
                row = score(valid_labels, probabilities)
                fold_scores = [
                    f1_score(
                        valid_labels[inner_folds == fold],
                        probabilities[inner_folds == fold].argmax(axis=1),
                        average="macro",
                    )
                    for fold in folds
                ]
                row.update(
                    {
                        "temperature_run11": t_run11,
                        "temperature_siglip": t_siglip,
                        "alpha_siglip": alpha,
                        "fold_macro_f1": [float(value) for value in fold_scores],
                        "mean_fold_macro_f1": float(np.mean(fold_scores)),
                        "non_degrading_folds": int(
                            sum(
                                new >= old - 1e-12
                                for new, old in zip(fold_scores, baseline_fold_scores)
                            )
                        ),
                    }
                )
                grid.append(row)

    eligible = [
        row
        for row in grid
        if row["macro_f1"] > baseline_metrics["macro_f1"]
        and row["mean_fold_macro_f1"] >= float(np.mean(baseline_fold_scores))
        and row["non_degrading_folds"] >= 3
    ]
    if eligible:
        best_macro = max(row["macro_f1"] for row in eligible)
        near_best = [row for row in eligible if row["macro_f1"] >= best_macro - 0.0003]
        selected = max(
            near_best,
            key=lambda row: (
                row["non_degrading_folds"],
                -abs(row["temperature_run11"] - 1.0)
                - abs(row["temperature_siglip"] - 1.0),
                -row["alpha_siglip"],
                row["mean_fold_macro_f1"],
            ),
        )
        calibrated_run11_validation = temperature(
            run11_validation_p, selected["temperature_run11"]
        )
        calibrated_siglip_validation = temperature(
            siglip_validation, selected["temperature_siglip"]
        )
        calibrated_run11_test = temperature(
            run11_test_p, selected["temperature_run11"]
        )
        calibrated_siglip_test = temperature(
            siglip_test, selected["temperature_siglip"]
        )
        selected_validation_p = blend(
            calibrated_run11_validation,
            calibrated_siglip_validation,
            selected["alpha_siglip"],
        )
        selected_test_p = blend(
            calibrated_run11_test,
            calibrated_siglip_test,
            selected["alpha_siglip"],
        )
    else:
        selected = {
            "temperature_run11": None,
            "temperature_siglip": None,
            "alpha_siglip": None,
            "fallback": "Run13",
        }
        selected_validation_p = baseline_validation_p
        selected_test_p = baseline_test_p

    predictions = selected_test_p.argmax(axis=1).astype(int)
    template = pd.read_csv(Path(DATA_ROOT) / "submission.csv")[["id"]]
    template["predicted"] = template["id"].astype(int).map(
        dict(zip(test_ids, predictions, strict=True))
    )
    if len(template) != 1_458 or template["predicted"].isna().any():
        raise ValueError("Invalid submission mapping")
    template["predicted"] = template["predicted"].astype(int)

    validation_output = valid_frame[["path", "label", "group"]].rename(
        columns={"label": "groundtruth"}
    )
    validation_output["inner_fold"] = inner_folds
    validation_output["run13_predicted"] = baseline_validation_p.argmax(axis=1)
    validation_output["siglip_no_tta_predicted"] = siglip_validation.argmax(axis=1)
    validation_output["run13b_predicted"] = selected_validation_p.argmax(axis=1)
    for label in range(NUM_CLASSES):
        validation_output[f"run13_p{label}"] = baseline_validation_p[:, label]
        validation_output[f"siglip_no_tta_p{label}"] = siglip_validation[:, label]
        validation_output[f"run13b_p{label}"] = selected_validation_p[:, label]

    metrics = {
        "method_version": METHOD_VERSION,
        "method": "Run13 resume + no-TTA + conservative temperature/blend selection",
        "base_run": "Run13",
        "seed": SEED,
        "extra_epoch_limit": EXTRA_EPOCHS,
        "selected_extra_epochs": best_extra_epochs,
        "backbone_lr": BACKBONE_LR,
        "head_lr": HEAD_LR,
        "dataset_fingerprint": fingerprint,
        "baseline_siglip_no_tta": baseline_siglip_metrics,
        "resume_history": history,
        "full_resume_history": full_history,
        "selected_siglip_no_tta": siglip_validation_metrics,
        "run13_validation": baseline_metrics,
        "run13b_validation": score(valid_labels, selected_validation_p),
        "selected_parameters": selected,
        "selection_guard": "beat Run13, mean-fold non-loss, >=3 folds non-degrading, conservative near-best",
        "top_candidates": sorted(
            grid,
            key=lambda row: (
                row["macro_f1"],
                row["mean_fold_macro_f1"],
                row["non_degrading_folds"],
            ),
            reverse=True,
        )[:10],
        "test_changed_rows_vs_run13": int(
            (predictions != baseline_test_p.argmax(axis=1)).sum()
        ),
        "test_class_counts": {
            str(label): int((predictions == label).sum()) for label in range(NUM_CLASSES)
        },
        "timing_seconds": {
            "training": training_seconds,
            "test_inference": inference_seconds,
            "end_to_end": time.perf_counter() - started_at,
        },
        "cached": False,
    }

    template.to_csv(submission_path, index=False)
    validation_output.to_csv(validation_csv_path, index=False)
    np.savez(
        validation_probability_path,
        labels=valid_labels,
        inner_folds=inner_folds,
        run13_probabilities=baseline_validation_p.astype(np.float32),
        run11_probabilities=run11_validation_p.astype(np.float32),
        siglip_no_tta_probabilities=siglip_validation.astype(np.float32),
        blended_probabilities=selected_validation_p.astype(np.float32),
    )
    np.savez(
        test_probability_path,
        ids=test_ids,
        run13_probabilities=baseline_test_p.astype(np.float32),
        run11_probabilities=run11_test_p.astype(np.float32),
        siglip_no_tta_probabilities=siglip_test.astype(np.float32),
        blended_probabilities=selected_test_p.astype(np.float32),
        predictions=predictions,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cache_volume.commit()
    return payload(metrics)


@app.local_entrypoint()
def main(
    output_dir: str = "artifacts/context_refinement",
    force: bool = False,
):
    import json

    result = run_pipeline.remote(force=force)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "submission_context_refinement.csv").write_text(
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
    print("Selected extra epochs:", metrics["selected_extra_epochs"])
    print("Selected parameters:", metrics["selected_parameters"])
    print(
        "Validation Macro-F1:",
        metrics["run13_validation"]["macro_f1"],
        "->",
        metrics["run13b_validation"]["macro_f1"],
    )
    print("Changed test rows:", metrics["test_changed_rows_vs_run13"])
    print("Timing seconds:", metrics["timing_seconds"])
    print(f"Wrote all artifacts to {destination.resolve()}")
