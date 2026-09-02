"""Inference-only class-0 logit calibration for augmented shape encoder.

Uses only Run06's held-out validation split to select one scalar bias. It never
reads local ground truth and does not retrain the backbone.

Run from the project root:
  modal run artifacts/class_calibration/modal_run11_pipeline.py

Add --force only to recompute validation inference instead of using Run11 cache.
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
    SEED,
    cache_volume,
    data_volume,
    hf_secret,
    image,
)


image = image.add_local_file(
    PROJECT_ROOT / "modal_backbone_app.py", "/root/modal_backbone_app.py"
)

APP_NAME = "bdc2026-class-boundary-calibration"
GPU = "A100-80GB"
MODEL_NAME = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
MODEL_REVISION = "e959efa74c867491dcfe3ec3e4f97382e39025b3"
BASE_RUN_NAME = "dinov3_convnext_large_bdc_augmentation_224_seed2026"
BASE_ROOT = f"{CACHE_ROOT}/runs/{BASE_RUN_NAME}"
RUN_NAME = "class_boundary_calibration_seed2026"
RUN_ROOT = f"{CACHE_ROOT}/runs/{RUN_NAME}"

NUM_CLASSES = 3
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 8
INNER_FOLDS = 5
BIAS_VALUES = tuple(round(-0.50 + 0.01 * index, 2) for index in range(201))
METHOD_VERSION = 1

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def run_pipeline(force: bool = False):
    import hashlib
    import io
    import json
    import os
    import random
    import time

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageFile, ImageOps
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModel

    started_at = time.perf_counter()
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    run_root = Path(RUN_ROOT)
    metrics_path = run_root / "metrics.json"
    submission_path = run_root / "submission_class_calibrated.csv"
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

    required_outputs = (
        metrics_path,
        submission_path,
        test_probability_path,
        validation_csv_path,
        validation_probability_path,
    )
    if all(path.exists() for path in required_outputs) and not force:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("method_version") == METHOD_VERSION:
            metrics["cached"] = True
            return payload(metrics)

    base_root = Path(BASE_ROOT)
    validation_checkpoint = base_root / "best_validation_model.pt"
    base_probability_path = base_root / "test_probabilities.npz"
    base_metrics_path = base_root / "training_metrics.json"
    for path in (validation_checkpoint, base_probability_path, base_metrics_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Run06 artifact: {path}. Run Run06 first."
            )

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
            f"{row.path}|{row.label}|{row.group}\n"
            for row in manifest.itertuples()
        ).encode()
    ).hexdigest()
    base_metrics = json.loads(base_metrics_path.read_text(encoding="utf-8"))
    if base_metrics.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Run06 checkpoint and current manifest do not match")

    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )
    _, valid_indices = next(
        splitter.split(manifest, manifest["label"], manifest["group"])
    )
    valid_frame = manifest.iloc[valid_indices].reset_index(drop=True)
    valid_labels = valid_frame["label"].to_numpy()

    inner_fold_ids = np.empty(len(valid_frame), dtype=np.int64)
    inner_splitter = StratifiedGroupKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + 11
    )
    for fold, (_, fold_indices) in enumerate(
        inner_splitter.split(
            valid_frame, valid_frame["label"], valid_frame["group"]
        )
    ):
        inner_fold_ids[fold_indices] = fold

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION
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
                (
                    (IMAGE_SIZE - resized.width) // 2,
                    (IMAGE_SIZE - resized.height) // 2,
                ),
            )
            return canvas

    transform = transforms.Compose(
        [
            ResizePad(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=processor.image_mean, std=processor.image_std
            ),
        ]
    )

    class WasteDataset(Dataset):
        def __len__(self):
            return len(valid_frame)

        def __getitem__(self, index):
            with Image.open(valid_frame.iloc[index]["path"]) as source:
                image_value = ImageOps.exif_transpose(source).convert("RGB")
                return transform(image_value)

    class DINOConvNextClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(
                MODEL_NAME, revision=MODEL_REVISION
            )
            self.classifier = nn.Linear(
                self.backbone.config.hidden_sizes[-1], NUM_CLASSES
            )

        def forward(self, pixel_values):
            pooled = self.backbone(pixel_values=pixel_values).pooler_output
            return self.classifier(pooled)

    device = torch.device("cuda")
    model = DINOConvNextClassifier().to(device)
    model.load_state_dict(
        torch.load(
            validation_checkpoint, map_location=device, weights_only=True
        )
    )
    model.eval()
    loader = DataLoader(
        WasteDataset(),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    inference_started_at = time.perf_counter()
    validation_probabilities = []
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = 0.5 * (
                    model(images) + model(torch.flip(images, dims=[3]))
                )
            validation_probabilities.append(
                torch.softmax(logits.float(), dim=1).cpu().numpy()
            )
    validation_probabilities = np.concatenate(validation_probabilities)
    validation_inference_seconds = time.perf_counter() - inference_started_at
    del model
    torch.cuda.empty_cache()

    def apply_bias(probabilities, bias):
        adjusted = probabilities.astype(np.float64, copy=True)
        adjusted[:, 0] *= np.exp(bias)
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        return adjusted

    def metrics(labels, probabilities):
        predictions = probabilities.argmax(axis=1)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "class_f1": [
                float(value)
                for value in f1_score(
                    labels, predictions, average=None, labels=[0, 1, 2]
                )
            ],
            "confusion_matrix": confusion_matrix(
                labels, predictions, labels=[0, 1, 2]
            ).tolist(),
            "errors": int((predictions != labels).sum()),
        }

    selection_started_at = time.perf_counter()
    baseline_predictions = validation_probabilities.argmax(axis=1)
    baseline_metrics = metrics(valid_labels, validation_probabilities)
    grid = []
    for bias in BIAS_VALUES:
        adjusted = apply_bias(validation_probabilities, bias)
        predictions = adjusted.argmax(axis=1)
        fold_macro_f1 = [
            float(
                f1_score(
                    valid_labels[inner_fold_ids == fold],
                    predictions[inner_fold_ids == fold],
                    average="macro",
                )
            )
            for fold in range(INNER_FOLDS)
        ]
        row_metrics = metrics(valid_labels, adjusted)
        grid.append(
            {
                "class0_logit_bias": bias,
                "macro_f1": row_metrics["macro_f1"],
                "accuracy": row_metrics["accuracy"],
                "errors": row_metrics["errors"],
                "changed_rows": int((predictions != baseline_predictions).sum()),
                "fold_macro_f1": fold_macro_f1,
                "mean_fold_macro_f1": float(np.mean(fold_macro_f1)),
            }
        )

    selected = max(
        grid,
        key=lambda row: (
            row["macro_f1"],
            row["accuracy"],
            -abs(row["class0_logit_bias"]),
        ),
    )
    if selected["macro_f1"] <= baseline_metrics["macro_f1"]:
        selected = next(row for row in grid if row["class0_logit_bias"] == 0.0)

    selected_bias = selected["class0_logit_bias"]
    calibrated_validation_probabilities = apply_bias(
        validation_probabilities, selected_bias
    )
    calibrated_validation_predictions = (
        calibrated_validation_probabilities.argmax(axis=1)
    )
    calibrated_validation_metrics = metrics(
        valid_labels, calibrated_validation_probabilities
    )

    leave_one_fold_out_biases = []
    for held_out_fold in range(INNER_FOLDS):
        keep = inner_fold_ids != held_out_fold
        fold_best = max(
            BIAS_VALUES,
            key=lambda bias: (
                f1_score(
                    valid_labels[keep],
                    apply_bias(validation_probabilities[keep], bias).argmax(axis=1),
                    average="macro",
                ),
                -abs(bias),
            ),
        )
        leave_one_fold_out_biases.append(float(fold_best))
    selection_seconds = time.perf_counter() - selection_started_at

    base_archive = np.load(base_probability_path, allow_pickle=False)
    test_ids = base_archive["ids"].astype(np.int64)
    base_test_probabilities = base_archive["probabilities"].astype(np.float64)
    if base_test_probabilities.shape != (1_458, NUM_CLASSES):
        raise ValueError("Invalid Run06 test probabilities")

    application_started_at = time.perf_counter()
    calibrated_test_probabilities = apply_bias(
        base_test_probabilities, selected_bias
    )
    base_test_predictions = base_test_probabilities.argmax(axis=1)
    calibrated_test_predictions = calibrated_test_probabilities.argmax(axis=1)
    application_seconds = time.perf_counter() - application_started_at

    template = pd.read_csv(Path(DATA_ROOT) / "submission.csv")[["id"]]
    prediction_map = dict(
        zip(test_ids, calibrated_test_predictions.astype(int), strict=True)
    )
    template["predicted"] = template["id"].astype(int).map(prediction_map)
    if (
        len(template) != 1_458
        or template["predicted"].isna().any()
        or not template["id"].is_unique
    ):
        raise ValueError("Invalid submission mapping")
    template["predicted"] = template["predicted"].astype(int)

    validation_output = valid_frame[["path", "label", "group"]].copy()
    validation_output = validation_output.rename(columns={"label": "groundtruth"})
    validation_output["base_predicted"] = baseline_predictions
    validation_output["calibrated_predicted"] = calibrated_validation_predictions
    for label in range(NUM_CLASSES):
        validation_output[f"base_p{label}"] = validation_probabilities[:, label]
        validation_output[f"calibrated_p{label}"] = (
            calibrated_validation_probabilities[:, label]
        )
    validation_output["inner_fold"] = inner_fold_ids

    metrics_output = {
        "method_version": METHOD_VERSION,
        "method": "validation-selected class-0 logit bias",
        "formula": "p0 *= exp(bias); normalize; argmax",
        "base_run": BASE_RUN_NAME,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "bias_grid": [min(BIAS_VALUES), max(BIAS_VALUES), 0.01],
        "selected_class0_logit_bias": selected_bias,
        "selection_guard": "use bias 0 if validation Macro-F1 does not improve",
        "baseline_validation": baseline_metrics,
        "calibrated_validation": calibrated_validation_metrics,
        "validation_macro_f1_gain": (
            calibrated_validation_metrics["macro_f1"]
            - baseline_metrics["macro_f1"]
        ),
        "leave_one_fold_out_selected_biases": leave_one_fold_out_biases,
        "top_candidates": sorted(
            grid,
            key=lambda row: (
                row["macro_f1"],
                row["accuracy"],
                -abs(row["class0_logit_bias"]),
            ),
            reverse=True,
        )[:10],
        "test_changed_rows": int(
            (calibrated_test_predictions != base_test_predictions).sum()
        ),
        "base_test_class_counts": {
            str(label): int((base_test_predictions == label).sum())
            for label in range(NUM_CLASSES)
        },
        "calibrated_test_class_counts": {
            str(label): int((calibrated_test_predictions == label).sum())
            for label in range(NUM_CLASSES)
        },
        "dataset_fingerprint": fingerprint,
        "timing_seconds": {
            "validation_inference": validation_inference_seconds,
            "parameter_selection": selection_seconds,
            "test_calibration": application_seconds,
            "end_to_end": time.perf_counter() - started_at,
        },
        "cached": False,
    }

    run_root.mkdir(parents=True, exist_ok=True)
    template.to_csv(submission_path, index=False)
    np.savez(
        test_probability_path,
        ids=test_ids,
        base_probabilities=base_test_probabilities.astype(np.float32),
        calibrated_probabilities=calibrated_test_probabilities.astype(np.float32),
        base_predictions=base_test_predictions.astype(np.int64),
        calibrated_predictions=calibrated_test_predictions.astype(np.int64),
    )
    validation_output.to_csv(validation_csv_path, index=False)
    np.savez(
        validation_probability_path,
        labels=valid_labels,
        inner_folds=inner_fold_ids,
        base_probabilities=validation_probabilities.astype(np.float32),
        calibrated_probabilities=(
            calibrated_validation_probabilities.astype(np.float32)
        ),
        base_predictions=baseline_predictions.astype(np.int64),
        calibrated_predictions=calibrated_validation_predictions.astype(np.int64),
    )
    metrics_path.write_text(
        json.dumps(metrics_output, indent=2), encoding="utf-8"
    )
    cache_volume.commit()
    return payload(metrics_output)


@app.local_entrypoint()
def main(
    output_dir: str = "artifacts/class_calibration",
    force: bool = False,
):
    import json

    result = run_pipeline.remote(force=force)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "submission_class_calibrated.csv").write_text(
        result["submission"], encoding="utf-8"
    )
    (destination / "test_probabilities.npz").write_bytes(
        result["test_probabilities"]
    )
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
    print("Selected class-0 logit bias:", metrics["selected_class0_logit_bias"])
    print(
        "Validation Macro-F1:",
        metrics["baseline_validation"]["macro_f1"],
        "->",
        metrics["calibrated_validation"]["macro_f1"],
    )
    print("Changed test rows:", metrics["test_changed_rows"])
    print("Timing seconds:", metrics["timing_seconds"])
    print(f"Wrote all artifacts to {destination.resolve()}")
