"""Fuse trained ConvNeXtV2 material evidence with context probability soup."""

from pathlib import Path
import sys

import modal


SCRIPT = Path(__file__).resolve()
ROOT = next(
    (path for path in (*SCRIPT.parents, Path.cwd()) if (path / "modal_backbone_app.py").is_file()),
    Path.cwd(),
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_backbone_app import (  # noqa: E402
    CACHE_ROOT,
    DATA_ROOT,
    SEED,
    cache_volume,
    data_volume,
    hf_secret,
    image,
)


image = image.add_local_file(ROOT / "modal_backbone_app.py", "/root/modal_backbone_app.py")
app = modal.App("bdc2026-material-context-fusion")

GPU = "A100-80GB"
MODEL = "facebook/convnextv2-large-22k-224"
REVISION = "e58a79c331e6c9acd20e3ba2de0e934c546f0eea"
RUN01_ROOT = f"{CACHE_ROOT}/runs/convnextv2_large_full_finetune_224_seed2026"
RUN17_ROOT = f"{CACHE_ROOT}/runs/material_context_fusion_seed2026"
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 8
METHOD_VERSION = 1


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=32768,
    timeout=2 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def extract_probabilities(force: bool = False):
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
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    started = time.perf_counter()
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    run01 = Path(RUN01_ROOT)
    run17 = Path(RUN17_ROOT)
    run17.mkdir(parents=True, exist_ok=True)
    validation_path = run17 / "validation_probabilities.npz"
    test_path = run17 / "test_probabilities.npz"
    submission_path = run17 / "submission_material_encoder_standalone.csv"
    metrics_path = run17 / "metrics.json"

    def payload(report):
        return {
            "metrics": report,
            "validation": validation_path.read_bytes(),
            "test": test_path.read_bytes(),
            "submission": submission_path.read_text(encoding="utf-8"),
        }

    if all(path.exists() for path in (validation_path, test_path, submission_path, metrics_path)) and not force:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        if report.get("method_version") == METHOD_VERSION:
            report["cached"] = True
            return payload(report)

    best_checkpoint = run01 / "best_validation.pt"
    final_checkpoint = run01 / "final_model.pt"
    for path in (best_checkpoint, final_checkpoint):
        if not path.exists():
            raise FileNotFoundError(f"Missing Run01 checkpoint: {path}")

    manifest = pd.read_csv(Path(DATA_ROOT) / "train_manifest.csv")
    manifest["label"] = manifest["label"].astype(int)
    if len(manifest) != 26_527 or set(manifest.columns) != {"path", "label", "group"}:
        raise ValueError("Invalid training manifest")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, valid_indices = next(splitter.split(manifest, manifest["label"], manifest["group"]))
    valid_frame = manifest.iloc[valid_indices].reset_index(drop=True)
    valid_labels = valid_frame["label"].to_numpy(np.int64)
    inner_folds = np.empty(len(valid_frame), dtype=np.int64)
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 11)
    for fold, (_, indices) in enumerate(inner.split(valid_frame, valid_frame["label"], valid_frame["group"])):
        inner_folds[indices] = fold

    test_files = sorted(
        (path for path in (Path(DATA_ROOT) / "test").iterdir() if path.is_file()),
        key=lambda path: int(path.stem),
    )
    if len(test_files) != 1_458:
        raise ValueError("Expected 1458 test images")
    test_ids = np.asarray([int(path.stem) for path in test_files], dtype=np.int64)
    test_frame = pd.DataFrame({"path": [str(path) for path in test_files]})

    processor = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION)

    class ResizePad:
        def __call__(self, source):
            resized = ImageOps.contain(source, (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (124, 116, 104))
            canvas.paste(resized, ((IMAGE_SIZE - resized.width) // 2, (IMAGE_SIZE - resized.height) // 2))
            return canvas

    transform = transforms.Compose(
        [
            ResizePad(),
            transforms.ToTensor(),
            transforms.Normalize(processor.image_mean, processor.image_std),
        ]
    )

    class Images(Dataset):
        def __init__(self, frame):
            self.paths = frame["path"].astype(str).tolist()

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            with Image.open(self.paths[index]) as source:
                return transform(ImageOps.exif_transpose(source).convert("RGB"))

    def loader(frame):
        return DataLoader(
            Images(frame),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )

    def new_model():
        return AutoModelForImageClassification.from_pretrained(
            MODEL,
            revision=REVISION,
            num_labels=3,
            ignore_mismatched_sizes=True,
        )

    @torch.inference_mode()
    def predict(model, frame):
        model.eval()
        output = []
        for images in loader(frame):
            images = images.to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pixel_values=images).logits
                logits = 0.5 * (logits + model(pixel_values=torch.flip(images, [3])).logits)
            output.append(torch.softmax(logits.float(), 1).cpu().numpy())
        return np.concatenate(output)

    def load_model(path):
        model = new_model().to("cuda")
        checkpoint = torch.load(path, map_location="cuda", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        return model

    validation_model = load_model(best_checkpoint)
    validation_probabilities = predict(validation_model, valid_frame)
    del validation_model
    torch.cuda.empty_cache()
    final_model = load_model(final_checkpoint)
    test_probabilities = predict(final_model, test_frame)
    predictions = test_probabilities.argmax(1).astype(int)

    validation_predictions = validation_probabilities.argmax(1)
    validation_metrics = {
        "accuracy": float(accuracy_score(valid_labels, validation_predictions)),
        "macro_f1": float(f1_score(valid_labels, validation_predictions, average="macro")),
        "class_f1": f1_score(valid_labels, validation_predictions, average=None, labels=[0, 1, 2]).tolist(),
        "confusion_matrix": confusion_matrix(valid_labels, validation_predictions, labels=[0, 1, 2]).tolist(),
        "errors": int((valid_labels != validation_predictions).sum()),
    }
    template = pd.read_csv(Path(DATA_ROOT) / "submission.csv")[["id"]]
    template["predicted"] = template["id"].astype(int).map(dict(zip(test_ids, predictions, strict=True)))
    if template["predicted"].isna().any():
        raise ValueError("Missing submission IDs")
    template["predicted"] = template["predicted"].astype(int)
    template.to_csv(submission_path, index=False)
    np.savez(validation_path, labels=valid_labels, inner_folds=inner_folds, probabilities=validation_probabilities.astype(np.float32))
    np.savez(test_path, ids=test_ids, probabilities=test_probabilities.astype(np.float32), predictions=predictions)
    report = {
        "method_version": METHOD_VERSION,
        "method": "Run01 best-validation/final checkpoints + flip TTA",
        "model": MODEL,
        "revision": REVISION,
        "seed": SEED,
        "validation": validation_metrics,
        "timing_seconds": {"end_to_end": time.perf_counter() - started},
        "cached": False,
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cache_volume.commit()
    return payload(report)


def write_remote(result, output):
    import json

    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_probabilities.npz").write_bytes(result["validation"])
    (output / "test_probabilities.npz").write_bytes(result["test"])
    (output / "submission_material_encoder_standalone.csv").write_text(result["submission"], encoding="utf-8")
    (output / "convnextv2_metrics.json").write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")


def ensemble_local(output, prefix="material_context"):
    import json

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    run15 = ROOT / "artifacts" / "probability_ensemble"
    base_v = np.load(run15 / "validation_probabilities.npz")
    base_t = np.load(run15 / "test_probabilities.npz")
    conv_v = np.load(output / "validation_probabilities.npz")
    conv_t = np.load(output / "test_probabilities.npz")
    labels = base_v["labels"].astype(np.int64)
    folds = base_v["inner_folds"].astype(np.int64)
    ids = base_t["ids"].astype(np.int64)
    base_prob = base_v["soup_probabilities"].astype(np.float64)
    base_test = base_t["soup_probabilities"].astype(np.float64)
    conv_prob = conv_v["probabilities"].astype(np.float64)
    conv_test = conv_t["probabilities"].astype(np.float64)
    if not np.array_equal(labels, conv_v["labels"]) or not np.array_equal(ids, conv_t["ids"]):
        raise ValueError("Run15 and Run17 rows differ")

    def score(predictions):
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "class_f1": f1_score(labels, predictions, average=None, labels=[0, 1, 2]).tolist(),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
            "errors": int((labels != predictions).sum()),
        }

    def fold_scores(predictions):
        return [float(f1_score(labels[folds == fold], predictions[folds == fold], average="macro")) for fold in sorted(np.unique(folds))]

    baseline_predictions = base_prob.argmax(1)
    baseline = score(baseline_predictions)
    baseline_folds = fold_scores(baseline_predictions)
    rows = []

    for temperature in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
        calibrated = np.clip(conv_prob, 1e-8, 1.0) ** (1.0 / temperature)
        calibrated /= calibrated.sum(1, keepdims=True)
        for weight in np.arange(0.025, 0.3001, 0.025):
            predictions = ((1.0 - weight) * base_prob + weight * calibrated).argmax(1)
            rows.append({"method": "soft", "temperature": temperature, "weight": float(weight), "predictions": predictions})

    conv_predictions = conv_prob.argmax(1)
    soup_margin = base_prob[:, 2] - base_prob[:, 0]
    for margin in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40):
        for confidence in (0.50, 0.60, 0.70, 0.80, 0.90):
            predictions = baseline_predictions.copy()
            gate = (predictions == 2) & (conv_predictions == 0) & (soup_margin <= margin) & (conv_prob[:, 0] >= confidence)
            predictions[gate] = 0
            rows.append({"method": "gated", "soup_margin_max": margin, "convnext_confidence_min": confidence, "predictions": predictions})

    def features(base, conv):
        entropy = lambda probabilities: -(np.clip(probabilities, 1e-8, 1.0) * np.log(np.clip(probabilities, 1e-8, 1.0))).sum(1)
        return np.column_stack(
            [
                np.log(np.clip(base[:, 0], 1e-8, 1.0) / np.clip(base[:, 2], 1e-8, 1.0)),
                np.log(np.clip(conv[:, 0], 1e-8, 1.0) / np.clip(conv[:, 2], 1e-8, 1.0)),
                base[:, 0],
                conv[:, 0],
                entropy(base),
                entropy(conv),
            ]
        )

    validation_features = features(base_prob, conv_prob)
    test_features = features(base_test, conv_test)
    for class_weight in (None, "balanced"):
        for regularization in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            predictions = baseline_predictions.copy()
            for fold in sorted(np.unique(folds)):
                train_mask = (folds != fold) & (labels != 1)
                valid_mask = (folds == fold) & (baseline_predictions == 2)
                model = LogisticRegression(C=regularization, class_weight=class_weight, random_state=SEED, max_iter=1_000).fit(validation_features[train_mask], labels[train_mask])
                predicted = model.predict(validation_features[valid_mask])
                indices = np.flatnonzero(valid_mask)
                predictions[indices[predicted == 0]] = 0
            rows.append({"method": "logistic", "C": regularization, "class_weight": class_weight, "predictions": predictions})

    for row in rows:
        predictions = row.pop("predictions")
        folds_now = fold_scores(predictions)
        row.update(
            {
                **score(predictions),
                "fold_macro_f1": folds_now,
                "mean_fold_macro_f1": float(np.mean(folds_now)),
                "non_degrading_folds": int(sum(new >= old - 1e-12 for new, old in zip(folds_now, baseline_folds))),
                "changed_validation_rows": int((predictions != baseline_predictions).sum()),
                "_predictions": predictions,
            }
        )
    eligible = [row for row in rows if row["macro_f1"] > baseline["macro_f1"] + 1e-12 and row["mean_fold_macro_f1"] >= np.mean(baseline_folds) - 1e-12 and row["non_degrading_folds"] >= 3]

    def best(method):
        pool = [row for row in rows if row["method"] == method]
        return max(pool, key=lambda row: (row["macro_f1"], row["mean_fold_macro_f1"], row["non_degrading_folds"]), default=None)

    best_soft, best_gated, best_logistic = best("soft"), best("gated"), best("logistic")
    selected = max(eligible, key=lambda row: (row["macro_f1"], row["mean_fold_macro_f1"], row["non_degrading_folds"], row["method"] in {"gated", "logistic"}), default=None)
    conv_test_predictions = conv_test.argmax(1)

    def apply_test(row):
        if row["method"] == "soft":
            calibrated = np.clip(conv_test, 1e-8, 1.0) ** (1.0 / row["temperature"])
            calibrated /= calibrated.sum(1, keepdims=True)
            return ((1.0 - row["weight"]) * base_test + row["weight"] * calibrated).argmax(1)
        if row["method"] == "logistic":
            predictions = base_test.argmax(1)
            train_mask = labels != 1
            test_mask = predictions == 2
            model = LogisticRegression(C=row["C"], class_weight=row["class_weight"], random_state=SEED, max_iter=1_000).fit(validation_features[train_mask], labels[train_mask])
            predicted = model.predict(test_features[test_mask])
            indices = np.flatnonzero(test_mask)
            predictions[indices[predicted == 0]] = 0
            return predictions
        predictions = base_test.argmax(1)
        margin = base_test[:, 2] - base_test[:, 0]
        gate = (predictions == 2) & (conv_test_predictions == 0) & (margin <= row["soup_margin_max"]) & (conv_test[:, 0] >= row["convnext_confidence_min"])
        predictions[gate] = 0
        return predictions

    soft_test = apply_test(best_soft)
    gated_test = apply_test(best_gated)
    logistic_test = apply_test(best_logistic)
    selected_test = apply_test(selected) if selected else base_test.argmax(1)
    template = pd.read_csv(ROOT / "BDC2026" / "submission.csv")[["id"]]

    def write(name, predictions):
        frame = template.copy()
        frame["predicted"] = frame["id"].astype(int).map(dict(zip(ids, predictions.astype(int), strict=True)))
        if frame["predicted"].isna().any():
            raise ValueError("Missing submission IDs")
        frame["predicted"] = frame["predicted"].astype(int)
        frame.to_csv(output / name, index=False)

    write(f"submission_{prefix}_soft_blend.csv", soft_test)
    write(f"submission_{prefix}_gated.csv", gated_test)
    write(f"submission_{prefix}_logistic_router.csv", logistic_test)
    write(f"submission_{prefix}_recommended.csv", selected_test)

    def clean(row):
        return None if row is None else {key: value for key, value in row.items() if key != "_predictions"}

    report = {
        "method_version": METHOD_VERSION,
        "method": "Run15 soup + Run01 ConvNeXtV2 validation-selected soft/gated ensemble",
        "test_labels_used": False,
        "baseline_validation": baseline,
        "convnext_validation": score(conv_prob.argmax(1)),
        "best_soft": clean(best_soft),
        "best_gated": clean(best_gated),
        "best_logistic": clean(best_logistic),
        "selected": clean(selected) or {"method": "none", "fallback": "Run15 soup"},
        "selected_validation": score(selected["_predictions"]) if selected else baseline,
        "test_changed_rows_vs_soup": int((selected_test != base_test.argmax(1)).sum()),
        "top_candidates": [clean(row) for row in sorted(rows, key=lambda row: (row["macro_f1"], row["mean_fold_macro_f1"]), reverse=True)[:20]],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Selected:", report["selected"])
    print("Validation Macro-F1:", baseline["macro_f1"], "->", report["selected_validation"]["macro_f1"])
    print("Changed test rows:", report["test_changed_rows_vs_soup"])


@app.local_entrypoint()
def main(action: str = "all", output_dir: str = "artifacts/material_context_fusion", force: bool = False):
    if action not in {"extract", "ensemble", "all"}:
        raise ValueError("action must be extract, ensemble, or all")
    output = Path(output_dir)
    if action in {"extract", "all"}:
        result = extract_probabilities.remote(force=force)
        write_remote(result, output)
        print("Run01 ConvNeXtV2 validation Macro-F1:", result["metrics"]["validation"]["macro_f1"])
        print("Wrote Run17 probability artifacts to", output.resolve())
    if action in {"ensemble", "all"}:
        ensemble_local(output)
