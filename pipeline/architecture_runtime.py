"""Shared runtime for three validation-pure BDC final architectures.

This module is mounted into Modal containers by the small architecture-specific
entrypoints. It reads only official BDC train/test images. Test labels and old
run artifacts are never read.
"""

from __future__ import annotations

import numpy as np


SEED = 2026
NUM_CLASSES = 3
DATA_ROOT = "/data/BDC2026"
CACHE_ROOT = "/cache/final_architectures"

MODEL_CONFIGS = {
    "siglip2": {
        "repo": "google/siglip2-so400m-patch14-384",
        "revision": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
        "size": 384,
        "batch": 24,
        "head_lr": 3e-5,
        "backbone_lr": 2e-6,
        "epochs": 4,
    },
    "dino": {
        "repo": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
        "revision": "e959efa74c867491dcfe3ec3e4f97382e39025b3",
        "size": 224,
        "batch": 32,
        "head_lr": 1e-4,
        "backbone_lr": 1e-5,
        "epochs": 4,
    },
    "pe_core": {
        "repo": "vit_pe_core_large_patch14_336.fb",
        "revision": "e63206c8e3a0e9b699e40f31080eebd78fd2258e",
        "size": 336,
        "batch": 24,
        "head_lr": 1e-4,
        "backbone_lr": 2e-6,
        "epochs": 3,
    },
}


def _imports():
    import hashlib
    import json
    import math
    import os
    import random
    import shutil
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageFile, ImageOps
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    return locals()


def run_architecture(architecture: str, full: bool = False, force: bool = False):
    modules = _imports()
    globals().update({key: value for key, value in modules.items() if not key.startswith("_")})

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    if architecture not in {
        "hierarchical_siglip2",
        "hierarchical_patch_mil_siglip2",
        "label_refinement",
        "tri_encoder_moe",
    }:
        raise ValueError(f"Unknown architecture: {architecture}")

    _seed_everything(SEED)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    run_name = f"{architecture}_{'full5fold' if full else 'pilot1fold'}_seed{SEED}"
    output = Path(CACHE_ROOT) / run_name
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    submission_path = output / "submission.csv"
    cache_files = [metrics_path, submission_path]
    if architecture == "hierarchical_siglip2" and full:
        cache_files.extend(
            [output / "oof_probabilities.npz", output / "test_probabilities.npz"]
        )
    if all(path.exists() for path in cache_files) and not force:
        return {
            "cached": True,
            "run_root": str(output),
            "metrics": metrics_path.read_text(encoding="utf-8"),
            "submission": submission_path.read_text(encoding="utf-8"),
        }

    started = time.perf_counter()
    manifest, template, test_paths = _load_data()
    fold_ids = _make_folds(manifest)
    active_folds = list(range(5)) if full else [0]

    if architecture == "hierarchical_siglip2":
        result = _run_hierarchical(manifest, fold_ids, test_paths, active_folds, output)
    elif architecture == "hierarchical_patch_mil_siglip2":
        result = _run_patch_mil(
            manifest,
            fold_ids,
            test_paths,
            active_folds,
            output,
            reuse_checkpoints=not force,
        )
    elif architecture == "label_refinement":
        result = _run_label_refinement(manifest, fold_ids, test_paths, active_folds, output)
    else:
        result = _run_tri_encoder(
            manifest,
            fold_ids,
            test_paths,
            active_folds,
            output,
            reuse_checkpoints=not force,
        )

    predictions = result.pop("test_predictions").astype(int)
    if len(predictions) != len(template):
        raise RuntimeError(f"Prediction count {len(predictions)} != template {len(template)}")
    submission = template[["id"]].copy()
    submission["predicted"] = predictions
    if set(submission["predicted"].unique()) - {0, 1, 2}:
        raise RuntimeError("Predictions contain invalid labels")
    submission.to_csv(submission_path, index=False)

    report = {
        "architecture": architecture,
        "mode": "full5fold" if full else "pilot1fold",
        "test_labels_used": False,
        "seed": SEED,
        "rows_train": int(len(manifest)),
        "rows_test": int(len(template)),
        "active_folds": active_folds,
        "model_configs": MODEL_CONFIGS,
        "runtime_seconds": float(time.perf_counter() - started),
        **result,
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "cached": False,
        "run_root": str(output),
        "metrics": metrics_path.read_text(encoding="utf-8"),
        "submission": submission_path.read_text(encoding="utf-8"),
    }


def run_frozen_dino_specialist(force: bool = False):
    modules = _imports()
    globals().update({key: value for key, value in modules.items() if not key.startswith("_")})
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    _seed_everything(SEED)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    anchor = Path(CACHE_ROOT) / f"hierarchical_siglip2_full5fold_seed{SEED}"
    required = (
        anchor / "metrics.json",
        anchor / "oof_probabilities.npz",
        anchor / "submission.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "FINAL/02 full anchor incomplete. Run pipeline.py --full first. Missing: "
            + ", ".join(missing)
        )

    output = Path(CACHE_ROOT) / f"final02_frozen_dino_specialist_seed{SEED}"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    submission_path = output / "submission.csv"
    if metrics_path.exists() and submission_path.exists() and not force:
        return {
            "cached": True,
            "run_root": str(output),
            "metrics": metrics_path.read_text(encoding="utf-8"),
            "submission": submission_path.read_text(encoding="utf-8"),
        }

    started = time.perf_counter()
    manifest, template, test_paths = _load_data()
    labels = manifest.label.to_numpy()
    folds = _make_folds(manifest)
    anchor_report = json.loads((anchor / "metrics.json").read_text(encoding="utf-8"))
    anchor_oof = np.load(anchor / "oof_probabilities.npz", allow_pickle=False)
    class_logits = anchor_oof["logits"]
    binary_logits = anchor_oof["binary_logits"]
    if not np.isfinite(class_logits).all() or not np.isfinite(binary_logits).all():
        raise ValueError("FINAL/02 anchor lacks complete five-fold OOF predictions")
    if not np.array_equal(anchor_oof["labels"].astype(int), labels):
        raise ValueError("FINAL/02 OOF labels do not match official manifest")
    if not np.array_equal(anchor_oof["folds"].astype(int), folds):
        raise ValueError("FINAL/02 OOF folds do not match deterministic split")

    base = _softmax(class_logits)
    hierarchical = _hierarchical_probabilities(base, binary_logits)
    alpha = float(anchor_report["selected_alpha_3class"])
    anchor_validation_probability = alpha * base + (1 - alpha) * hierarchical
    test_probability_path = anchor / "test_probabilities.npz"
    if test_probability_path.exists():
        anchor_test_probability = np.load(
            test_probability_path, allow_pickle=False
        )["probabilities"]
    else:
        config = MODEL_CONFIGS["siglip2"]
        model = _build_model("siglip2", multitask=True)
        test_loader = _loader(
            test_paths,
            None,
            config["size"],
            False,
            config["batch"],
            kind="siglip2",
        )
        test_class_logits, test_binary_logits = [], []
        for fold in range(5):
            checkpoint = anchor / f"siglip2_fold{fold}_multi.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing FINAL/02 checkpoint: {checkpoint}")
            model.load_state_dict(
                torch.load(checkpoint, map_location="cuda", weights_only=True)
            )
            class_output, binary_output = _predict(
                model, test_loader, multitask=True
            )
            test_class_logits.append(class_output)
            test_binary_logits.append(binary_output)
        test_base = _softmax(np.mean(test_class_logits, axis=0))
        test_hierarchical = _hierarchical_probabilities(
            test_base, np.mean(test_binary_logits, axis=0)
        )
        anchor_test_probability = alpha * test_base + (1 - alpha) * test_hierarchical
        np.savez_compressed(
            test_probability_path,
            probabilities=anchor_test_probability.astype(np.float32),
        )
        del model
        torch.cuda.empty_cache()
    if anchor_test_probability.shape != (len(template), NUM_CLASSES):
        raise ValueError("Invalid FINAL/02 test probability shape")

    train_features = _extract_features(
        "dino", manifest.path.tolist(), output / "dino_train_features.npz"
    )
    test_features = _extract_features(
        "dino", test_paths, output / "dino_test_features.npz"
    )
    binary_rows = labels != 1
    baseline_prediction = anchor_validation_probability.argmax(1)
    baseline_metrics = _score(labels, baseline_prediction)
    baseline_fold_scores = [
        _score(labels[folds == fold], baseline_prediction[folds == fold])["macro_f1"]
        for fold in range(5)
    ]
    anchor_margin = np.abs(
        anchor_validation_probability[:, 0] - anchor_validation_probability[:, 2]
    )
    test_margin = np.abs(anchor_test_probability[:, 0] - anchor_test_probability[:, 2])

    selected = None
    probability_cache = {}
    for regularization in (0.1, 0.3, 1.0, 3.0):
        validation_q = np.zeros(len(manifest), dtype=np.float64)
        fold_test_q = []
        for fold in range(5):
            train_mask = binary_rows & (folds != fold)
            validation_mask = folds == fold
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=regularization,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=SEED,
                ),
            )
            classifier.fit(
                train_features[train_mask], (labels[train_mask] == 0).astype(int)
            )
            validation_q[validation_mask] = classifier.predict_proba(
                train_features[validation_mask]
            )[:, 1]
            fold_test_q.append(classifier.predict_proba(test_features)[:, 1])
        test_q = np.mean(fold_test_q, axis=0)
        probability_cache[regularization] = (validation_q, test_q)

        for direction in ("both", "2_to_0", "0_to_2"):
            for confidence in (0.80, 0.85, 0.90, 0.925, 0.95):
                for margin in (0.05, 0.10, 0.20, 0.30):
                    prediction = baseline_prediction.copy()
                    uncertain = anchor_margin <= margin
                    if direction in {"both", "2_to_0"}:
                        prediction[
                            uncertain
                            & (prediction == 2)
                            & (validation_q >= confidence)
                        ] = 0
                    if direction in {"both", "0_to_2"}:
                        prediction[
                            uncertain
                            & (prediction == 0)
                            & (validation_q <= 1 - confidence)
                        ] = 2
                    metrics = _score(labels, prediction)
                    fold_scores = [
                        _score(labels[folds == fold], prediction[folds == fold])[
                            "macro_f1"
                        ]
                        for fold in range(5)
                    ]
                    non_degrading = sum(
                        candidate + 1e-12 >= baseline
                        for candidate, baseline in zip(
                            fold_scores, baseline_fold_scores
                        )
                    )
                    changed = int(np.sum(prediction != baseline_prediction))
                    valid = (
                        metrics["macro_f1"] > baseline_metrics["macro_f1"]
                        and np.mean(fold_scores) + 1e-12
                        >= np.mean(baseline_fold_scores)
                        and non_degrading >= 4
                    )
                    candidate = (
                        int(valid),
                        metrics["macro_f1"],
                        non_degrading,
                        -changed,
                        confidence,
                        -margin,
                        regularization,
                        direction,
                        margin,
                        metrics,
                        fold_scores,
                        prediction,
                    )
                    if selected is None or candidate[:6] > selected[:6]:
                        selected = candidate

    if selected[0] == 0:
        selected_prediction = baseline_prediction
        test_prediction = anchor_test_probability.argmax(1)
        selection = {
            "accepted": False,
            "reason": "no OOF candidate passed aggregate and fold guards",
        }
    else:
        regularization, direction, margin = selected[6], selected[7], selected[8]
        validation_q, test_q = probability_cache[regularization]
        confidence = selected[4]
        selected_prediction = selected[11]
        test_prediction = anchor_test_probability.argmax(1)
        uncertain = test_margin <= margin
        if direction in {"both", "2_to_0"}:
            test_prediction[
                uncertain & (test_prediction == 2) & (test_q >= confidence)
            ] = 0
        if direction in {"both", "0_to_2"}:
            test_prediction[
                uncertain & (test_prediction == 0) & (test_q <= 1 - confidence)
            ] = 2
        selection = {
            "accepted": True,
            "C": regularization,
            "direction": direction,
            "confidence": confidence,
            "anchor_margin_max": margin,
            "changed_validation_rows": int(
                np.sum(selected_prediction != baseline_prediction)
            ),
            "non_degrading_folds": int(selected[2]),
        }

    submission = template[["id"]].copy()
    submission["predicted"] = test_prediction.astype(int)
    submission.to_csv(submission_path, index=False)
    np.savez_compressed(
        output / "specialist_probabilities.npz",
        labels=labels,
        folds=folds,
        anchor_validation_probabilities=anchor_validation_probability.astype(np.float32),
        anchor_test_probabilities=anchor_test_probability.astype(np.float32),
        selected_validation_predictions=selected_prediction.astype(np.int8),
    )
    report = {
        "architecture": "FINAL/02 hierarchical SigLIP2 + frozen DINO specialist",
        "test_labels_used": False,
        "seed": SEED,
        "rows_train": int(len(manifest)),
        "rows_test": int(len(template)),
        "anchor": str(anchor),
        "specialist_model": MODEL_CONFIGS["dino"],
        "baseline_validation": baseline_metrics,
        "baseline_fold_macro_f1": baseline_fold_scores,
        "selection": selection,
        "selected_validation": _score(labels, selected_prediction),
        "selected_fold_macro_f1": [
            _score(labels[folds == fold], selected_prediction[folds == fold])[
                "macro_f1"
            ]
            for fold in range(5)
        ],
        "changed_test_rows": int(
            np.sum(test_prediction != anchor_test_probability.argmax(1))
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "cached": False,
        "run_root": str(output),
        "metrics": metrics_path.read_text(encoding="utf-8"),
        "submission": submission_path.read_text(encoding="utf-8"),
    }


def run_bidirectional_dino_specialist(force: bool = False):
    """Fit separate frozen-DINO gates for 0->2 and 2->0 corrections.

    Selection uses only five-fold OOF predictions from official train data. The
    FINAL/02 full anchor and frozen DINO descriptors are reused from the shared
    Modal cache; no test label is loaded.
    """

    modules = _imports()
    globals().update({key: value for key, value in modules.items() if not key.startswith("_")})
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    _seed_everything(SEED)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    output = Path(CACHE_ROOT) / f"bidirectional_dino_boundary_experts_seed{SEED}"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    submission_path = output / "submission.csv"
    validation_path = output / "validation_predictions.csv"
    test_path = output / "test_predictions.csv"
    probability_path = output / "expert_probabilities.npz"
    completed = (
        metrics_path,
        submission_path,
        validation_path,
        test_path,
        probability_path,
    )
    if all(path.exists() for path in completed) and not force:
        return {
            "cached": True,
            "run_root": str(output),
            "metrics": metrics_path.read_text(encoding="utf-8"),
            "submission": submission_path.read_text(encoding="utf-8"),
        }

    # Reuse FINAL/02 anchor probabilities and frozen-DINO descriptors. Running
    # the old specialist here only materializes missing shared cache files; its
    # selected predictions are never used by this experiment.
    shared = Path(CACHE_ROOT) / f"final02_frozen_dino_specialist_seed{SEED}"
    shared_required = (
        shared / "specialist_probabilities.npz",
        shared / "dino_train_features.npz",
        shared / "dino_test_features.npz",
    )
    if not all(path.exists() for path in shared_required):
        run_frozen_dino_specialist(force=False)
    if not all(path.exists() for path in shared_required):
        run_frozen_dino_specialist(force=True)
    missing = [str(path) for path in shared_required if not path.exists()]
    if missing:
        raise FileNotFoundError("Could not materialize shared DINO cache: " + ", ".join(missing))

    started = time.perf_counter()
    manifest, template, _ = _load_data()
    labels = manifest.label.to_numpy(dtype=int)
    folds = _make_folds(manifest)
    shared_probabilities = np.load(
        shared / "specialist_probabilities.npz", allow_pickle=False
    )
    if not np.array_equal(shared_probabilities["labels"].astype(int), labels):
        raise ValueError("Shared specialist labels do not match official manifest")
    if not np.array_equal(shared_probabilities["folds"].astype(int), folds):
        raise ValueError("Shared specialist folds do not match deterministic split")

    anchor_validation = shared_probabilities[
        "anchor_validation_probabilities"
    ].astype(np.float64)
    anchor_test = shared_probabilities["anchor_test_probabilities"].astype(np.float64)
    train_features = np.load(
        shared / "dino_train_features.npz", allow_pickle=False
    )["features"]
    test_features = np.load(
        shared / "dino_test_features.npz", allow_pickle=False
    )["features"]
    if anchor_validation.shape != (len(manifest), NUM_CLASSES):
        raise ValueError("Invalid anchor OOF probability shape")
    if anchor_test.shape != (len(template), NUM_CLASSES):
        raise ValueError("Invalid anchor test probability shape")
    if len(train_features) != len(manifest) or len(test_features) != len(template):
        raise ValueError("Frozen DINO feature rows do not match official data")

    baseline_prediction = anchor_validation.argmax(1)
    baseline_test_prediction = anchor_test.argmax(1)
    baseline_metrics = _score(labels, baseline_prediction)
    baseline_fold_scores = [
        _score(labels[folds == fold], baseline_prediction[folds == fold])["macro_f1"]
        for fold in range(5)
    ]
    validation_margin = np.abs(anchor_validation[:, 0] - anchor_validation[:, 2])
    test_margin = np.abs(anchor_test[:, 0] - anchor_test[:, 2])
    binary_rows = labels != 1

    direction_best = {"0_to_2": None, "2_to_0": None}
    probability_cache = {}
    regularizations = (0.03, 0.1, 0.3, 1.0, 3.0)
    confidences = (0.75, 0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975)
    margins = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)

    directions = {"0_to_2": (0, 2), "2_to_0": (2, 0)}
    for direction, (source_class, target_class) in directions.items():
        for regularization in regularizations:
            validation_correction_probability = np.zeros(
                len(manifest), dtype=np.float64
            )
            fold_test_correction_probability = []
            for fold in range(5):
                # Each expert learns only from rows on its own side of the
                # anchor boundary. This makes the two residual tasks distinct.
                train_mask = (
                    binary_rows
                    & (folds != fold)
                    & (baseline_prediction == source_class)
                )
                validation_mask = folds == fold
                target = (labels[train_mask] == target_class).astype(int)
                if np.unique(target).size != 2:
                    raise RuntimeError(
                        f"Fold {fold} lacks both targets for expert {direction}"
                    )
                classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=regularization,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=SEED,
                    ),
                )
                classifier.fit(train_features[train_mask], target)
                validation_correction_probability[validation_mask] = (
                    classifier.predict_proba(train_features[validation_mask])[:, 1]
                )
                fold_test_correction_probability.append(
                    classifier.predict_proba(test_features)[:, 1]
                )
            test_correction_probability = np.mean(
                fold_test_correction_probability, axis=0
            )
            probability_cache[(direction, regularization)] = (
                validation_correction_probability,
                test_correction_probability,
            )

            for confidence in confidences:
                for margin in margins:
                    prediction = baseline_prediction.copy()
                    uncertain = validation_margin <= margin
                    changed_mask = (
                        uncertain
                        & (prediction == source_class)
                        & (validation_correction_probability >= confidence)
                    )
                    prediction[changed_mask] = target_class

                    changed = int(changed_mask.sum())
                    if changed == 0:
                        continue
                    metrics = _score(labels, prediction)
                    fold_scores = [
                        _score(labels[folds == fold], prediction[folds == fold])[
                            "macro_f1"
                        ]
                        for fold in range(5)
                    ]
                    non_degrading = sum(
                        candidate + 1e-12 >= baseline
                        for candidate, baseline in zip(
                            fold_scores, baseline_fold_scores
                        )
                    )
                    valid = (
                        metrics["macro_f1"] > baseline_metrics["macro_f1"]
                        and np.mean(fold_scores) + 1e-12
                        >= np.mean(baseline_fold_scores)
                        and non_degrading >= 4
                    )
                    rank = (
                        int(valid),
                        metrics["macro_f1"],
                        non_degrading,
                        -changed,
                        confidence,
                        -margin,
                        -regularization,
                    )
                    current = direction_best[direction]
                    if current is None or rank > current["rank"]:
                        direction_best[direction] = {
                            "rank": rank,
                            "accepted": bool(valid),
                            "direction": direction,
                            "C": regularization,
                            "confidence": confidence,
                            "anchor_margin_max": margin,
                            "changed_validation_rows": changed,
                            "non_degrading_folds": int(non_degrading),
                            "metrics": metrics,
                            "fold_scores": fold_scores,
                            "validation_prediction": prediction,
                        }

    accepted = [
        candidate
        for candidate in direction_best.values()
        if candidate is not None and candidate["accepted"]
    ]
    selected_prediction = baseline_prediction.copy()
    for candidate in accepted:
        candidate_prediction = candidate["validation_prediction"]
        changed = candidate_prediction != baseline_prediction
        selected_prediction[changed] = candidate_prediction[changed]

    selected_metrics = _score(labels, selected_prediction)
    selected_fold_scores = [
        _score(labels[folds == fold], selected_prediction[folds == fold])["macro_f1"]
        for fold in range(5)
    ]
    selected_non_degrading = sum(
        candidate + 1e-12 >= baseline
        for candidate, baseline in zip(selected_fold_scores, baseline_fold_scores)
    )
    union_accepted = bool(
        accepted
        and selected_metrics["macro_f1"] > baseline_metrics["macro_f1"]
        and np.mean(selected_fold_scores) + 1e-12 >= np.mean(baseline_fold_scores)
        and selected_non_degrading >= 4
    )
    if not union_accepted:
        if accepted:
            winner = max(accepted, key=lambda candidate: candidate["rank"])
            accepted = [winner]
            selected_prediction = winner["validation_prediction"].copy()
            selected_metrics = winner["metrics"]
            selected_fold_scores = winner["fold_scores"]
            selected_non_degrading = winner["non_degrading_folds"]
        else:
            selected_prediction = baseline_prediction.copy()
            selected_metrics = baseline_metrics
            selected_fold_scores = baseline_fold_scores
            selected_non_degrading = 5

    test_prediction = baseline_test_prediction.copy()
    selected_validation_probability = {}
    selected_test_probability = {}
    for candidate in accepted:
        direction = candidate["direction"]
        validation_probability, test_probability = probability_cache[
            (direction, candidate["C"])
        ]
        selected_validation_probability[direction] = validation_probability
        selected_test_probability[direction] = test_probability
        uncertain = test_margin <= candidate["anchor_margin_max"]
        source_class, target_class = directions[direction]
        change = (
            uncertain
            & (test_prediction == source_class)
            & (test_probability >= candidate["confidence"])
        )
        test_prediction[change] = target_class

    submission = template[["id"]].copy()
    submission["predicted"] = test_prediction.astype(int)
    submission.to_csv(submission_path, index=False)

    validation_frame = pd.DataFrame(
        {
            "row_index": np.arange(len(manifest)),
            "path": manifest.path,
            "label": labels,
            "fold": folds,
            "anchor_prediction": baseline_prediction,
            "selected_prediction": selected_prediction,
            "changed": selected_prediction != baseline_prediction,
        }
    )
    validation_frame.to_csv(validation_path, index=False)
    test_frame = template[["id"]].copy()
    test_frame["anchor_prediction"] = baseline_test_prediction
    test_frame["selected_prediction"] = test_prediction
    test_frame["changed"] = test_prediction != baseline_test_prediction
    test_frame.to_csv(test_path, index=False)

    nan_validation = np.full(len(manifest), np.nan, dtype=np.float32)
    nan_test = np.full(len(template), np.nan, dtype=np.float32)
    np.savez_compressed(
        probability_path,
        labels=labels.astype(np.int8),
        folds=folds.astype(np.int8),
        anchor_validation_probabilities=anchor_validation.astype(np.float32),
        anchor_test_probabilities=anchor_test.astype(np.float32),
        expert_0_to_2_validation_correction_probability=selected_validation_probability.get(
            "0_to_2", nan_validation
        ).astype(np.float32),
        expert_0_to_2_test_correction_probability=selected_test_probability.get(
            "0_to_2", nan_test
        ).astype(np.float32),
        expert_2_to_0_validation_correction_probability=selected_validation_probability.get(
            "2_to_0", nan_validation
        ).astype(np.float32),
        expert_2_to_0_test_correction_probability=selected_test_probability.get(
            "2_to_0", nan_test
        ).astype(np.float32),
        selected_validation_predictions=selected_prediction.astype(np.int8),
        selected_test_predictions=test_prediction.astype(np.int8),
    )

    def serializable(candidate):
        if candidate is None:
            return {"accepted": False, "reason": "no candidate changed any row"}
        return {
            key: value
            for key, value in candidate.items()
            if key not in {"rank", "validation_prediction"}
        }

    report = {
        "architecture": "FINAL/02 hierarchical SigLIP2 + bidirectional frozen-DINO boundary experts",
        "test_labels_used": False,
        "seed": SEED,
        "rows_train": int(len(manifest)),
        "rows_test": int(len(template)),
        "cache_sources": {
            "anchor": str(
                Path(CACHE_ROOT) / f"hierarchical_siglip2_full5fold_seed{SEED}"
            ),
            "shared_dino_descriptors": str(shared),
        },
        "specialist_model": MODEL_CONFIGS["dino"],
        "baseline_validation": baseline_metrics,
        "baseline_fold_macro_f1": baseline_fold_scores,
        "direction_candidates": {
            direction: serializable(candidate)
            for direction, candidate in direction_best.items()
        },
        "active_directions": [candidate["direction"] for candidate in accepted],
        "union_guard_passed": union_accepted,
        "selected_validation": selected_metrics,
        "selected_fold_macro_f1": selected_fold_scores,
        "selected_non_degrading_folds": int(selected_non_degrading),
        "changed_validation_rows": int(
            np.sum(selected_prediction != baseline_prediction)
        ),
        "changed_test_rows": int(np.sum(test_prediction != baseline_test_prediction)),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "cached": False,
        "run_root": str(output),
        "metrics": metrics_path.read_text(encoding="utf-8"),
        "submission": submission_path.read_text(encoding="utf-8"),
    }


def run_hierarchical_patch_dino_consensus(
    force: bool = False,
    retrain_patch: bool = False,
    from_scratch: bool = False,
):
    """Full five-fold Hierarchical + Patch-MIL + frozen-DINO consensus."""

    modules = _imports()
    globals().update({key: value for key, value in modules.items() if not key.startswith("_")})
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    _seed_everything(SEED)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    output = Path(CACHE_ROOT) / f"final08_clean_seed{SEED}"
    if from_scratch and output.exists():
        resolved = output.resolve()
        cache_root = Path(CACHE_ROOT).resolve()
        if resolved.parent != cache_root or not resolved.name.startswith("final08_clean_"):
            raise RuntimeError(f"Refusing to clear unsafe cache path: {resolved}")
        shutil.rmtree(resolved)
    if from_scratch:
        force = True
        retrain_patch = True
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    submission_path = output / "submission.csv"
    validation_path = output / "validation_predictions.csv"
    test_path = output / "test_predictions.csv"
    router_path = output / "router_probabilities.npz"
    stage_submission_paths = {
        "control": output / "submission_01_hierarchical_dino_control.csv",
        "patch_dino_consensus": output / "submission_02_patch_dino_consensus.csv",
        "tri_encoder_router": output / "submission_03_tri_encoder_router.csv",
    }
    complete = (
        metrics_path,
        submission_path,
        validation_path,
        test_path,
        router_path,
        *stage_submission_paths.values(),
    )
    if (
        all(path.exists() for path in complete)
        and not force
        and not retrain_patch
        and not from_scratch
    ):
        return {
            "cached": True,
            "run_root": str(output),
            "metrics": metrics_path.read_text(encoding="utf-8"),
            "submission": submission_path.read_text(encoding="utf-8"),
            "stage_submissions": {
                name: path.read_text(encoding="utf-8")
                for name, path in stage_submission_paths.items()
            },
        }

    started = time.perf_counter()
    manifest, template, test_paths = _load_data(
        output / "official_train_manifest.csv"
    )
    labels = manifest.label.to_numpy(dtype=int)
    folds = _make_folds(manifest)

    anchor = output / "01_hierarchical_anchor"
    anchor.mkdir(parents=True, exist_ok=True)
    anchor_required = (
        anchor / "metrics.json",
        anchor / "submission.csv",
        anchor / "oof_probabilities.npz",
        anchor / "test_probabilities.npz",
    )
    if not all(path.exists() for path in anchor_required):
        anchor_started = time.perf_counter()
        anchor_result = _run_hierarchical(
            manifest,
            folds,
            test_paths,
            list(range(5)),
            anchor,
            reuse_checkpoints=not from_scratch,
        )
        anchor_predictions = anchor_result.pop("test_predictions").astype(int)
        anchor_submission = template[["id"]].copy()
        anchor_submission["predicted"] = anchor_predictions
        anchor_submission.to_csv(anchor / "submission.csv", index=False)
        anchor_report = {
            "architecture": "FINAL08 clean hierarchical SigLIP2 anchor",
            "test_labels_used": False,
            "seed": SEED,
            "active_folds": list(range(5)),
            "runtime_seconds": float(time.perf_counter() - anchor_started),
            **anchor_result,
        }
        (anchor / "metrics.json").write_text(
            json.dumps(anchor_report, indent=2), encoding="utf-8"
        )

    anchor_report = json.loads((anchor / "metrics.json").read_text(encoding="utf-8"))
    anchor_oof = np.load(anchor / "oof_probabilities.npz", allow_pickle=False)
    if not np.array_equal(anchor_oof["labels"].astype(int), labels):
        raise ValueError("FINAL08 anchor labels do not match official manifest")
    if not np.array_equal(anchor_oof["folds"].astype(int), folds):
        raise ValueError("FINAL08 anchor folds do not match deterministic split")
    anchor_base = _softmax(anchor_oof["logits"].astype(np.float64))
    anchor_hierarchical = _hierarchical_probabilities(
        anchor_base, anchor_oof["binary_logits"].astype(np.float64)
    )
    anchor_alpha = float(anchor_report["selected_alpha_3class"])
    anchor_validation_probability = (
        anchor_alpha * anchor_base + (1.0 - anchor_alpha) * anchor_hierarchical
    )
    anchor_test_probability = np.load(
        anchor / "test_probabilities.npz", allow_pickle=False
    )["probabilities"].astype(np.float64)

    # All DINO artifacts live inside FINAL08. Only public model weights may be
    # served from the shared download cache.
    dino_source = output / "02_frozen_dino"
    dino_source.mkdir(parents=True, exist_ok=True)

    # Train only missing Patch-MIL folds. This cache uses the candidate loss
    # weights (0.25 global binary, 0.15 patch binary), distinct from FINAL/05.
    patch_root = output / "03_patch_mil"
    patch_root.mkdir(parents=True, exist_ok=True)
    patch_required = (
        patch_root / "metrics.json",
        patch_root / "submission.csv",
        patch_root / "oof_probabilities.npz",
        patch_root / "test_probabilities.npz",
    )
    if retrain_patch or not all(path.exists() for path in patch_required):
        patch_started = time.perf_counter()
        patch_result = _run_patch_mil(
            manifest,
            folds,
            test_paths,
            list(range(5)),
            patch_root,
            reuse_checkpoints=not retrain_patch,
            global_binary_weight=0.25,
            patch_binary_weight=0.15,
        )
        patch_predictions = patch_result.pop("test_predictions").astype(int)
        patch_submission = template[["id"]].copy()
        patch_submission["predicted"] = patch_predictions
        patch_submission.to_csv(patch_root / "submission.csv", index=False)
        patch_report = {
            "architecture": "full five-fold Patch-MIL residual component",
            "test_labels_used": False,
            "seed": SEED,
            "active_folds": list(range(5)),
            "loss_weights": {
                "global_ce": 1.0,
                "global_binary": 0.25,
                "patch_binary": 0.15,
                "supcon": 0.05,
                "flip_consistency": 0.02,
            },
            "runtime_seconds": float(time.perf_counter() - patch_started),
            **patch_result,
        }
        (patch_root / "metrics.json").write_text(
            json.dumps(patch_report, indent=2), encoding="utf-8"
        )

    patch_oof = np.load(patch_root / "oof_probabilities.npz", allow_pickle=False)
    patch_test = np.load(patch_root / "test_probabilities.npz", allow_pickle=False)
    required_oof_keys = {"labels", "folds", "patch_binary_logits", "active_mask"}
    required_test_keys = {"patch_binary_logits"}
    if not required_oof_keys.issubset(patch_oof.files) or not required_test_keys.issubset(
        patch_test.files
    ):
        raise ValueError(
            "Legacy/incomplete Patch-MIL cache. Rerun with --retrain-patch."
        )
    if not np.array_equal(patch_oof["labels"].astype(int), labels):
        raise ValueError("Patch-MIL OOF labels do not match official manifest")
    if not np.array_equal(patch_oof["folds"].astype(int), folds):
        raise ValueError("Patch-MIL OOF folds do not match deterministic split")
    if not patch_oof["active_mask"].all():
        raise ValueError("Patch-MIL cache is not full five-fold OOF")

    patch_validation_q0 = 1.0 / (
        1.0 + np.exp(-patch_oof["patch_binary_logits"].astype(np.float64))
    )
    patch_test_q0 = 1.0 / (
        1.0 + np.exp(-patch_test["patch_binary_logits"].astype(np.float64))
    )
    train_features = _extract_features(
        "dino", manifest.path.tolist(), dino_source / "dino_train_features.npz"
    )
    test_features = _extract_features(
        "dino", test_paths, dino_source / "dino_test_features.npz"
    )

    regularizations = np.asarray((0.03, 0.1, 0.3, 1.0, 3.0), dtype=np.float64)
    dino_cache_path = dino_source / "dino_crossfit_probabilities.npz"
    if dino_cache_path.exists():
        dino_cache = np.load(dino_cache_path, allow_pickle=False)
        if not np.array_equal(dino_cache["regularizations"], regularizations):
            raise ValueError("DINO probability cache uses incompatible C grid")
        dino_validation_by_c = dino_cache["validation_q0"].astype(np.float64)
        dino_test_by_c = dino_cache["test_q0"].astype(np.float64)
    else:
        dino_validation_by_c = np.zeros(
            (len(regularizations), len(manifest)), dtype=np.float32
        )
        dino_test_by_c = np.zeros(
            (len(regularizations), len(template)), dtype=np.float32
        )
        binary = labels != 1
        for index, regularization in enumerate(regularizations):
            fold_test = []
            for fold in range(5):
                train_mask = binary & (folds != fold)
                valid_mask = folds == fold
                classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=float(regularization),
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=SEED,
                    ),
                )
                classifier.fit(
                    train_features[train_mask], (labels[train_mask] == 0).astype(int)
                )
                dino_validation_by_c[index, valid_mask] = classifier.predict_proba(
                    train_features[valid_mask]
                )[:, 1]
                fold_test.append(classifier.predict_proba(test_features)[:, 1])
            dino_test_by_c[index] = np.mean(fold_test, axis=0)
        np.savez_compressed(
            dino_cache_path,
            regularizations=regularizations,
            validation_q0=dino_validation_by_c,
            test_q0=dino_test_by_c,
        )

    anchor_margin = np.abs(
        anchor_validation_probability[:, 0] - anchor_validation_probability[:, 2]
    )
    test_margin = np.abs(anchor_test_probability[:, 0] - anchor_test_probability[:, 2])
    anchor_entropy = -np.sum(
        anchor_validation_probability
        * np.log(np.clip(anchor_validation_probability, 1e-12, 1.0)),
        axis=1,
    ) / np.log(NUM_CLASSES)
    test_entropy = -np.sum(
        anchor_test_probability
        * np.log(np.clip(anchor_test_probability, 1e-12, 1.0)),
        axis=1,
    ) / np.log(NUM_CLASSES)

    # Recreate the frozen-DINO control entirely inside FINAL08.
    anchor_prediction = anchor_validation_probability.argmax(1)
    anchor_test_prediction = anchor_test_probability.argmax(1)
    anchor_metrics = _score(labels, anchor_prediction)
    anchor_fold_scores = [
        _score(labels[folds == fold], anchor_prediction[folds == fold])["macro_f1"]
        for fold in range(5)
    ]
    dino_selection = None
    for c_index, regularization in enumerate(regularizations):
        validation_q0 = dino_validation_by_c[c_index]
        for direction in ("both", "2_to_0", "0_to_2"):
            for confidence in (0.80, 0.85, 0.90, 0.925, 0.95):
                for margin in (0.05, 0.10, 0.20, 0.30):
                    prediction = anchor_prediction.copy()
                    uncertain = anchor_margin <= margin
                    if direction in {"both", "2_to_0"}:
                        prediction[
                            uncertain
                            & (prediction == 2)
                            & (validation_q0 >= confidence)
                        ] = 0
                    if direction in {"both", "0_to_2"}:
                        prediction[
                            uncertain
                            & (prediction == 0)
                            & (validation_q0 <= 1.0 - confidence)
                        ] = 2
                    changed = int(np.sum(prediction != anchor_prediction))
                    if changed == 0:
                        continue
                    metrics = _score(labels, prediction)
                    fold_scores = [
                        _score(labels[folds == fold], prediction[folds == fold])[
                            "macro_f1"
                        ]
                        for fold in range(5)
                    ]
                    non_degrading = sum(
                        candidate + 1e-12 >= baseline
                        for candidate, baseline in zip(
                            fold_scores, anchor_fold_scores
                        )
                    )
                    valid = (
                        metrics["macro_f1"] > anchor_metrics["macro_f1"]
                        and non_degrading >= 4
                    )
                    rank = (
                        int(valid),
                        metrics["macro_f1"],
                        non_degrading,
                        -changed,
                        confidence,
                        -margin,
                    )
                    if dino_selection is None or rank > dino_selection["rank"]:
                        dino_selection = {
                            "rank": rank,
                            "accepted": bool(valid),
                            "C": float(regularization),
                            "C_index": c_index,
                            "direction": direction,
                            "confidence": float(confidence),
                            "anchor_margin_max": float(margin),
                            "changed_validation_rows": changed,
                            "non_degrading_folds": int(non_degrading),
                            "metrics": metrics,
                            "fold_scores": fold_scores,
                            "prediction": prediction,
                        }

    baseline_prediction = anchor_prediction.copy()
    baseline_test_prediction = anchor_test_prediction.copy()
    if dino_selection is not None and dino_selection["accepted"]:
        baseline_prediction = dino_selection["prediction"].copy()
        test_q0 = dino_test_by_c[dino_selection["C_index"]]
        uncertain = test_margin <= dino_selection["anchor_margin_max"]
        if dino_selection["direction"] in {"both", "2_to_0"}:
            baseline_test_prediction[
                uncertain
                & (baseline_test_prediction == 2)
                & (test_q0 >= dino_selection["confidence"])
            ] = 0
        if dino_selection["direction"] in {"both", "0_to_2"}:
            baseline_test_prediction[
                uncertain
                & (baseline_test_prediction == 0)
                & (test_q0 <= 1.0 - dino_selection["confidence"])
            ] = 2

    dino_submission = template[["id"]].copy()
    dino_submission["predicted"] = baseline_test_prediction
    dino_submission.to_csv(dino_source / "submission.csv", index=False)
    np.savez_compressed(
        dino_source / "specialist_probabilities.npz",
        labels=labels.astype(np.int8),
        folds=folds.astype(np.int8),
        anchor_validation_probabilities=anchor_validation_probability.astype(np.float32),
        anchor_test_probabilities=anchor_test_probability.astype(np.float32),
        selected_validation_predictions=baseline_prediction.astype(np.int8),
        selected_test_predictions=baseline_test_prediction.astype(np.int8),
    )
    dino_report_selection = None
    if dino_selection is not None:
        dino_report_selection = {
            key: value
            for key, value in dino_selection.items()
            if key not in {"rank", "C_index", "prediction"}
        }
    (dino_source / "metrics.json").write_text(
        json.dumps(
            {
                "architecture": "FINAL08 clean frozen-DINO residual control",
                "test_labels_used": False,
                "seed": SEED,
                "anchor_validation": anchor_metrics,
                "selection": dino_report_selection,
                "selected_validation": _score(labels, baseline_prediction),
                "changed_test_rows": int(
                    np.sum(baseline_test_prediction != anchor_test_prediction)
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    baseline_metrics = _score(labels, baseline_prediction)
    baseline_fold_scores = [
        _score(labels[folds == fold], baseline_prediction[folds == fold])["macro_f1"]
        for fold in range(5)
    ]
    nested_prediction = baseline_prediction.copy()
    outer_selections = []
    margins = (0.05, 0.10, 0.20, 0.30, 0.40)
    confidences = (0.70, 0.80, 0.90, 0.95)

    for outer_fold in range(5):
        fit_mask = folds != outer_fold
        holdout_mask = folds == outer_fold
        fit_folds = [fold for fold in range(5) if fold != outer_fold]
        baseline_fit_metrics = _score(labels[fit_mask], baseline_prediction[fit_mask])
        baseline_fit_folds = {
            fold: _score(
                labels[folds == fold], baseline_prediction[folds == fold]
            )["macro_f1"]
            for fold in fit_folds
        }
        entropy_candidates = np.quantile(
            anchor_entropy[fit_mask & (baseline_prediction != 1)],
            (0.0, 0.25, 0.50),
        )
        best = None
        for c_index, regularization in enumerate(regularizations):
            dino_q0 = dino_validation_by_c[c_index]
            for margin in margins:
                for patch_confidence in confidences:
                    for dino_confidence in confidences:
                        for entropy_min in entropy_candidates:
                            parameters = {
                                "C": float(regularization),
                                "margin_max": float(margin),
                                "patch_confidence": float(patch_confidence),
                                "dino_confidence": float(dino_confidence),
                                "entropy_min": float(entropy_min),
                            }
                            gate = _tri_consensus_gate(
                                baseline_prediction,
                                anchor_margin,
                                anchor_entropy,
                                patch_validation_q0,
                                dino_q0,
                                parameters,
                            ) & fit_mask
                            changed = int(gate.sum())
                            if changed == 0:
                                continue
                            prediction = baseline_prediction.copy()
                            prediction[gate] = 2 - prediction[gate]
                            metrics = _score(labels[fit_mask], prediction[fit_mask])
                            fold_scores = {
                                fold: _score(
                                    labels[folds == fold], prediction[folds == fold]
                                )["macro_f1"]
                                for fold in fit_folds
                            }
                            non_degrading = sum(
                                fold_scores[fold] + 1e-12
                                >= baseline_fit_folds[fold]
                                for fold in fit_folds
                            )
                            fixes = int(
                                np.sum(
                                    gate
                                    & (baseline_prediction != labels)
                                    & (prediction == labels)
                                )
                            )
                            harms = int(
                                np.sum(
                                    gate
                                    & (baseline_prediction == labels)
                                    & (prediction != labels)
                                )
                            )
                            valid = (
                                metrics["macro_f1"]
                                > baseline_fit_metrics["macro_f1"]
                                and non_degrading >= 3
                                and fixes > harms
                            )
                            precision = fixes / changed
                            rank = (
                                int(valid),
                                metrics["macro_f1"],
                                non_degrading,
                                precision,
                                -changed,
                            )
                            if best is None or rank > best["rank"]:
                                best = {
                                    "rank": rank,
                                    "accepted": bool(valid),
                                    "parameters": parameters,
                                    "fit_macro_f1": metrics["macro_f1"],
                                    "fit_non_degrading_folds": int(non_degrading),
                                    "fit_changed": changed,
                                    "fit_fixes": fixes,
                                    "fit_harms": harms,
                                }

        if best is not None and best["accepted"]:
            c_index = int(
                np.argmin(np.abs(regularizations - best["parameters"]["C"]))
            )
            holdout_gate = _tri_consensus_gate(
                baseline_prediction,
                anchor_margin,
                anchor_entropy,
                patch_validation_q0,
                dino_validation_by_c[c_index],
                best["parameters"],
            ) & holdout_mask
            nested_prediction[holdout_gate] = 2 - nested_prediction[holdout_gate]
            best["holdout_changed"] = int(holdout_gate.sum())
            best["holdout_fixes"] = int(
                np.sum(
                    holdout_gate
                    & (baseline_prediction != labels)
                    & (nested_prediction == labels)
                )
            )
            best["holdout_harms"] = int(
                np.sum(
                    holdout_gate
                    & (baseline_prediction == labels)
                    & (nested_prediction != labels)
                )
            )
        else:
            best = {
                "accepted": False,
                "reason": "no outer-train candidate passed guard",
                "holdout_changed": 0,
                "holdout_fixes": 0,
                "holdout_harms": 0,
            }
        best.pop("rank", None)
        best["outer_fold"] = outer_fold
        outer_selections.append(best)

    nested_metrics = _score(labels, nested_prediction)
    nested_fold_scores = [
        _score(labels[folds == fold], nested_prediction[folds == fold])["macro_f1"]
        for fold in range(5)
    ]
    non_degrading = sum(
        candidate + 1e-12 >= baseline
        for candidate, baseline in zip(nested_fold_scores, baseline_fold_scores)
    )
    changed_mask = nested_prediction != baseline_prediction
    fixes = int(
        np.sum(
            changed_mask
            & (baseline_prediction != labels)
            & (nested_prediction == labels)
        )
    )
    harms = int(
        np.sum(
            changed_mask
            & (baseline_prediction == labels)
            & (nested_prediction != labels)
        )
    )
    passes_guard = bool(
        nested_metrics["macro_f1"] > baseline_metrics["macro_f1"]
        and non_degrading >= 4
        and nested_metrics["class_f1"][1] + 1e-12
        >= baseline_metrics["class_f1"][1]
        and fixes > harms
    )
    candidate_metrics = nested_metrics
    candidate_fold_scores = nested_fold_scores
    candidate_fixes = fixes
    candidate_harms = harms
    candidate_changed_rows = int(changed_mask.sum())

    accepted_parameters = [
        row["parameters"] for row in outer_selections if row.get("accepted")
    ]
    final_parameters = None
    test_prediction = baseline_test_prediction.copy()
    if passes_guard and accepted_parameters:
        median_c = float(np.median([row["C"] for row in accepted_parameters]))
        final_c_index = int(np.argmin(np.abs(regularizations - median_c)))
        final_parameters = {
            "C": float(regularizations[final_c_index]),
            "margin_max": float(
                np.median([row["margin_max"] for row in accepted_parameters])
            ),
            "patch_confidence": float(
                np.median([row["patch_confidence"] for row in accepted_parameters])
            ),
            "dino_confidence": float(
                np.median([row["dino_confidence"] for row in accepted_parameters])
            ),
            "entropy_min": float(
                np.median([row["entropy_min"] for row in accepted_parameters])
            ),
        }
        test_gate = _tri_consensus_gate(
            baseline_test_prediction,
            test_margin,
            test_entropy,
            patch_test_q0,
            dino_test_by_c[final_c_index],
            final_parameters,
        )
        test_prediction[test_gate] = 2 - test_prediction[test_gate]
    else:
        nested_prediction = baseline_prediction.copy()
        nested_metrics = baseline_metrics
        nested_fold_scores = baseline_fold_scores
        non_degrading = 5
        fixes = harms = 0

    consensus_validation_prediction = nested_prediction.copy()
    consensus_test_prediction = test_prediction.copy()
    consensus_metrics = nested_metrics
    consensus_fold_scores = nested_fold_scores

    # Improvement 3: frozen PE-Core diversity evidence plus a nested linear
    # residual router. PE-Core remains frozen; only official-train linear heads
    # and router are fitted.
    pe_train_features = _extract_features(
        "pe_core", manifest.path.tolist(), output / "pe_core_train_features.npz"
    )
    pe_test_features = _extract_features(
        "pe_core", test_paths, output / "pe_core_test_features.npz"
    )
    if len(pe_train_features) != len(manifest) or len(pe_test_features) != len(template):
        raise ValueError("PE-Core feature rows do not match official data")
    pe_probability_path = output / "pe_core_crossfit_probabilities.npz"
    if pe_probability_path.exists():
        pe_cache = np.load(pe_probability_path, allow_pickle=False)
        if not np.array_equal(pe_cache["labels"].astype(int), labels):
            raise ValueError("PE-Core OOF labels do not match official manifest")
        if not np.array_equal(pe_cache["folds"].astype(int), folds):
            raise ValueError("PE-Core OOF folds do not match deterministic split")
        pe_validation_probability = pe_cache["validation_probabilities"].astype(
            np.float64
        )
        pe_test_probability = pe_cache["test_probabilities"].astype(np.float64)
    else:
        pe_validation_probability = np.zeros(
            (len(manifest), NUM_CLASSES), dtype=np.float32
        )
        pe_fold_test = []
        for fold in range(5):
            train_mask = folds != fold
            valid_mask = folds == fold
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=SEED,
                ),
            )
            classifier.fit(pe_train_features[train_mask], labels[train_mask])
            pe_validation_probability[valid_mask] = classifier.predict_proba(
                pe_train_features[valid_mask]
            )
            pe_fold_test.append(classifier.predict_proba(pe_test_features))
        pe_test_probability = np.mean(pe_fold_test, axis=0)
        np.savez_compressed(
            pe_probability_path,
            labels=labels.astype(np.int8),
            folds=folds.astype(np.int8),
            validation_probabilities=pe_validation_probability,
            test_probabilities=pe_test_probability.astype(np.float32),
        )

    dino_router_index = int(np.argmin(np.abs(regularizations - 0.1)))
    dino_validation_q0 = dino_validation_by_c[dino_router_index]
    dino_test_q0 = dino_test_by_c[dino_router_index]

    def binary_to_three_class(q0, electronic_probability):
        remaining = 1.0 - electronic_probability
        return np.column_stack(
            [remaining * q0, electronic_probability, remaining * (1.0 - q0)]
        )

    dino_validation_probability = binary_to_three_class(
        dino_validation_q0, anchor_validation_probability[:, 1]
    )
    dino_test_probability = binary_to_three_class(
        dino_test_q0, anchor_test_probability[:, 1]
    )
    patch_validation_probability = binary_to_three_class(
        patch_validation_q0, anchor_validation_probability[:, 1]
    )
    patch_test_probability = binary_to_three_class(
        patch_test_q0, anchor_test_probability[:, 1]
    )
    residual_validation_features = _stack_features(
        [
            anchor_validation_probability,
            pe_validation_probability,
            dino_validation_probability,
            patch_validation_probability,
        ]
    )
    residual_test_features = _stack_features(
        [
            anchor_test_probability,
            pe_test_probability,
            dino_test_probability,
            patch_test_probability,
        ]
    )
    validation_expert_predictions = np.column_stack(
        [
            pe_validation_probability.argmax(1),
            dino_validation_probability.argmax(1),
            patch_validation_probability.argmax(1),
        ]
    )
    test_expert_predictions = np.column_stack(
        [
            pe_test_probability.argmax(1),
            dino_test_probability.argmax(1),
            patch_test_probability.argmax(1),
        ]
    )
    validation_disagreement = np.any(
        validation_expert_predictions
        != consensus_validation_prediction[:, None],
        axis=1,
    )
    test_disagreement = np.any(
        test_expert_predictions != consensus_test_prediction[:, None], axis=1
    )

    router_regularizations = (0.01, 0.03, 0.1, 0.3, 1.0)
    router_thresholds = (0.75, 0.80, 0.85, 0.90, 0.925, 0.95)
    router_margins = (0.10, 0.20, 0.30, 0.40)
    router_max_fractions = (0.001, 0.0025, 0.005)
    router_oof_q0 = np.full(len(manifest), 0.5, dtype=np.float64)
    router_prediction = consensus_validation_prediction.copy()
    router_outer_selections = []
    binary = labels != 1

    for outer_fold in range(5):
        fit_mask = folds != outer_fold
        holdout_mask = folds == outer_fold
        fit_indices = np.flatnonzero(fit_mask)
        holdout_indices = np.flatnonzero(holdout_mask)
        fit_folds = [fold for fold in range(5) if fold != outer_fold]
        baseline_fit = _score(
            labels[fit_mask], consensus_validation_prediction[fit_mask]
        )
        baseline_fit_folds = {
            fold: _score(
                labels[folds == fold],
                consensus_validation_prediction[folds == fold],
            )["macro_f1"]
            for fold in fit_folds
        }
        best = None
        inner_probabilities = {}
        for regularization in router_regularizations:
            inner_q0 = np.full(len(manifest), 0.5, dtype=np.float64)
            for inner_fold in fit_folds:
                train_mask = fit_mask & binary & (folds != inner_fold)
                valid_mask = fit_mask & (folds == inner_fold)
                classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=regularization,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=SEED,
                    ),
                )
                classifier.fit(
                    residual_validation_features[train_mask],
                    (labels[train_mask] == 0).astype(int),
                )
                inner_q0[valid_mask] = classifier.predict_proba(
                    residual_validation_features[valid_mask]
                )[:, 1]
            inner_probabilities[regularization] = inner_q0

            for threshold in router_thresholds:
                for margin_max in router_margins:
                    for max_fraction in router_max_fractions:
                        local_gate = _residual_router_gate(
                            consensus_validation_prediction[fit_mask],
                            inner_q0[fit_mask],
                            anchor_margin[fit_mask],
                            validation_disagreement[fit_mask],
                            threshold,
                            margin_max,
                            max_fraction,
                        )
                        if not local_gate.any():
                            continue
                        gate = np.zeros(len(manifest), dtype=bool)
                        gate[fit_indices[local_gate]] = True
                        prediction = consensus_validation_prediction.copy()
                        prediction[gate] = 2 - prediction[gate]
                        metrics = _score(labels[fit_mask], prediction[fit_mask])
                        fold_scores = {
                            fold: _score(
                                labels[folds == fold], prediction[folds == fold]
                            )["macro_f1"]
                            for fold in fit_folds
                        }
                        non_degrading = sum(
                            fold_scores[fold] + 1e-12
                            >= baseline_fit_folds[fold]
                            for fold in fit_folds
                        )
                        fixes_now = int(
                            np.sum(
                                gate
                                & (consensus_validation_prediction != labels)
                                & (prediction == labels)
                            )
                        )
                        harms_now = int(
                            np.sum(
                                gate
                                & (consensus_validation_prediction == labels)
                                & (prediction != labels)
                            )
                        )
                        changed_now = int(gate.sum())
                        valid = (
                            metrics["macro_f1"] > baseline_fit["macro_f1"]
                            and non_degrading >= 3
                            and fixes_now > harms_now
                        )
                        rank = (
                            int(valid),
                            metrics["macro_f1"],
                            non_degrading,
                            fixes_now / changed_now,
                            -changed_now,
                        )
                        if best is None or rank > best["rank"]:
                            best = {
                                "rank": rank,
                                "accepted": bool(valid),
                                "parameters": {
                                    "C": float(regularization),
                                    "threshold": float(threshold),
                                    "margin_max": float(margin_max),
                                    "max_fraction": float(max_fraction),
                                },
                                "fit_macro_f1": metrics["macro_f1"],
                                "fit_non_degrading_folds": int(non_degrading),
                                "fit_changed": changed_now,
                                "fit_fixes": fixes_now,
                                "fit_harms": harms_now,
                            }

        if best is not None and best["accepted"]:
            parameters = best["parameters"]
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=parameters["C"],
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=SEED,
                ),
            )
            classifier.fit(
                residual_validation_features[fit_mask & binary],
                (labels[fit_mask & binary] == 0).astype(int),
            )
            holdout_q0 = classifier.predict_proba(
                residual_validation_features[holdout_mask]
            )[:, 1]
            router_oof_q0[holdout_mask] = holdout_q0
            local_gate = _residual_router_gate(
                consensus_validation_prediction[holdout_mask],
                holdout_q0,
                anchor_margin[holdout_mask],
                validation_disagreement[holdout_mask],
                parameters["threshold"],
                parameters["margin_max"],
                parameters["max_fraction"],
            )
            holdout_gate = np.zeros(len(manifest), dtype=bool)
            holdout_gate[holdout_indices[local_gate]] = True
            router_prediction[holdout_gate] = 2 - router_prediction[holdout_gate]
            best["holdout_changed"] = int(holdout_gate.sum())
            best["holdout_fixes"] = int(
                np.sum(
                    holdout_gate
                    & (consensus_validation_prediction != labels)
                    & (router_prediction == labels)
                )
            )
            best["holdout_harms"] = int(
                np.sum(
                    holdout_gate
                    & (consensus_validation_prediction == labels)
                    & (router_prediction != labels)
                )
            )
        else:
            best = {
                "accepted": False,
                "reason": "no outer-train router candidate passed guard",
                "holdout_changed": 0,
                "holdout_fixes": 0,
                "holdout_harms": 0,
            }
        best.pop("rank", None)
        best["outer_fold"] = outer_fold
        router_outer_selections.append(best)

    router_candidate_metrics = _score(labels, router_prediction)
    router_candidate_fold_scores = [
        _score(labels[folds == fold], router_prediction[folds == fold])["macro_f1"]
        for fold in range(5)
    ]
    router_non_degrading = sum(
        candidate + 1e-12 >= baseline
        for candidate, baseline in zip(
            router_candidate_fold_scores, consensus_fold_scores
        )
    )
    router_changed = router_prediction != consensus_validation_prediction
    router_fixes = int(
        np.sum(
            router_changed
            & (consensus_validation_prediction != labels)
            & (router_prediction == labels)
        )
    )
    router_harms = int(
        np.sum(
            router_changed
            & (consensus_validation_prediction == labels)
            & (router_prediction != labels)
        )
    )
    router_passes_guard = bool(
        router_candidate_metrics["macro_f1"] > consensus_metrics["macro_f1"]
        and router_non_degrading >= 4
        and router_candidate_metrics["class_f1"][1] + 1e-12
        >= consensus_metrics["class_f1"][1]
        and router_fixes > router_harms
    )

    router_parameters = [
        row["parameters"]
        for row in router_outer_selections
        if row.get("accepted")
    ]
    final_router_parameters = None
    router_test_q0 = np.full(len(template), 0.5, dtype=np.float64)
    final_validation_prediction = consensus_validation_prediction.copy()
    final_test_prediction = consensus_test_prediction.copy()
    if router_passes_guard and router_parameters:
        c_values = np.asarray(router_regularizations)
        median_c = float(np.median([row["C"] for row in router_parameters]))
        selected_c = float(c_values[np.argmin(np.abs(c_values - median_c))])
        final_router_parameters = {
            "C": selected_c,
            "threshold": float(
                np.median([row["threshold"] for row in router_parameters])
            ),
            "margin_max": float(
                np.median([row["margin_max"] for row in router_parameters])
            ),
            "max_fraction": float(
                np.median([row["max_fraction"] for row in router_parameters])
            ),
        }
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=selected_c,
                class_weight="balanced",
                max_iter=3000,
                random_state=SEED,
            ),
        )
        classifier.fit(
            residual_validation_features[binary],
            (labels[binary] == 0).astype(int),
        )
        router_test_q0 = classifier.predict_proba(residual_test_features)[:, 1]
        test_gate = _residual_router_gate(
            consensus_test_prediction,
            router_test_q0,
            test_margin,
            test_disagreement,
            final_router_parameters["threshold"],
            final_router_parameters["margin_max"],
            final_router_parameters["max_fraction"],
        )
        final_validation_prediction = router_prediction
        final_test_prediction[test_gate] = 2 - final_test_prediction[test_gate]

    final_metrics = _score(labels, final_validation_prediction)
    final_fold_scores = [
        _score(labels[folds == fold], final_validation_prediction[folds == fold])[
            "macro_f1"
        ]
        for fold in range(5)
    ]

    stage_predictions = {
        "control": baseline_test_prediction,
        "patch_dino_consensus": consensus_test_prediction,
        "tri_encoder_router": final_test_prediction,
    }
    for name, predictions in stage_predictions.items():
        stage_submission = template[["id"]].copy()
        stage_submission["predicted"] = predictions.astype(int)
        stage_submission.to_csv(stage_submission_paths[name], index=False)
    final_submission = template[["id"]].copy()
    final_submission["predicted"] = final_test_prediction.astype(int)
    final_submission.to_csv(submission_path, index=False)

    pd.DataFrame(
        {
            "row_index": np.arange(len(manifest)),
            "path": manifest.path,
            "label": labels,
            "fold": folds,
            "control_prediction": baseline_prediction,
            "consensus_prediction": consensus_validation_prediction,
            "router_prediction": final_validation_prediction,
            "anchor_margin": anchor_margin,
            "anchor_entropy": anchor_entropy,
            "patch_q0": patch_validation_q0,
            "dino_q0": dino_validation_q0,
            "pe_core_prediction": pe_validation_probability.argmax(1),
            "router_q0": router_oof_q0,
        }
    ).to_csv(validation_path, index=False)
    test_frame = template[["id"]].copy()
    test_frame["control_prediction"] = baseline_test_prediction
    test_frame["consensus_prediction"] = consensus_test_prediction
    test_frame["router_prediction"] = final_test_prediction
    test_frame["router_q0"] = router_test_q0
    test_frame.to_csv(test_path, index=False)
    np.savez_compressed(
        router_path,
        labels=labels.astype(np.int8),
        folds=folds.astype(np.int8),
        anchor_validation_probabilities=anchor_validation_probability.astype(np.float32),
        anchor_test_probabilities=anchor_test_probability.astype(np.float32),
        patch_validation_q0=patch_validation_q0.astype(np.float32),
        patch_test_q0=patch_test_q0.astype(np.float32),
        dino_regularizations=regularizations,
        dino_validation_q0=dino_validation_by_c.astype(np.float32),
        dino_test_q0=dino_test_by_c.astype(np.float32),
        pe_core_validation_probabilities=pe_validation_probability.astype(np.float32),
        pe_core_test_probabilities=pe_test_probability.astype(np.float32),
        router_validation_q0=router_oof_q0.astype(np.float32),
        router_test_q0=router_test_q0.astype(np.float32),
        control_validation_predictions=baseline_prediction.astype(np.int8),
        consensus_validation_predictions=consensus_validation_prediction.astype(np.int8),
        final_validation_predictions=final_validation_prediction.astype(np.int8),
        control_test_predictions=baseline_test_prediction.astype(np.int8),
        consensus_test_predictions=consensus_test_prediction.astype(np.int8),
        final_test_predictions=final_test_prediction.astype(np.int8),
    )

    report = {
        "architecture": "Full 5-fold Hierarchical + Patch-MIL + DINO tri-consensus + confidence-aware PE-Core residual router",
        "test_labels_used": False,
        "selection_source": "official-train nested OOF only",
        "seed": SEED,
        "rows_train": int(len(manifest)),
        "rows_test": int(len(template)),
        "cache_sources": {
            "hierarchical_anchor": str(anchor),
            "frozen_dino": str(dino_source),
            "full_patch_mil": str(patch_root),
            "frozen_pe_core": str(output),
        },
        "control_validation": baseline_metrics,
        "control_fold_macro_f1": baseline_fold_scores,
        "nested_outer_selections": outer_selections,
        "candidate_validation_before_guard": candidate_metrics,
        "candidate_fold_macro_f1_before_guard": candidate_fold_scores,
        "candidate_fixes_before_guard": candidate_fixes,
        "candidate_harms_before_guard": candidate_harms,
        "candidate_changed_rows_before_guard": candidate_changed_rows,
        "passes_guard": passes_guard,
        "selected_validation": nested_metrics,
        "selected_fold_macro_f1": nested_fold_scores,
        "non_degrading_folds": int(non_degrading),
        "fixes": fixes,
        "harms": harms,
        "final_median_parameters": final_parameters,
        "changed_validation_rows": int(
            np.sum(nested_prediction != baseline_prediction)
        ),
        "changed_test_rows": int(np.sum(test_prediction != baseline_test_prediction)),
        "improvement3": {
            "method": "frozen PE-Core five-fold linear probe + nested confidence-aware residual router",
            "outer_selections": router_outer_selections,
            "candidate_validation_before_guard": router_candidate_metrics,
            "candidate_fold_macro_f1_before_guard": router_candidate_fold_scores,
            "passes_guard": router_passes_guard,
            "selected_parameters": final_router_parameters,
            "selected_validation": final_metrics,
            "selected_fold_macro_f1": final_fold_scores,
            "non_degrading_folds": int(router_non_degrading),
            "fixes": router_fixes,
            "harms": router_harms,
            "changed_validation_rows": int(
                np.sum(final_validation_prediction != consensus_validation_prediction)
            ),
            "changed_test_rows": int(
                np.sum(final_test_prediction != consensus_test_prediction)
            ),
        },
        "stage_submissions": {
            name: str(path) for name, path in stage_submission_paths.items()
        },
        "final_stage": (
            "tri_encoder_router"
            if router_passes_guard
            else "patch_dino_consensus"
            if passes_guard
            else "control"
        ),
        "final_validation": final_metrics,
        "final_fold_macro_f1": final_fold_scores,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "cached": False,
        "run_root": str(output),
        "metrics": metrics_path.read_text(encoding="utf-8"),
        "submission": submission_path.read_text(encoding="utf-8"),
        "stage_submissions": {
            name: path.read_text(encoding="utf-8")
            for name, path in stage_submission_paths.items()
        },
    }


def extend_hierarchical_pilot(
    extra_epochs: int = 2,
    force: bool = False,
    unfrozen_blocks: int = 4,
    head_lr_scale: float = 0.5,
    backbone_lr_scale: float = 0.5,
    run_name: str | None = None,
):
    """Continue the saved fold-0 checkpoint with a fresh optimizer."""
    if extra_epochs < 1 or extra_epochs > 4:
        raise ValueError("extra_epochs must be between 1 and 4")
    if unfrozen_blocks < 1 or unfrozen_blocks > 12:
        raise ValueError("unfrozen_blocks must be between 1 and 12")
    if head_lr_scale <= 0 or backbone_lr_scale <= 0:
        raise ValueError("learning-rate scales must be positive")
    modules = _imports()
    globals().update({key: value for key, value in modules.items() if not key.startswith("_")})
    _seed_everything(SEED + 100)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    base = Path(CACHE_ROOT) / f"hierarchical_siglip2_pilot1fold_seed{SEED}"
    base_checkpoint = base / "siglip2_fold0_multi.pt"
    base_metrics = base / "metrics.json"
    if not base_checkpoint.exists() or not base_metrics.exists():
        raise FileNotFoundError(
            "Pilot checkpoint missing. Run pipeline.py without --extend-epochs first."
        )
    output = base / (
        run_name
        or (
            f"extension_{extra_epochs}epoch_half_lr"
            if unfrozen_blocks == 4
            and head_lr_scale == 0.5
            and backbone_lr_scale == 0.5
            else f"extension_{extra_epochs}epoch_{unfrozen_blocks}blocks"
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    submission_path = output / "submission.csv"
    extended_checkpoint = output / "siglip2_fold0_multi_extended.pt"
    if metrics_path.exists() and submission_path.exists() and extended_checkpoint.exists() and not force:
        return {
            "cached": True,
            "run_root": str(output),
            "metrics": metrics_path.read_text(encoding="utf-8"),
            "submission": submission_path.read_text(encoding="utf-8"),
        }

    started = time.perf_counter()
    manifest, template, test_paths = _load_data()
    folds = _make_folds(manifest)
    train_frame = manifest[folds != 0].reset_index(drop=True)
    validation_frame = manifest[folds == 0].reset_index(drop=True)
    config = MODEL_CONFIGS["siglip2"]
    model = _build_model("siglip2", multitask=True)
    model.load_state_dict(torch.load(base_checkpoint, map_location="cuda", weights_only=True))
    _unfreeze_tail(model, "siglip2", count=unfrozen_blocks)

    class_counts = np.bincount(train_frame.label.to_numpy(), minlength=NUM_CLASSES)
    class_weights = np.sqrt(class_counts.sum() / np.maximum(class_counts, 1))
    class_weights = torch.tensor(
        class_weights / class_weights.mean(), dtype=torch.float32, device="cuda"
    )
    train_loader = _loader(
        train_frame.path,
        train_frame.label,
        config["size"],
        True,
        config["batch"],
        kind="siglip2",
    )
    validation_loader = _loader(
        validation_frame.path,
        validation_frame.label,
        config["size"],
        False,
        config["batch"],
        kind="siglip2",
    )
    test_loader = _loader(
        test_paths, None, config["size"], False, config["batch"], kind="siglip2"
    )
    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": config["head_lr"] * head_lr_scale},
            {
                "params": backbone_parameters,
                "lr": config["backbone_lr"] * backbone_lr_scale,
            },
        ],
        weight_decay=0.05,
    )

    original_validation = _predict(model, validation_loader, multitask=True)
    original_logits, original_binary = original_validation
    original_metrics = _score(validation_frame.label.to_numpy(), original_logits.argmax(1))
    best_score = original_metrics["macro_f1"]
    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    extension_history = []

    for offset in range(extra_epochs):
        model.train()
        running, seen = 0.0, 0
        for images, labels, _, weights in train_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            weights = weights.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, binary_logits, projection = model(images)
                row_loss = F.cross_entropy(
                    logits,
                    labels,
                    weight=class_weights,
                    label_smoothing=0.05,
                    reduction="none",
                )
                loss = (row_loss * weights).sum() / weights.sum().clamp_min(1.0)
                binary_mask = labels != 1
                binary_target = (labels[binary_mask] == 0).float()
                binary_loss = F.binary_cross_entropy_with_logits(
                    binary_logits[binary_mask], binary_target
                )
                contrastive = _supcon(projection, labels)
                flip_logits = model(torch.flip(images, dims=[3]))[0]
                p = F.log_softmax(logits, dim=1)
                q = F.log_softmax(flip_logits, dim=1)
                consistency = 0.5 * (
                    F.kl_div(p, q.exp(), reduction="batchmean")
                    + F.kl_div(q, p.exp(), reduction="batchmean")
                )
                loss = loss + 0.25 * binary_loss + 0.05 * contrastive + 0.02 * consistency
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(labels)
            seen += len(labels)

        validation_logits, _ = _predict(model, validation_loader, multitask=True)
        metrics = _score(validation_frame.label.to_numpy(), validation_logits.argmax(1))
        row = {
            "phase": "partial_extension",
            "epoch": 5 + offset,
            "train_rows": seen,
            "train_loss": running / max(seen, 1),
            **metrics,
        }
        extension_history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }

    torch.save(best_state, extended_checkpoint)
    model.load_state_dict(best_state)
    validation_logits, validation_binary = _predict(
        model, validation_loader, multitask=True
    )
    test_logits, test_binary = _predict(model, test_loader, multitask=True)
    validation_base = _softmax(validation_logits)
    validation_q = 1.0 / (1.0 + np.exp(-validation_binary))
    validation_hierarchical = np.column_stack(
        [
            (1 - validation_base[:, 1]) * validation_q,
            validation_base[:, 1],
            (1 - validation_base[:, 1]) * (1 - validation_q),
        ]
    )
    labels = validation_frame.label.to_numpy()
    selected = None
    for alpha in (0.50, 0.65, 0.80, 1.00):
        probabilities = alpha * validation_base + (1 - alpha) * validation_hierarchical
        metrics = _score(labels, probabilities.argmax(1))
        candidate = (metrics["macro_f1"], alpha, metrics)
        if selected is None or candidate[0] > selected[0]:
            selected = candidate

    test_base = _softmax(test_logits)
    test_q = 1.0 / (1.0 + np.exp(-test_binary))
    test_hierarchical = np.column_stack(
        [
            (1 - test_base[:, 1]) * test_q,
            test_base[:, 1],
            (1 - test_base[:, 1]) * (1 - test_q),
        ]
    )
    test_probabilities = selected[1] * test_base + (1 - selected[1]) * test_hierarchical
    submission = template[["id"]].copy()
    submission["predicted"] = test_probabilities.argmax(1).astype(int)
    submission.to_csv(submission_path, index=False)
    report = {
        "architecture": "hierarchical_siglip2",
        "mode": "pilot_extension",
        "test_labels_used": False,
        "seed": SEED,
        "source_checkpoint": str(base_checkpoint),
        "optimizer_reset": True,
        "unfrozen_blocks": unfrozen_blocks,
        "extra_epochs_requested": extra_epochs,
        "head_lr": config["head_lr"] * head_lr_scale,
        "backbone_lr": config["backbone_lr"] * backbone_lr_scale,
        "original_checkpoint_validation": original_metrics,
        "extension_history": extension_history,
        "best_raw_validation_macro_f1": best_score,
        "selected_alpha_3class": selected[1],
        "validation": selected[2],
        "checkpoint": str(extended_checkpoint),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "cached": False,
        "run_root": str(output),
        "metrics": metrics_path.read_text(encoding="utf-8"),
        "submission": submission_path.read_text(encoding="utf-8"),
    }


def _seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_data(manifest_override=None):
    root = Path(DATA_ROOT)
    manifest_path = (
        Path(manifest_override)
        if manifest_override is not None
        else root / "train_manifest.csv"
    )
    if not manifest_path.exists():
        rows = []
        folders = {
            0: root / "train" / "0_Recyclable",
            1: root / "train" / "1_Electronic",
            2: root / "train" / "2_Organic",
        }
        for label, folder in folders.items():
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing official train folder: {folder}")
            for path in sorted(item for item in folder.iterdir() if item.is_file()):
                rows.append(
                    {
                        "path": str(path),
                        "label": label,
                        "group": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        pd.DataFrame(rows).to_csv(manifest_path, index=False)

    manifest = pd.read_csv(manifest_path)
    required = {"path", "label", "group"}
    if set(manifest.columns) != required or set(manifest.label.astype(int)) != {0, 1, 2}:
        raise ValueError("Invalid train_manifest.csv")
    manifest["label"] = manifest["label"].astype(int)

    template_path = root / "submission.csv"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing submission template: {template_path}")
    template = pd.read_csv(template_path)
    if list(template.columns)[:2] != ["id", "predicted"]:
        raise ValueError("submission.csv must start with id,predicted")

    test_dir = root / "test"
    test_paths = []
    for identifier in template["id"]:
        candidates = [test_dir / str(identifier), test_dir / f"{identifier}.jpg", test_dir / f"{identifier}.png"]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            numeric = int(identifier)
            matches = sorted(test_dir.glob(f"{numeric}.*"))
            path = matches[0] if matches else None
        if path is None:
            raise FileNotFoundError(f"Missing test image for id={identifier}")
        test_paths.append(str(path))
    return manifest, template, test_paths


def _make_folds(manifest):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.full(len(manifest), -1, dtype=np.int64)
    for fold, (_, validation) in enumerate(
        splitter.split(manifest.path, manifest.label, groups=manifest.group)
    ):
        folds[validation] = fold
    if np.any(folds < 0):
        raise RuntimeError("Incomplete fold assignment")
    return folds


class _PadSquare:
    def __call__(self, image):
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        return ImageOps.expand(image, (left, top, side - width - left, side - height - top), fill=(0, 0, 0))


def _transform(size, training, kind):
    if kind == "dino":
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.70, 1.0), ratio=(0.80, 1.25)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.1),
            ]
        )
    return transforms.Compose([_PadSquare(), transforms.Resize((size, size)), transforms.ToTensor(), normalize])


class _Images:
    def __init__(self, paths, labels=None, size=224, training=False, soft_targets=None, weights=None, kind="siglip2"):
        self.paths = list(paths)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.transform = _transform(size, training, kind)
        self.soft_targets = soft_targets
        self.weights = weights

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = self.transform(source.convert("RGB"))
        label = -1 if self.labels is None else int(self.labels[index])
        target = (
            torch.zeros(NUM_CLASSES, dtype=torch.float32)
            if self.soft_targets is None
            else torch.as_tensor(self.soft_targets[index], dtype=torch.float32)
        )
        weight = 1.0 if self.weights is None else float(self.weights[index])
        return image, label, target, weight


def _unfreeze_tail(model, kind, count=4):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    modules = []
    if kind == "siglip2":
        modules = list(model.backbone.vision_model.encoder.layers[-count:])
        modules.append(model.backbone.vision_model.post_layernorm)
    elif kind == "dino":
        stages = list(model.backbone.stages)
        modules = stages[-min(2, len(stages)):]
        modules.append(model.backbone.layer_norm)
    elif kind == "pe_core":
        modules = list(model.backbone.blocks[-count:])
        modules.append(model.backbone.norm)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def _build_model(kind, multitask=False):
    config = MODEL_CONFIGS[kind]

    def retry_load(loader, attempts=4):
        for attempt in range(1, attempts + 1):
            try:
                return loader()
            except Exception:
                if attempt == attempts:
                    raise
                wait_seconds = min(5 * attempt, 20)
                print(
                    f"Model download/load failed ({attempt}/{attempts}); retry in {wait_seconds}s",
                    flush=True,
                )
                time.sleep(wait_seconds)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "siglip2":
                from transformers import SiglipVisionModel

                self.backbone = retry_load(
                    lambda: SiglipVisionModel.from_pretrained(
                        config["repo"], revision=config["revision"],
                    )
                )
                dimension = self.backbone.config.hidden_size
            elif kind == "dino":
                from transformers import AutoModel

                self.backbone = retry_load(
                    lambda: AutoModel.from_pretrained(
                        config["repo"], revision=config["revision"],
                    )
                )
                dimension = self.backbone.config.hidden_sizes[-1]
            else:
                import timm

                self.backbone = retry_load(
                    lambda: timm.create_model(config["repo"], pretrained=True, num_classes=0)
                )
                dimension = self.backbone.num_features
            self.classifier = nn.Linear(dimension, NUM_CLASSES)
            self.binary = nn.Linear(dimension, 1) if multitask else None
            self.projection = nn.Sequential(nn.Linear(dimension, 256), nn.GELU(), nn.Linear(256, 128)) if multitask else None

        def encode(self, images):
            if kind == "siglip2":
                return self.backbone(pixel_values=images).pooler_output
            if kind == "dino":
                return self.backbone(pixel_values=images).pooler_output
            return self.backbone(images)

        def forward(self, images):
            features = self.encode(images)
            logits = self.classifier(features)
            if not multitask:
                return logits
            return logits, self.binary(features).squeeze(1), F.normalize(self.projection(features), dim=1)

    model = Classifier().cuda()
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    return model


def _build_patch_mil_model():
    config = MODEL_CONFIGS["siglip2"]
    from transformers import SiglipVisionModel

    class PatchMILClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            last_error = None
            for attempt in range(1, 5):
                try:
                    self.backbone = SiglipVisionModel.from_pretrained(
                        config["repo"], revision=config["revision"]
                    )
                    break
                except Exception as error:
                    last_error = error
                    if attempt == 4:
                        raise
                    time.sleep(5 * attempt)
            if last_error is not None:
                print("SigLIP2 load recovered after retry", flush=True)
            dimension = self.backbone.config.hidden_size
            self.classifier = nn.Linear(dimension, NUM_CLASSES)
            self.binary = nn.Linear(dimension, 1)
            self.patch_attention = nn.Linear(dimension, 1)
            self.patch_binary = nn.Linear(dimension, 1)
            self.projection = nn.Sequential(
                nn.Linear(dimension, 256), nn.GELU(), nn.Linear(256, 128)
            )

        def forward(self, images):
            output = self.backbone(pixel_values=images)
            pooled = output.pooler_output
            tokens = output.last_hidden_state
            attention = torch.softmax(self.patch_attention(tokens).squeeze(-1), dim=1)
            patch_pooled = torch.sum(tokens * attention.unsqueeze(-1), dim=1)
            return (
                self.classifier(pooled),
                self.binary(pooled).squeeze(1),
                self.patch_binary(patch_pooled).squeeze(1),
                F.normalize(self.projection(pooled), dim=1),
            )

    model = PatchMILClassifier().cuda()
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    return model


def _predict_patch_mil(model, loader):
    model.eval()
    class_logits, binary_logits, patch_logits = [], [], []
    with torch.no_grad():
        for images, _, _, _ in loader:
            images = images.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, binary, patch, _ = model(images)
            class_logits.append(logits.float().cpu().numpy())
            binary_logits.append(binary.float().cpu().numpy())
            patch_logits.append(patch.float().cpu().numpy())
    return (
        np.concatenate(class_logits),
        np.concatenate(binary_logits),
        np.concatenate(patch_logits),
    )


def _ema_update(ema_state, model, decay):
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                ema_state[name].mul_(decay).add_(value.detach(), alpha=1 - decay)
            else:
                ema_state[name].copy_(value)


def _train_patch_mil_fold(
    train_frame,
    validation_frame,
    test_paths,
    fold,
    output,
    reuse_checkpoint=False,
    global_binary_weight=0.20,
    patch_binary_weight=0.20,
):
    config = MODEL_CONFIGS["siglip2"]
    checkpoint = output / f"siglip2_patch_mil_fold{fold}.pt"
    model = _build_patch_mil_model()
    ema_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    ema_decay = 0.995

    class_counts = np.bincount(train_frame.label.to_numpy(), minlength=NUM_CLASSES)
    class_weights = np.sqrt(class_counts.sum() / np.maximum(class_counts, 1))
    class_weights = torch.tensor(
        class_weights / class_weights.mean(), dtype=torch.float32, device="cuda"
    )
    validation_loader = _loader(
        validation_frame.path,
        validation_frame.label,
        config["size"],
        False,
        config["batch"],
        kind="siglip2",
    )
    test_loader = _loader(
        test_paths, None, config["size"], False, config["batch"], kind="siglip2"
    )

    if reuse_checkpoint and checkpoint.exists():
        model.load_state_dict(
            torch.load(checkpoint, map_location="cuda", weights_only=True)
        )
        validation_output = _predict_patch_mil(model, validation_loader)
        test_output = _predict_patch_mil(model, test_loader)
        del model, ema_state
        torch.cuda.empty_cache()
        return (
            validation_output,
            test_output,
            [{"phase": "cached", "checkpoint": str(checkpoint)}],
            str(checkpoint),
            "cached",
        )

    history = []
    best_score, best_state, best_variant = -1.0, None, None
    for phase, phase_epochs in (("head", 1), ("partial", config["epochs"])):
        if phase == "partial":
            _unfreeze_tail(model, "siglip2")
        backbone_parameters = [
            parameter for parameter in model.backbone.parameters() if parameter.requires_grad
        ]
        head_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("backbone")
        ]
        groups = [{"params": head_parameters, "lr": config["head_lr"]}]
        if backbone_parameters:
            groups.append({"params": backbone_parameters, "lr": config["backbone_lr"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.05)

        for epoch in range(phase_epochs):
            train_loader = _loader(
                train_frame.path,
                train_frame.label,
                config["size"],
                True,
                config["batch"],
                kind="siglip2",
            )
            model.train()
            running, seen = 0.0, 0
            for images, labels, _, weights in train_loader:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                weights = weights.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, binary_logits, patch_logits, projection = model(images)
                    row_loss = F.cross_entropy(
                        logits,
                        labels,
                        weight=class_weights,
                        label_smoothing=0.05,
                        reduction="none",
                    )
                    loss = (row_loss * weights).sum() / weights.sum().clamp_min(1.0)
                    binary_mask = labels != 1
                    binary_target = (labels[binary_mask] == 0).float()
                    global_binary_loss = F.binary_cross_entropy_with_logits(
                        binary_logits[binary_mask], binary_target
                    )
                    patch_binary_loss = F.binary_cross_entropy_with_logits(
                        patch_logits[binary_mask], binary_target
                    )
                    contrastive = _supcon(projection, labels)
                    flip_logits = model(torch.flip(images, dims=[3]))[0]
                    p = F.log_softmax(logits, dim=1)
                    q = F.log_softmax(flip_logits, dim=1)
                    consistency = 0.5 * (
                        F.kl_div(p, q.exp(), reduction="batchmean")
                        + F.kl_div(q, p.exp(), reduction="batchmean")
                    )
                    loss = (
                        loss
                        + global_binary_weight * global_binary_loss
                        + patch_binary_weight * patch_binary_loss
                        + 0.05 * contrastive
                        + 0.02 * consistency
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                _ema_update(ema_state, model, ema_decay)
                running += float(loss.detach()) * len(labels)
                seen += len(labels)

            raw_output = _predict_patch_mil(model, validation_loader)
            raw_metrics = _score(
                validation_frame.label.to_numpy(), raw_output[0].argmax(1)
            )
            raw_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            model.load_state_dict(ema_state)
            ema_output = _predict_patch_mil(model, validation_loader)
            ema_metrics = _score(
                validation_frame.label.to_numpy(), ema_output[0].argmax(1)
            )
            model.load_state_dict(raw_state)
            del raw_state

            variant, metrics = max(
                (("raw", raw_metrics), ("ema", ema_metrics)),
                key=lambda item: item[1]["macro_f1"],
            )
            row = {
                "phase": phase,
                "epoch": epoch + 1,
                "train_rows": seen,
                "train_loss": running / max(seen, 1),
                "raw_macro_f1": raw_metrics["macro_f1"],
                "ema_macro_f1": ema_metrics["macro_f1"],
                "selected_variant": variant,
                **metrics,
            }
            history.append(row)
            print(json.dumps({"model": "siglip2_patch_mil", "fold": fold, **row}), flush=True)
            if metrics["macro_f1"] > best_score:
                best_score = metrics["macro_f1"]
                source = model.state_dict() if variant == "raw" else ema_state
                best_state = {
                    name: value.detach().cpu().clone() for name, value in source.items()
                }
                best_variant = variant

    if best_state is None:
        raise RuntimeError("Patch-MIL training produced no checkpoint")
    torch.save(best_state, checkpoint)
    model.load_state_dict(best_state)
    validation_output = _predict_patch_mil(model, validation_loader)
    test_output = _predict_patch_mil(model, test_loader)
    del model, ema_state
    torch.cuda.empty_cache()
    return validation_output, test_output, history, str(checkpoint), best_variant


def _loader(paths, labels, size, training, batch, soft_targets=None, weights=None, shuffle=None, kind="siglip2"):
    dataset = _Images(paths, labels, size, training, soft_targets, weights, kind)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=training if shuffle is None else shuffle,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )


def _predict(model, loader, multitask=False):
    model.eval()
    logits, binaries = [], []
    with torch.no_grad():
        for images, _, _, _ in loader:
            images = images.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(images)
            if multitask:
                class_logits, binary_logits, _ = output
                binaries.append(binary_logits.float().cpu().numpy())
            else:
                class_logits = output
            logits.append(class_logits.float().cpu().numpy())
    class_array = np.concatenate(logits)
    return (class_array, np.concatenate(binaries)) if multitask else class_array


def _score(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "class_f1": f1_score(labels, predictions, average=None, labels=[0, 1, 2]).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "errors": int(np.sum(np.asarray(labels) != np.asarray(predictions))),
    }


def _tri_consensus_gate(
    predictions,
    margin_values,
    entropy_values,
    patch_q0,
    dino_q0,
    parameters,
):
    uncertain = (
        (margin_values <= parameters["margin_max"])
        & (entropy_values >= parameters["entropy_min"])
    )
    to_zero = (
        (predictions == 2)
        & (patch_q0 >= parameters["patch_confidence"])
        & (dino_q0 >= parameters["dino_confidence"])
    )
    to_two = (
        (predictions == 0)
        & (patch_q0 <= 1.0 - parameters["patch_confidence"])
        & (dino_q0 <= 1.0 - parameters["dino_confidence"])
    )
    return (predictions != 1) & uncertain & (to_zero | to_two)


def _residual_router_gate(
    predictions,
    probability_class0,
    anchor_margin,
    disagreement,
    threshold,
    margin_max,
    max_fraction,
):
    direction = (
        ((predictions == 2) & (probability_class0 >= threshold))
        | ((predictions == 0) & (probability_class0 <= 1.0 - threshold))
    )
    gate = (
        (predictions != 1)
        & direction
        & ((anchor_margin <= margin_max) | disagreement)
    )
    indices = np.flatnonzero(gate)
    limit = max(1, int(np.ceil(len(predictions) * max_fraction)))
    if len(indices) > limit:
        confidence = np.maximum(
            probability_class0[indices], 1.0 - probability_class0[indices]
        )
        keep = indices[np.argsort(-confidence, kind="mergesort")[:limit]]
        gate[:] = False
        gate[keep] = True
    return gate


def _supcon(embeddings, labels, temperature=0.1):
    similarity = embeddings @ embeddings.T / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(len(labels), device=labels.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~identity
    denominator = torch.logsumexp(similarity.masked_fill(identity, -1e9), dim=1)
    log_probability = similarity - denominator[:, None]
    valid = positives.sum(1) > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    return -((log_probability * positives).sum(1) / positives.sum(1).clamp_min(1))[valid].mean()


def _train_fold(
    kind,
    train_frame,
    validation_frame,
    test_paths,
    fold,
    output,
    multitask=False,
    soft_targets=None,
    sample_weights=None,
    lambda_binary=0.25,
    lambda_contrastive=0.05,
    lambda_consistency=0.02,
    curriculum_groups=None,
    reuse_checkpoint=False,
):
    config = MODEL_CONFIGS[kind]
    checkpoint = output / f"{kind}_fold{fold}_{'multi' if multitask else 'standard'}.pt"
    model = _build_model(kind, multitask=multitask)
    class_counts = np.bincount(train_frame.label.to_numpy(), minlength=NUM_CLASSES)
    class_weights = np.sqrt(class_counts.sum() / np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float32, device="cuda")

    validation_loader = _loader(validation_frame.path, validation_frame.label, config["size"], False, config["batch"], kind=kind)
    test_loader = _loader(test_paths, None, config["size"], False, config["batch"], kind=kind)

    if reuse_checkpoint and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
        validation_output = _predict(model, validation_loader, multitask)
        test_output = _predict(model, test_loader, multitask)
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"model": kind, "fold": fold, "phase": "reused_checkpoint"}), flush=True)
        return validation_output, test_output, [{"phase": "reused_checkpoint"}], str(checkpoint)

    history = []
    best = (-1.0, None)
    phases = [("head", 1), ("partial", config["epochs"])]
    for phase, phase_epochs in phases:
        if phase == "partial":
            _unfreeze_tail(model, kind)
        backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
        head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("backbone")]
        groups = [{"params": head_parameters, "lr": config["head_lr"]}]
        if backbone_parameters:
            groups.append({"params": backbone_parameters, "lr": config["backbone_lr"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.05)

        for epoch in range(phase_epochs):
            if curriculum_groups is None:
                active = np.ones(len(train_frame), dtype=bool)
            elif phase == "head":
                active = np.asarray(curriculum_groups) == 0
            elif epoch == 0:
                active = np.asarray(curriculum_groups) <= 1
            else:
                active = np.ones(len(train_frame), dtype=bool)
            train_loader = _loader(
                train_frame.loc[active, "path"],
                train_frame.loc[active, "label"],
                config["size"],
                True,
                config["batch"],
                None if soft_targets is None else np.asarray(soft_targets)[active],
                None if sample_weights is None else np.asarray(sample_weights)[active],
                kind=kind,
            )
            model.train()
            running = 0.0
            seen = 0
            for images, labels, targets, weights in train_loader:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
                weights = weights.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(images)
                    if multitask:
                        logits, binary_logits, projection = output
                    else:
                        logits = output
                    if soft_targets is None:
                        row_loss = F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=0.05, reduction="none")
                    else:
                        row_loss = -(targets * F.log_softmax(logits, dim=1)).sum(1)
                    loss = (row_loss * weights).sum() / weights.sum().clamp_min(1.0)
                    if multitask:
                        binary_mask = labels != 1
                        binary_target = (labels[binary_mask] == 0).float()
                        binary_loss = F.binary_cross_entropy_with_logits(binary_logits[binary_mask], binary_target) if binary_mask.any() else loss * 0.0
                        contrastive = _supcon(projection, labels)
                        flipped = torch.flip(images, dims=[3])
                        flip_logits = model(flipped)[0]
                        p = F.log_softmax(logits, dim=1)
                        q = F.log_softmax(flip_logits, dim=1)
                        consistency = 0.5 * (
                            F.kl_div(p, q.exp(), reduction="batchmean")
                            + F.kl_div(q, p.exp(), reduction="batchmean")
                        )
                        loss = loss + lambda_binary * binary_loss + lambda_contrastive * contrastive + lambda_consistency * consistency
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running += float(loss.detach()) * len(labels)
                seen += len(labels)

            validation_output = _predict(model, validation_loader, multitask)
            validation_logits = validation_output[0] if multitask else validation_output
            metrics = _score(validation_frame.label.to_numpy(), validation_logits.argmax(1))
            row = {"phase": phase, "epoch": epoch + 1, "train_rows": seen, "train_loss": running / max(seen, 1), **metrics}
            history.append(row)
            print(json.dumps({"model": kind, "fold": fold, **row}), flush=True)
            if metrics["macro_f1"] > best[0]:
                best = (metrics["macro_f1"], {key: value.detach().cpu() for key, value in model.state_dict().items()})

    if best[1] is None:
        raise RuntimeError("Training produced no checkpoint")
    torch.save(best[1], checkpoint)
    model.load_state_dict(best[1])
    validation_output = _predict(model, validation_loader, multitask)
    test_output = _predict(model, test_loader, multitask)
    del model
    torch.cuda.empty_cache()
    return validation_output, test_output, history, str(checkpoint)


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _fold_report(labels, folds, predictions, active_folds):
    return [
        {"fold": int(fold), **_score(labels[folds == fold], predictions[folds == fold])}
        for fold in active_folds
    ]


def _run_hierarchical(
    manifest, folds, test_paths, active_folds, output, reuse_checkpoints=False
):
    labels = manifest.label.to_numpy()
    oof_class = np.full((len(manifest), NUM_CLASSES), np.nan, dtype=np.float32)
    oof_binary = np.full(len(manifest), np.nan, dtype=np.float32)
    test_class, test_binary, histories, checkpoints = [], [], [], []
    for fold in active_folds:
        train = manifest[folds != fold].reset_index(drop=True)
        validation = manifest[folds == fold].reset_index(drop=True)
        validation_output, test_output, history, checkpoint = _train_fold(
            "siglip2",
            train,
            validation,
            test_paths,
            fold,
            output,
            multitask=True,
            reuse_checkpoint=reuse_checkpoints,
        )
        validation_logits, validation_binary = validation_output
        test_logits, test_binary_logits = test_output
        oof_class[folds == fold] = validation_logits
        oof_binary[folds == fold] = validation_binary
        test_class.append(test_logits)
        test_binary.append(test_binary_logits)
        histories.append({"fold": fold, "history": history})
        checkpoints.append(checkpoint)

    mask = np.isfinite(oof_binary)
    base_prob = _softmax(oof_class[mask])
    q = 1.0 / (1.0 + np.exp(-oof_binary[mask]))
    hierarchical = np.column_stack([(1 - base_prob[:, 1]) * q, base_prob[:, 1], (1 - base_prob[:, 1]) * (1 - q)])
    best = None
    for alpha in (0.50, 0.65, 0.80, 1.00):
        probabilities = alpha * base_prob + (1 - alpha) * hierarchical
        metrics = _score(labels[mask], probabilities.argmax(1))
        candidate = (metrics["macro_f1"], alpha, metrics)
        if best is None or candidate[0] > best[0]:
            best = candidate

    test_class_logits = np.mean(test_class, axis=0)
    test_binary_logits = np.mean(test_binary, axis=0)
    test_base = _softmax(test_class_logits)
    test_q = 1.0 / (1.0 + np.exp(-test_binary_logits))
    test_hierarchical = np.column_stack([(1 - test_base[:, 1]) * test_q, test_base[:, 1], (1 - test_base[:, 1]) * (1 - test_q)])
    test_probabilities = best[1] * test_base + (1 - best[1]) * test_hierarchical
    oof_predictions = np.full(len(manifest), -1, dtype=int)
    selected_probabilities = best[1] * base_prob + (1 - best[1]) * hierarchical
    oof_predictions[mask] = selected_probabilities.argmax(1)
    np.savez_compressed(
        output / "oof_probabilities.npz",
        logits=oof_class,
        binary_logits=oof_binary,
        labels=labels,
        folds=folds,
    )
    np.savez_compressed(
        output / "test_probabilities.npz",
        probabilities=test_probabilities.astype(np.float32),
    )
    return {
        "method": "shared SigLIP2 encoder + 3-class/binary/contrastive heads",
        "selected_alpha_3class": float(best[1]),
        "validation": best[2],
        "fold_metrics": _fold_report(labels, folds, oof_predictions, active_folds),
        "histories": histories,
        "checkpoints": checkpoints,
        "test_predictions": test_probabilities.argmax(1),
    }


def _hierarchical_probabilities(base_probability, binary_logits):
    q = 1.0 / (1.0 + np.exp(-binary_logits))
    return np.column_stack(
        [
            (1 - base_probability[:, 1]) * q,
            base_probability[:, 1],
            (1 - base_probability[:, 1]) * (1 - q),
        ]
    )


def _run_patch_mil(
    manifest,
    folds,
    test_paths,
    active_folds,
    output,
    reuse_checkpoints=False,
    global_binary_weight=0.20,
    patch_binary_weight=0.20,
):
    labels = manifest.label.to_numpy()
    oof_class = np.full((len(manifest), NUM_CLASSES), np.nan, dtype=np.float32)
    oof_global = np.full(len(manifest), np.nan, dtype=np.float32)
    oof_patch = np.full(len(manifest), np.nan, dtype=np.float32)
    test_class, test_global, test_patch = [], [], []
    histories, checkpoints, checkpoint_variants = [], [], []

    for fold in active_folds:
        train = manifest[folds != fold].reset_index(drop=True)
        validation = manifest[folds == fold].reset_index(drop=True)
        validation_output, test_output, history, checkpoint, variant = _train_patch_mil_fold(
            train,
            validation,
            test_paths,
            fold,
            output,
            reuse_checkpoint=reuse_checkpoints,
            global_binary_weight=global_binary_weight,
            patch_binary_weight=patch_binary_weight,
        )
        mask = folds == fold
        oof_class[mask], oof_global[mask], oof_patch[mask] = validation_output
        test_class.append(test_output[0])
        test_global.append(test_output[1])
        test_patch.append(test_output[2])
        histories.append({"fold": fold, "history": history})
        checkpoints.append(checkpoint)
        checkpoint_variants.append({"fold": fold, "variant": variant})

    active = np.isfinite(oof_global)
    base = _softmax(oof_class[active])
    global_hierarchical = _hierarchical_probabilities(base, oof_global[active])
    patch_hierarchical = _hierarchical_probabilities(base, oof_patch[active])
    selected = None
    for base_weight in (0.50, 0.65, 0.80):
        for patch_share in (0.25, 0.50, 0.75, 1.00):
            specialist = (
                (1 - patch_share) * global_hierarchical
                + patch_share * patch_hierarchical
            )
            probabilities = base_weight * base + (1 - base_weight) * specialist
            metrics = _score(labels[active], probabilities.argmax(1))
            candidate = (
                metrics["macro_f1"],
                -base_weight,
                -abs(patch_share - 0.5),
                base_weight,
                patch_share,
                metrics,
                probabilities,
            )
            if selected is None or candidate[:3] > selected[:3]:
                selected = candidate

    test_base = _softmax(np.mean(test_class, axis=0))
    test_global_hierarchical = _hierarchical_probabilities(
        test_base, np.mean(test_global, axis=0)
    )
    test_patch_hierarchical = _hierarchical_probabilities(
        test_base, np.mean(test_patch, axis=0)
    )
    base_weight, patch_share = selected[3], selected[4]
    test_specialist = (
        (1 - patch_share) * test_global_hierarchical
        + patch_share * test_patch_hierarchical
    )
    test_probabilities = base_weight * test_base + (1 - base_weight) * test_specialist

    oof_predictions = np.full(len(manifest), -1, dtype=int)
    oof_predictions[active] = selected[6].argmax(1)
    np.savez_compressed(
        output / "oof_probabilities.npz",
        class_logits=oof_class,
        global_binary_logits=oof_global,
        patch_binary_logits=oof_patch,
        labels=labels,
        folds=folds,
        selected_probabilities=selected[6].astype(np.float32),
        active_mask=active,
    )
    np.savez_compressed(
        output / "test_probabilities.npz",
        probabilities=test_probabilities.astype(np.float32),
        class_logits=np.mean(test_class, axis=0).astype(np.float32),
        global_binary_logits=np.mean(test_global, axis=0).astype(np.float32),
        patch_binary_logits=np.mean(test_patch, axis=0).astype(np.float32),
    )
    return {
        "method": "SigLIP2 global classifier + global/patch 0-vs-2 heads + EMA",
        "selected_base_weight": float(base_weight),
        "selected_patch_share": float(patch_share),
        "checkpoint_variants": checkpoint_variants,
        "validation": selected[5],
        "fold_metrics": _fold_report(labels, folds, oof_predictions, active_folds),
        "histories": histories,
        "checkpoints": checkpoints,
        "test_predictions": test_probabilities.argmax(1),
    }


def _extract_features(kind, paths, output_path):
    if output_path.exists():
        return np.load(output_path, allow_pickle=False)["features"]
    config = MODEL_CONFIGS[kind]
    model = _build_model(kind, multitask=False)
    loader = _loader(paths, None, config["size"], False, config["batch"], kind=kind)
    features = []
    model.eval()
    with torch.no_grad():
        for images, _, _, _ in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = model.encode(images.cuda(non_blocking=True))
            features.append(encoded.float().cpu().numpy())
    array = np.concatenate(features)
    np.savez_compressed(output_path, features=array.astype(np.float32))
    del model
    torch.cuda.empty_cache()
    return array


def _inner_teacher_probabilities(features, labels, groups, outer_train_indices):
    probabilities = np.zeros((len(outer_train_indices), NUM_CLASSES), dtype=np.float64)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    local_labels = labels[outer_train_indices]
    local_groups = groups[outer_train_indices]
    for inner_train, inner_validation in splitter.split(features[outer_train_indices], local_labels, local_groups):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED),
        )
        classifier.fit(features[outer_train_indices[inner_train]], local_labels[inner_train])
        probabilities[inner_validation] = classifier.predict_proba(features[outer_train_indices[inner_validation]])
    return probabilities


def _run_label_refinement(manifest, folds, test_paths, active_folds, output):
    labels = manifest.label.to_numpy()
    groups = manifest.group.to_numpy()
    paths = manifest.path.tolist()
    teacher_features = [
        _extract_features("siglip2", paths, output / "teacher_siglip2_features.npz"),
        _extract_features("dino", paths, output / "teacher_dino_features.npz"),
    ]
    oof_logits = np.full((len(manifest), NUM_CLASSES), np.nan, dtype=np.float32)
    test_logits, histories, checkpoints, reliability_counts = [], [], [], []

    for fold in active_folds:
        train_indices = np.flatnonzero(folds != fold)
        validation_indices = np.flatnonzero(folds == fold)
        teacher_sets = [
            _inner_teacher_probabilities(features, labels, groups, train_indices)
            for features in teacher_features
        ]
        teacher_probabilities = np.mean(teacher_sets, axis=0)
        official = np.eye(NUM_CLASSES, dtype=np.float32)[labels[train_indices]]
        teacher_prediction = teacher_probabilities.argmax(1)
        teacher_confidence = teacher_probabilities.max(1)
        official_probability = teacher_probabilities[np.arange(len(train_indices)), labels[train_indices]]
        teacher_votes = np.column_stack([probabilities.argmax(1) for probabilities in teacher_sets])
        teachers_agree = np.all(teacher_votes == teacher_votes[:, [0]], axis=1)
        clean = teachers_agree & (teacher_prediction == labels[train_indices]) & (official_probability >= 0.80)
        likely_noisy = teachers_agree & (teacher_prediction != labels[train_indices]) & (teacher_confidence >= 0.95)
        ambiguous = ~(clean | likely_noisy)
        targets = official.copy()
        targets[ambiguous] = 0.85 * official[ambiguous] + 0.15 * teacher_probabilities[ambiguous]
        targets[likely_noisy] = 0.60 * official[likely_noisy] + 0.40 * teacher_probabilities[likely_noisy]
        weights = np.ones(len(train_indices), dtype=np.float32)
        weights[likely_noisy] = 0.50
        hard_pair = ambiguous & np.isin(labels[train_indices], [0, 2])
        weights[hard_pair] *= 1.50
        curriculum_groups = np.ones(len(train_indices), dtype=np.int64)
        curriculum_groups[clean] = 0
        curriculum_groups[likely_noisy] = 2

        train_frame = manifest.iloc[train_indices].reset_index(drop=True)
        validation_frame = manifest.iloc[validation_indices].reset_index(drop=True)
        validation_output, test_output, history, checkpoint = _train_fold(
            "siglip2",
            train_frame,
            validation_frame,
            test_paths,
            fold,
            output,
            soft_targets=targets,
            sample_weights=weights,
            curriculum_groups=curriculum_groups,
        )
        oof_logits[validation_indices] = validation_output
        test_logits.append(test_output)
        histories.append({"fold": fold, "history": history})
        checkpoints.append(checkpoint)
        reliability_counts.append(
            {
                "fold": fold,
                "clean": int(clean.sum()),
                "ambiguous": int(ambiguous.sum()),
                "likely_noisy": int(likely_noisy.sum()),
            }
        )

    mask = np.isfinite(oof_logits).all(1)
    predictions = oof_logits[mask].argmax(1)
    full_predictions = np.full(len(manifest), -1, dtype=int)
    full_predictions[mask] = predictions
    np.savez_compressed(output / "oof_probabilities.npz", logits=oof_logits, labels=labels, folds=folds)
    return {
        "method": "cross-fitted frozen teachers + noise-aware SigLIP2 student",
        "validation": _score(labels[mask], predictions),
        "fold_metrics": _fold_report(labels, folds, full_predictions, active_folds),
        "reliability_counts": reliability_counts,
        "histories": histories,
        "checkpoints": checkpoints,
        "test_predictions": np.mean(test_logits, axis=0).argmax(1),
    }


def _stack_features(probability_sets):
    columns = []
    for probabilities in probability_sets:
        clipped = np.clip(probabilities, 1e-8, 1.0)
        columns.extend(
            [
                clipped,
                -(clipped * np.log(clipped)).sum(1, keepdims=True),
                np.log(clipped[:, [0]] / clipped[:, [2]]),
                (clipped[:, [0]] - clipped[:, [2]]),
            ]
        )
    margins = np.column_stack([probabilities[:, 0] - probabilities[:, 2] for probabilities in probability_sets])
    columns.extend([margins.mean(1, keepdims=True), margins.std(1, keepdims=True), margins.min(1, keepdims=True), margins.max(1, keepdims=True)])
    return np.column_stack(columns)


def _run_tri_encoder(
    manifest,
    folds,
    test_paths,
    active_folds,
    output,
    reuse_checkpoints=False,
):
    labels = manifest.label.to_numpy()
    model_names = ("siglip2", "pe_core", "dino")
    oof_probabilities, test_probabilities, all_histories, checkpoints = [], [], {}, []
    active_mask = np.isin(folds, active_folds)
    for kind in model_names:
        oof_logits = np.full((len(manifest), NUM_CLASSES), np.nan, dtype=np.float32)
        fold_test, histories = [], []
        for fold in active_folds:
            train = manifest[folds != fold].reset_index(drop=True)
            validation = manifest[folds == fold].reset_index(drop=True)
            validation_logits, test_logits, history, checkpoint = _train_fold(
                kind,
                train,
                validation,
                test_paths,
                fold,
                output,
                reuse_checkpoint=reuse_checkpoints,
            )
            oof_logits[folds == fold] = validation_logits
            fold_test.append(test_logits)
            histories.append({"fold": fold, "history": history})
            checkpoints.append(checkpoint)
        oof_probabilities.append(_softmax(oof_logits[active_mask]))
        test_probabilities.append(_softmax(np.mean(fold_test, axis=0)))
        all_histories[kind] = histories

    validation_labels = labels[active_mask]
    validation_folds = folds[active_mask]
    validation_features = _stack_features(oof_probabilities)
    test_features = _stack_features(test_probabilities)

    if len(active_folds) == 1:
        validation_probability = np.mean(oof_probabilities, axis=0)
        test_probability = np.mean(test_probabilities, axis=0)
        selected = {"method": "pilot probability average", "C": None}
    else:
        selected = None
        selected_oof = None
        for regularization in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            oof = np.zeros((len(validation_labels), NUM_CLASSES), dtype=np.float64)
            for fold in active_folds:
                train_mask = validation_folds != fold
                validation_mask = validation_folds == fold
                classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=regularization, class_weight="balanced", max_iter=3000, random_state=SEED),
                )
                classifier.fit(validation_features[train_mask], validation_labels[train_mask])
                oof[validation_mask] = classifier.predict_proba(validation_features[validation_mask])
            metrics = _score(validation_labels, oof.argmax(1))
            candidate = (metrics["macro_f1"], -regularization, regularization, metrics)
            if selected is None or candidate[:2] > selected[:2]:
                selected, selected_oof = candidate, oof
        regularization = selected[2]
        validation_probability = selected_oof
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=regularization, class_weight="balanced", max_iter=3000, random_state=SEED),
        )
        classifier.fit(validation_features, validation_labels)
        test_probability = classifier.predict_proba(test_features)
        selected = {"method": "cross-fitted multinomial logistic stack", "C": regularization}

        binary = validation_labels != 1
        binary_validation = np.zeros(len(validation_labels), dtype=np.float64)
        for fold in active_folds:
            binary_train = binary & (validation_folds != fold)
            fold_validation = validation_folds == fold
            fold_binary_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000, random_state=SEED),
            )
            fold_binary_model.fit(
                validation_features[binary_train],
                (validation_labels[binary_train] == 0).astype(int),
            )
            binary_validation[fold_validation] = fold_binary_model.predict_proba(
                validation_features[fold_validation]
            )[:, 1]
        binary_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000, random_state=SEED),
        )
        binary_model.fit(validation_features[binary], (validation_labels[binary] == 0).astype(int))
        binary_test = binary_model.predict_proba(test_features)[:, 1]
        baseline = validation_probability.argmax(1)
        best_gate = (_score(validation_labels, baseline)["macro_f1"], 1.01, baseline)
        for threshold in (0.80, 0.85, 0.90, 0.925, 0.95):
            gated = baseline.copy()
            candidate_mask = gated != 1
            gated[candidate_mask & (binary_validation >= threshold)] = 0
            gated[candidate_mask & (binary_validation <= 1 - threshold)] = 2
            score = _score(validation_labels, gated)["macro_f1"]
            if score > best_gate[0]:
                best_gate = (score, threshold, gated)
        if best_gate[1] <= 1.0:
            test_predictions = test_probability.argmax(1)
            candidate_mask = test_predictions != 1
            test_predictions[candidate_mask & (binary_test >= best_gate[1])] = 0
            test_predictions[candidate_mask & (binary_test <= 1 - best_gate[1])] = 2
            validation_predictions = best_gate[2]
            selected["binary_threshold"] = best_gate[1]
        else:
            test_predictions = test_probability.argmax(1)
            validation_predictions = baseline
            selected["binary_threshold"] = None

    if len(active_folds) == 1:
        validation_predictions = validation_probability.argmax(1)
        test_predictions = test_probability.argmax(1)

    full_oof_predictions = np.full(len(manifest), -1, dtype=int)
    full_oof_predictions[active_mask] = validation_predictions
    np.savez_compressed(
        output / "stack_probabilities.npz",
        labels=labels,
        folds=folds,
        active_mask=active_mask,
        validation_probabilities=validation_probability,
        test_probabilities=test_probability,
    )
    return {
        "method": "SigLIP2 + PE-Core + DINOv3 cross-fitted mixture of experts",
        "selected_stack": selected,
        "validation": _score(validation_labels, validation_predictions),
        "fold_metrics": _fold_report(labels, folds, full_oof_predictions, active_folds),
        "histories": all_histories,
        "checkpoints": checkpoints,
        "test_predictions": test_predictions,
    }


def self_check():
    import numpy as np

    logits = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
    probabilities = _softmax(logits)
    assert probabilities.shape == (2, 3)
    assert np.allclose(probabilities.sum(1), 1.0)
    hierarchical = _hierarchical_probabilities(
        probabilities, np.array([8.0, -8.0])
    )
    assert np.allclose(hierarchical.sum(1), 1.0)
    assert hierarchical[0, 0] > hierarchical[0, 2]
    assert hierarchical[1, 2] > hierarchical[1, 0]
    features = _stack_features([probabilities, probabilities])
    assert features.shape[0] == 2 and np.isfinite(features).all()
    gate = _tri_consensus_gate(
        np.array([0, 1, 2]),
        np.array([0.1, 0.1, 0.1]),
        np.array([0.8, 0.8, 0.8]),
        np.array([0.05, 0.99, 0.95]),
        np.array([0.05, 0.99, 0.95]),
        {
            "margin_max": 0.2,
            "entropy_min": 0.5,
            "patch_confidence": 0.9,
            "dino_confidence": 0.9,
        },
    )
    assert gate.tolist() == [True, False, True]
    residual = _residual_router_gate(
        np.array([0, 1, 2, 2]),
        np.array([0.05, 0.99, 0.95, 0.80]),
        np.array([0.1, 0.1, 0.1, 0.8]),
        np.array([False, True, False, False]),
        threshold=0.9,
        margin_max=0.2,
        max_fraction=1.0,
    )
    assert residual.tolist() == [True, False, True, False]


if __name__ == "__main__":
    self_check()
