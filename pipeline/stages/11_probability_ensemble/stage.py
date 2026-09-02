"""Validation-selected context soup + class-0 calibration. No training."""

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


SEED = 2026
WEIGHTS = np.arange(0.0, 0.5001, 0.025)
BIAS_GRID = np.arange(-0.20, 0.4001, 0.005)
THRESHOLD_GRID = np.arange(0.30, 0.5001, 0.005)
LOGISTIC_C = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

SCRIPT = Path(__file__).resolve()
ROOT = next(path for path in SCRIPT.parents if (path / "BDC2026").is_dir())
RUN13 = ROOT / "artifacts" / "context_encoder"
RUN13B = ROOT / "artifacts" / "context_refinement"


def metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "class_f1": f1_score(
            labels, predictions, average=None, labels=[0, 1, 2]
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1, 2]
        ).tolist(),
        "errors": int((labels != predictions).sum()),
    }


def fold_scores(labels, predictions, folds):
    return [
        float(
            f1_score(
                labels[folds == fold], predictions[folds == fold], average="macro"
            )
        )
        for fold in sorted(np.unique(folds))
    ]


def candidate(method, labels, predictions, folds, baseline_folds, **params):
    scores = fold_scores(labels, predictions, folds)
    return {
        "method": method,
        **params,
        **metrics(labels, predictions),
        "fold_macro_f1": scores,
        "mean_fold_macro_f1": float(np.mean(scores)),
        "non_degrading_folds": int(
            sum(new >= old - 1e-12 for new, old in zip(scores, baseline_folds))
        ),
    }


def main(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    v13 = np.load(RUN13 / "validation_probabilities.npz")
    t13 = np.load(RUN13 / "test_probabilities.npz")
    v13b = np.load(RUN13B / "validation_probabilities.npz")
    t13b = np.load(RUN13B / "test_probabilities.npz")
    labels = v13b["labels"].astype(np.int64)
    folds = v13b["inner_folds"].astype(np.int64)
    ids = t13b["ids"].astype(np.int64)
    if not np.array_equal(labels, v13["labels"]):
        raise ValueError("Run13 and Run13b validation labels differ")
    if not np.array_equal(ids, t13["ids"]):
        raise ValueError("Run13 and Run13b test IDs differ")

    validation_models = [
        v13b["run11_probabilities"].astype(np.float64),
        v13["siglip_probabilities"].astype(np.float64),
        v13b["siglip_no_tta_probabilities"].astype(np.float64),
    ]
    test_models = [
        t13b["run11_probabilities"].astype(np.float64),
        t13["siglip_probabilities"].astype(np.float64),
        t13b["siglip_no_tta_probabilities"].astype(np.float64),
    ]

    run13b_predictions = v13b["blended_probabilities"].argmax(axis=1)
    run13b_metrics = metrics(labels, run13b_predictions)
    run13b_folds = fold_scores(labels, run13b_predictions, folds)
    soup_candidates = []
    for old_siglip_weight in WEIGHTS:
        for resumed_siglip_weight in WEIGHTS:
            if old_siglip_weight + resumed_siglip_weight > 0.6001:
                continue
            run11_weight = 1.0 - old_siglip_weight - resumed_siglip_weight
            probabilities = (
                run11_weight * validation_models[0]
                + old_siglip_weight * validation_models[1]
                + resumed_siglip_weight * validation_models[2]
            )
            soup_candidates.append(
                candidate(
                    "soup",
                    labels,
                    probabilities.argmax(axis=1),
                    folds,
                    run13b_folds,
                    run11_weight=float(run11_weight),
                    old_siglip_weight=float(old_siglip_weight),
                    resumed_siglip_weight=float(resumed_siglip_weight),
                )
            )
    eligible_soups = [
        row
        for row in soup_candidates
        if row["macro_f1"] > run13b_metrics["macro_f1"]
        and row["mean_fold_macro_f1"] >= np.mean(run13b_folds) - 1e-12
        and row["non_degrading_folds"] >= 3
    ]
    selected_soup = max(
        eligible_soups,
        key=lambda row: (
            row["macro_f1"],
            row["mean_fold_macro_f1"],
            row["non_degrading_folds"],
            -row["resumed_siglip_weight"],
        ),
        default={
            "run11_weight": 0.0,
            "old_siglip_weight": 0.0,
            "resumed_siglip_weight": 0.0,
            "fallback": "Run13b",
        },
    )
    if eligible_soups:
        weights = np.array(
            [
                selected_soup["run11_weight"],
                selected_soup["old_siglip_weight"],
                selected_soup["resumed_siglip_weight"],
            ]
        )
        validation_soup = sum(
            weight * probabilities
            for weight, probabilities in zip(weights, validation_models)
        )
        test_soup = sum(
            weight * probabilities for weight, probabilities in zip(weights, test_models)
        )
    else:
        validation_soup = v13b["blended_probabilities"].astype(np.float64)
        test_soup = t13b["blended_probabilities"].astype(np.float64)

    baseline_predictions = validation_soup.argmax(axis=1)
    baseline_metrics = metrics(labels, baseline_predictions)
    baseline_folds = fold_scores(labels, baseline_predictions, folds)
    calibration_candidates = []
    candidate_predictions = {}

    for bias in BIAS_GRID:
        adjusted = validation_soup.copy()
        adjusted[:, 0] *= np.exp(bias)
        predictions = adjusted.argmax(axis=1)
        row = candidate(
            "class0_logit_bias",
            labels,
            predictions,
            folds,
            baseline_folds,
            class0_logit_bias=float(bias),
        )
        calibration_candidates.append(row)
        candidate_predictions[(row["method"], float(bias))] = predictions

    pair_ratio = validation_soup[:, 0] / (
        validation_soup[:, 0] + validation_soup[:, 2] + 1e-12
    )
    for threshold in THRESHOLD_GRID:
        predictions = baseline_predictions.copy()
        predictions[(predictions == 2) & (pair_ratio >= threshold)] = 0
        row = candidate(
            "directional_threshold",
            labels,
            predictions,
            folds,
            baseline_folds,
            threshold_0_vs_2=float(threshold),
        )
        calibration_candidates.append(row)
        candidate_predictions[(row["method"], float(threshold))] = predictions

    validation_features = np.stack(
        [
            np.log(
                np.clip(probabilities[:, 0], 1e-8, 1.0)
                / np.clip(probabilities[:, 2], 1e-8, 1.0)
            )
            for probabilities in validation_models
        ],
        axis=1,
    )
    test_features = np.stack(
        [
            np.log(
                np.clip(probabilities[:, 0], 1e-8, 1.0)
                / np.clip(probabilities[:, 2], 1e-8, 1.0)
            )
            for probabilities in test_models
        ],
        axis=1,
    )
    for class_weight in (None, "balanced"):
        for regularization in LOGISTIC_C:
            predictions = baseline_predictions.copy()
            for fold in sorted(np.unique(folds)):
                train_mask = (folds != fold) & (labels != 1)
                valid_mask = (folds == fold) & (baseline_predictions != 1)
                model = LogisticRegression(
                    C=regularization,
                    class_weight=class_weight,
                    random_state=SEED,
                    max_iter=1_000,
                ).fit(validation_features[train_mask], labels[train_mask])
                predictions[valid_mask] = model.predict(
                    validation_features[valid_mask]
                )
            row = candidate(
                "binary_logistic_stack",
                labels,
                predictions,
                folds,
                baseline_folds,
                C=regularization,
                class_weight=class_weight,
            )
            calibration_candidates.append(row)
            candidate_predictions[(row["method"], regularization, class_weight)] = predictions

    eligible = [
        row
        for row in calibration_candidates
        if row["macro_f1"] > baseline_metrics["macro_f1"] + 1e-12
        and row["mean_fold_macro_f1"] >= np.mean(baseline_folds) - 1e-12
        and row["non_degrading_folds"] >= 3
    ]
    selected = max(
        eligible,
        key=lambda row: (
            row["macro_f1"],
            row["mean_fold_macro_f1"],
            row["non_degrading_folds"],
        ),
        default={"method": "none", "fallback": "SigLIP soup"},
    )

    if selected["method"] == "class0_logit_bias":
        test_adjusted = test_soup.copy()
        test_adjusted[:, 0] *= np.exp(selected["class0_logit_bias"])
        validation_predictions = candidate_predictions[
            (selected["method"], selected["class0_logit_bias"])
        ]
        test_predictions = test_adjusted.argmax(axis=1)
    elif selected["method"] == "directional_threshold":
        validation_predictions = candidate_predictions[
            (selected["method"], selected["threshold_0_vs_2"])
        ]
        test_predictions = test_soup.argmax(axis=1)
        test_ratio = test_soup[:, 0] / (test_soup[:, 0] + test_soup[:, 2] + 1e-12)
        test_predictions[
            (test_predictions == 2) & (test_ratio >= selected["threshold_0_vs_2"])
        ] = 0
    elif selected["method"] == "binary_logistic_stack":
        validation_predictions = candidate_predictions[
            (selected["method"], selected["C"], selected["class_weight"])
        ]
        test_predictions = test_soup.argmax(axis=1)
        train_mask = labels != 1
        test_mask = test_predictions != 1
        model = LogisticRegression(
            C=selected["C"],
            class_weight=selected["class_weight"],
            random_state=SEED,
            max_iter=1_000,
        ).fit(validation_features[train_mask], labels[train_mask])
        test_predictions[test_mask] = model.predict(test_features[test_mask])
    else:
        validation_predictions = baseline_predictions
        test_predictions = test_soup.argmax(axis=1)

    template = pd.read_csv(ROOT / "BDC2026" / "submission.csv")[["id"]]
    if len(template) != len(ids):
        raise ValueError("Submission template and test probabilities differ")
    mapping = dict(zip(ids, test_predictions.astype(int), strict=True))
    recommended = template.copy()
    recommended["predicted"] = recommended["id"].astype(int).map(mapping)
    if recommended["predicted"].isna().any():
        raise ValueError("Missing test ID mapping")
    recommended["predicted"] = recommended["predicted"].astype(int)
    soup = template.copy()
    soup["predicted"] = soup["id"].astype(int).map(
        dict(zip(ids, test_soup.argmax(axis=1).astype(int), strict=True))
    ).astype(int)

    soup.to_csv(output / "submission_probability_ensemble.csv", index=False)
    recommended.to_csv(output / "submission_probability_calibrated.csv", index=False)
    np.savez(
        output / "validation_probabilities.npz",
        labels=labels,
        inner_folds=folds,
        soup_probabilities=validation_soup.astype(np.float32),
        predictions=validation_predictions.astype(np.int64),
    )
    np.savez(
        output / "test_probabilities.npz",
        ids=ids,
        soup_probabilities=test_soup.astype(np.float32),
        predictions=test_predictions.astype(np.int64),
    )
    report = {
        "method": "validation-selected SigLIP soup + guarded class-0 calibration",
        "seed": SEED,
        "test_labels_used": False,
        "soup_parameters": selected_soup,
        "baseline_validation": baseline_metrics,
        "selected_calibration": selected,
        "selected_validation": metrics(labels, validation_predictions),
        "test_changed_rows_vs_soup": int(
            (test_predictions != test_soup.argmax(axis=1)).sum()
        ),
        "top_calibration_candidates": sorted(
            calibration_candidates,
            key=lambda row: (
                row["macro_f1"],
                row["mean_fold_macro_f1"],
                row["non_degrading_folds"],
            ),
            reverse=True,
        )[:20],
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Soup validation Macro-F1:", baseline_metrics["macro_f1"])
    print("Selected calibration:", selected)
    print("Selected validation Macro-F1:", report["selected_validation"]["macro_f1"])
    print("Changed test rows:", report["test_changed_rows_vs_soup"])
    print("Wrote Run15 artifacts to", output.resolve())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-dir", default=str(SCRIPT.parent))
    main(parser.parse_args().output_dir)
