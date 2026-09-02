"""Validation-gated residual consensus for hierarchical and Patch-MIL experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


SCRIPT = Path(__file__).resolve()
ROOT = next(
    (path for path in (*SCRIPT.parents, Path.cwd()) if (path / "modal_backbone_app.py").is_file()),
    Path.cwd(),
)
ROUTER_ROOT = ROOT / "stages/18_oof_decision_router"
EXPERT_ROOT = ROOT / "artifacts/hierarchical_patch_experts"
OUTPUT = ROOT / "artifacts/hierarchical_patch_consensus"


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def score(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "class_f1": f1_score(labels, predictions, labels=[0, 1, 2], average=None).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "errors": int((labels != predictions).sum()),
    }


def fold_scores(labels: np.ndarray, folds: np.ndarray, predictions: np.ndarray) -> list[float]:
    return [
        float(f1_score(labels[folds == fold], predictions[folds == fold], average="macro"))
        for fold in sorted(np.unique(folds))
    ]


def hierarchical_probabilities(
    class_logits: np.ndarray,
    binary_logits: np.ndarray,
    alpha: float,
) -> np.ndarray:
    base = softmax(class_logits)
    q = 1.0 / (1.0 + np.exp(-binary_logits))
    pair = np.column_stack(
        [
            (1.0 - base[:, 1]) * q,
            base[:, 1],
            (1.0 - base[:, 1]) * (1.0 - q),
        ]
    )
    return alpha * base + (1.0 - alpha) * pair


def main() -> None:
    global ROOT, ROUTER_ROOT, EXPERT_ROOT, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--router-root", type=Path, default=ROUTER_ROOT)
    parser.add_argument("--expert-root", type=Path, default=EXPERT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    ROOT = args.root.resolve()
    ROUTER_ROOT = args.router_root.resolve()
    EXPERT_ROOT = args.expert_root.resolve()
    OUTPUT = args.output_dir.resolve()

    router = np.load(ROUTER_ROOT / "router_probabilities.npz", allow_pickle=False)
    hierarchical_oof = np.load(EXPERT_ROOT / "hierarchical_oof_probabilities.npz", allow_pickle=False)
    hierarchical_test = np.load(EXPERT_ROOT / "hierarchical_test_probabilities.npz", allow_pickle=False)
    patch_oof = np.load(EXPERT_ROOT / "patch_mil_oof_probabilities.npz", allow_pickle=False)
    patch_test = np.load(EXPERT_ROOT / "patch_mil_test_probabilities.npz", allow_pickle=False)
    hierarchical_metrics = json.loads(
        (EXPERT_ROOT / "hierarchical_metrics.json").read_text(encoding="utf-8")
    )

    labels = router["validation_labels"].astype(np.int64)
    folds = router["validation_folds"].astype(np.int64)
    anchor_validation = router["validation_final_predictions"].astype(np.int64)
    anchor_test = router["final_predictions"].astype(np.int64)
    ids = router["ids"].astype(np.int64)

    if not np.isfinite(hierarchical_oof["logits"]).all():
        raise ValueError("Hierarchical expert lacks full five-fold OOF predictions")
    if not patch_oof["active_mask"].all():
        raise ValueError("Patch-MIL expert lacks full five-fold OOF predictions")
    official_folds = hierarchical_oof["folds"].astype(np.int64)
    validation_mask = official_folds == 0
    if not np.array_equal(official_folds, patch_oof["folds"].astype(np.int64)):
        raise ValueError("Expert fold assignments differ")
    if not np.array_equal(labels, hierarchical_oof["labels"][validation_mask].astype(np.int64)):
        raise ValueError("Hierarchical validation rows differ from anchor")
    if not np.array_equal(labels, patch_oof["labels"][validation_mask].astype(np.int64)):
        raise ValueError("Patch-MIL validation rows differ from anchor")

    alpha = float(hierarchical_metrics["selected_alpha_3class"])
    hierarchical_validation = hierarchical_probabilities(
        hierarchical_oof["logits"][validation_mask].astype(np.float64),
        hierarchical_oof["binary_logits"][validation_mask].astype(np.float64),
        alpha,
    )
    hierarchical_test_probability = hierarchical_test["probabilities"].astype(np.float64)
    patch_validation = patch_oof["selected_probabilities"][validation_mask].astype(np.float64)
    patch_test_probability = patch_test["probabilities"].astype(np.float64)
    if hierarchical_test_probability.shape != patch_test_probability.shape or hierarchical_test_probability.shape != (1_458, 3):
        raise ValueError("Invalid expert test probability shapes")

    baseline = score(labels, anchor_validation)
    baseline_folds = fold_scores(labels, folds, anchor_validation)

    def apply(
        anchor: np.ndarray,
        hierarchical: np.ndarray,
        patch: np.ndarray,
        hierarchical_threshold: float,
        patch_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        gate = (
            (anchor == 2)
            & (hierarchical.argmax(axis=1) == 0)
            & (patch.argmax(axis=1) == 0)
            & (hierarchical[:, 0] >= hierarchical_threshold)
            & (patch[:, 0] >= patch_threshold)
        )
        predictions = anchor.copy()
        predictions[gate] = 0
        return predictions, gate

    candidates = []
    for hierarchical_threshold in np.arange(0.50, 0.951, 0.01):
        for patch_threshold in np.arange(0.50, 0.951, 0.01):
            predictions, gate = apply(
                anchor_validation,
                hierarchical_validation,
                patch_validation,
                float(hierarchical_threshold),
                float(patch_threshold),
            )
            metrics = score(labels, predictions)
            current_folds = fold_scores(labels, folds, predictions)
            row = {
                **metrics,
                "hierarchical_p0_min": float(hierarchical_threshold),
                "patch_mil_p0_min": float(patch_threshold),
                "changed_validation_rows": int(gate.sum()),
                "fold_macro_f1": current_folds,
                "mean_fold_macro_f1": float(np.mean(current_folds)),
                "non_degrading_folds": int(
                    sum(new >= old - 1e-12 for new, old in zip(current_folds, baseline_folds))
                ),
            }
            row["passes_guard"] = bool(
                row["macro_f1"] > baseline["macro_f1"] + 1e-12
                and row["mean_fold_macro_f1"] >= np.mean(baseline_folds) - 1e-12
                and row["non_degrading_folds"] >= 4
                and abs(row["class_f1"][1] - baseline["class_f1"][1]) <= 1e-12
            )
            candidates.append(row)

    eligible = [row for row in candidates if row["passes_guard"]]
    selected = (
        max(
            eligible,
            key=lambda row: (
                row["macro_f1"],
                row["mean_fold_macro_f1"],
                row["non_degrading_folds"],
                -row["changed_validation_rows"],
                row["hierarchical_p0_min"] + row["patch_mil_p0_min"],
            ),
        )
        if eligible
        else None
    )
    if selected:
        final_validation, validation_gate = apply(
            anchor_validation,
            hierarchical_validation,
            patch_validation,
            selected["hierarchical_p0_min"],
            selected["patch_mil_p0_min"],
        )
        final_test, test_gate = apply(
            anchor_test,
            hierarchical_test_probability,
            patch_test_probability,
            selected["hierarchical_p0_min"],
            selected["patch_mil_p0_min"],
        )
    else:
        final_validation = anchor_validation.copy()
        final_test = anchor_test.copy()
        validation_gate = np.zeros(len(labels), dtype=bool)
        test_gate = np.zeros(len(ids), dtype=bool)

    template = pd.read_csv(ROOT / "BDC2026/submission.csv")[["id"]]
    template["predicted"] = template["id"].astype(int).map(
        dict(zip(ids, final_test.astype(int), strict=True))
    )
    if len(template) != 1_458 or template["predicted"].isna().any():
        raise ValueError("Invalid final submission mapping")
    template["predicted"] = template["predicted"].astype(int)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    template.to_csv(OUTPUT / "submission_final_expert_consensus.csv", index=False)
    np.savez_compressed(
        OUTPUT / "validation_probabilities.npz",
        labels=labels,
        folds=folds,
        hierarchical_probabilities=hierarchical_validation.astype(np.float32),
        patch_mil_probabilities=patch_validation.astype(np.float32),
        gate=validation_gate,
        predictions=final_validation,
    )
    np.savez_compressed(
        OUTPUT / "test_probabilities.npz",
        ids=ids,
        hierarchical_probabilities=hierarchical_test_probability.astype(np.float32),
        patch_mil_probabilities=patch_test_probability.astype(np.float32),
        gate=test_gate,
        predictions=final_test,
    )
    report = {
        "method": "OOF-gated hierarchical and Patch-MIL residual consensus",
        "test_labels_used": False,
        "seed": 2026,
        "selection_source": "official-train OOF only",
        "baseline_validation": baseline,
        "selected": selected,
        "final_validation": score(labels, final_validation),
        "fallback_to_anchor": selected is None,
        "changed_test_rows": int(test_gate.sum()),
        "changed_test_ids": ids[test_gate].astype(int).tolist(),
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Validation Macro-F1:", baseline["macro_f1"], "->", report["final_validation"]["macro_f1"])
    print("Selected:", selected)
    print("Changed test IDs:", report["changed_test_ids"])
    print("Wrote final consensus artifacts to", OUTPUT)


if __name__ == "__main__":
    main()
