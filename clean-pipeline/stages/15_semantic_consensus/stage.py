"""Validation-selected consensus across material and semantic encoders.

Local CPU only. No test labels or ground-truth files are read.

  python stages/15_semantic_consensus/stage.py
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


SCRIPT = Path(__file__).resolve()
ROOT = next(
    (path for path in SCRIPT.parents if (path / "artifacts").is_dir() and (path / "BDC2026").is_dir()),
    Path.cwd(),
)
RUN17 = ROOT / "artifacts" / "material_context_fusion"
RUN25 = ROOT / "artifacts" / "high_resolution_gate"
RUN11 = ROOT / "artifacts" / "class_calibration"
DEFAULT_OUTPUT = ROOT / "artifacts" / "semantic_consensus"
THRESHOLDS = np.round(np.arange(0.34, 0.951, 0.01), 2)
SEED = 2026


def score(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "class_f1": f1_score(
            labels, predictions, labels=[0, 1, 2], average=None
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


def apply_consensus(anchor, siglip2, run11, threshold):
    predictions = anchor.copy()
    gate = (
        (anchor == 2)
        & (siglip2 == 0)
        & (run11.argmax(axis=1) == 0)
        & (run11[:, 0] >= threshold)
    )
    predictions[gate] = 0
    return predictions, gate


def run(output_dir):
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = np.load(RUN25 / "validation_probabilities.npz", allow_pickle=False)
    test = np.load(RUN25 / "test_probabilities.npz", allow_pickle=False)
    run11_validation = np.load(
        RUN11 / "validation_probabilities.npz", allow_pickle=False
    )
    run11_test = np.load(RUN11 / "test_probabilities.npz", allow_pickle=False)

    labels = validation["labels"].astype(np.int64)
    folds = validation["inner_folds"].astype(np.int64)
    ids = test["ids"].astype(np.int64)
    if not np.array_equal(labels, run11_validation["labels"]):
        raise ValueError("Run25 and Run11 validation labels differ")
    if not np.array_equal(folds, run11_validation["inner_folds"]):
        raise ValueError("Run25 and Run11 validation folds differ")
    if not np.array_equal(ids, run11_test["ids"]):
        raise ValueError("Run25 and Run11 test IDs differ")

    anchor_validation = validation["run17_gated_probabilities"].argmax(axis=1)
    anchor_test = test["run17_gated_predictions"].astype(np.int64)
    siglip2_validation = validation["siglip2_probabilities"].argmax(axis=1)
    siglip2_test = test["siglip2_probabilities"].argmax(axis=1)
    expert_validation = run11_validation["calibrated_probabilities"].astype(np.float64)
    expert_test = run11_test["calibrated_probabilities"].astype(np.float64)

    saved_anchor = (
        pd.read_csv(RUN17 / "submission_material_context_gated.csv")
        .set_index("id")
        .loc[ids, "predicted"]
        .to_numpy(dtype=np.int64)
    )
    if not np.array_equal(anchor_test, saved_anchor):
        raise ValueError("Run25 anchor differs from saved Run17 gated submission")

    baseline = score(labels, anchor_validation)
    baseline_folds = fold_scores(labels, anchor_validation, folds)
    candidates = []
    for threshold in THRESHOLDS:
        predictions, gate = apply_consensus(
            anchor_validation, siglip2_validation, expert_validation, threshold
        )
        metrics = score(labels, predictions)
        current_folds = fold_scores(labels, predictions, folds)
        metrics.update(
            {
                "threshold": float(threshold),
                "changed_validation_rows": int(gate.sum()),
                "fold_macro_f1": current_folds,
                "mean_fold_macro_f1": float(np.mean(current_folds)),
                "non_degrading_folds": int(
                    sum(
                        new >= old - 1e-12
                        for new, old in zip(current_folds, baseline_folds)
                    )
                ),
            }
        )
        candidates.append(metrics)

    eligible = [
        row
        for row in candidates
        if row["macro_f1"] > baseline["macro_f1"] + 1e-12
        and row["mean_fold_macro_f1"] >= np.mean(baseline_folds) - 1e-12
        and row["non_degrading_folds"] >= 4
    ]
    selected = max(
        eligible,
        key=lambda row: (
            row["macro_f1"],
            row["mean_fold_macro_f1"],
            row["non_degrading_folds"],
            -row["changed_validation_rows"],
            row["threshold"],
        ),
        default={
            **baseline,
            "threshold": 1.01,
            "changed_validation_rows": 0,
            "fold_macro_f1": baseline_folds,
            "mean_fold_macro_f1": float(np.mean(baseline_folds)),
            "non_degrading_folds": len(baseline_folds),
        },
    )

    validation_predictions, validation_gate = apply_consensus(
        anchor_validation,
        siglip2_validation,
        expert_validation,
        selected["threshold"],
    )
    test_predictions, test_gate = apply_consensus(
        anchor_test, siglip2_test, expert_test, selected["threshold"]
    )

    template = pd.read_csv(ROOT / "BDC2026" / "submission.csv")[["id"]]

    def submission(predictions):
        result = template.copy()
        result["predicted"] = result["id"].astype(int).map(
            dict(zip(ids, predictions.astype(int), strict=True))
        )
        if len(result) != 1_458 or result["predicted"].isna().any():
            raise ValueError("Invalid submission mapping")
        result["predicted"] = result["predicted"].astype(int)
        return result

    submission(anchor_test).to_csv(
        output_dir / "submission_semantic_consensus_material_context.csv", index=False
    )
    submission(test_predictions).to_csv(
        output_dir / "submission_semantic_consensus_guarded_consensus.csv", index=False
    )

    validation_output = pd.DataFrame(
        {
            "groundtruth": labels,
            "inner_fold": folds,
            "run17_gated_predicted": anchor_validation,
            "siglip2_predicted": siglip2_validation,
            "run11_calibrated_predicted": expert_validation.argmax(axis=1),
            "run11_p0": expert_validation[:, 0],
            "consensus_gate": validation_gate,
            "predicted": validation_predictions,
        }
    )
    validation_output.to_csv(output_dir / "validation_predictions.csv", index=False)
    np.savez(
        output_dir / "test_probabilities.npz",
        ids=ids,
        run17_gated_predictions=anchor_test,
        siglip2_predictions=siglip2_test,
        run11_calibrated_probabilities=expert_test.astype(np.float32),
        consensus_gate=test_gate,
        predictions=test_predictions,
    )

    report = {
        "method_version": 1,
        "method": "Run17 gated + SigLIP2 + Run11 calibrated guarded consensus",
        "test_labels_used": False,
        "seed": SEED,
        "selection": "validation aggregate gain, mean-fold non-loss, >=4/5 non-degrading folds",
        "baseline_validation": baseline,
        "selected": selected,
        "selected_validation": score(labels, validation_predictions),
        "changed_test_rows": int(test_gate.sum()),
        "changed_test_ids": ids[test_gate].astype(int).tolist(),
        "test_class_counts": {
            str(label): int((test_predictions == label).sum()) for label in range(3)
        },
        "top_candidates": sorted(
            candidates,
            key=lambda row: (
                row["macro_f1"],
                row["mean_fold_macro_f1"],
                row["non_degrading_folds"],
            ),
            reverse=True,
        )[:15],
        "timing_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Selected threshold:", selected["threshold"])
    print(
        "Validation Macro-F1:",
        baseline["macro_f1"],
        "->",
        report["selected_validation"]["macro_f1"],
    )
    print("Changed test IDs:", report["changed_test_ids"])
    print("Timing seconds:", report["timing_seconds"])
    print("Wrote artifacts to", output_dir.resolve())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output_dir.resolve())
