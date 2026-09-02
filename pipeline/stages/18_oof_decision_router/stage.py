"""CPU-only OOF binary router plus guarded boundary consensus.

Run from project root:

  python stages/18_oof_decision_router/stage.py

No test labels are read. Outputs conservative and aggressive submissions.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
SEED = 2026


def load(path):
    return np.load(ROOT / path, allow_pickle=False)


def softmax(logits):
    logits = logits.astype(np.float64)
    values = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def features(view_logits, probability_sets):
    eps = 1e-7
    views = softmax(view_logits)
    columns = []
    for index in range(views.shape[1]):
        probabilities = views[:, index]
        columns.extend(
            [
                probabilities[:, 0],
                probabilities[:, 2],
                np.log((probabilities[:, 0] + eps) / (probabilities[:, 2] + eps)),
                probabilities[:, 0] - probabilities[:, 2],
            ]
        )
    odds = np.log((views[:, :, 0] + eps) / (views[:, :, 2] + eps))
    columns.extend(
        [
            odds.mean(axis=1),
            odds.std(axis=1),
            odds.min(axis=1),
            odds.max(axis=1),
            (views.argmax(axis=2) == 0).sum(axis=1),
        ]
    )
    for probabilities in probability_sets:
        probabilities = probabilities.astype(np.float64)
        columns.extend(
            [
                probabilities[:, 0],
                probabilities[:, 2],
                np.log((probabilities[:, 0] + eps) / (probabilities[:, 2] + eps)),
                probabilities[:, 0] - probabilities[:, 2],
                probabilities.max(axis=1),
            ]
        )
    return np.stack(columns, axis=1)


def classifier(c):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            max_iter=3_000,
            class_weight="balanced",
            random_state=SEED,
        ),
    )


def fold_scores(labels, folds, predictions):
    return [
        float(f1_score(labels[folds == fold], predictions[folds == fold], average="macro"))
        for fold in sorted(np.unique(folds))
    ]


def score(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "class_f1": f1_score(labels, predictions, labels=[0, 1, 2], average=None).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "errors": int((labels != predictions).sum()),
    }


def main():
    started = __import__("time").perf_counter()
    validation_frame = pd.read_csv(
        ROOT / "artifacts/boundary_consensus/validation_predictions.csv"
    )
    consensus_test = load("artifacts/boundary_consensus/test_probabilities.npz")
    consensus_metrics = json.loads(
        (ROOT / "artifacts/boundary_consensus/metrics.json").read_text(encoding="utf-8")
    )
    validation29 = load("artifacts/multiview_features/validation_probabilities.npz")
    test29 = load("artifacts/multiview_features/test_probabilities.npz")
    validation11 = load("artifacts/class_calibration/validation_probabilities.npz")
    test11 = load("artifacts/class_calibration/test_probabilities.npz")
    validation15 = load("artifacts/probability_ensemble/validation_probabilities.npz")
    test15 = load("artifacts/probability_ensemble/test_probabilities.npz")
    validation24 = load("artifacts/high_resolution_context/validation_probabilities.npz")
    test24 = load("artifacts/high_resolution_context/test_probabilities.npz")
    validation10 = load("artifacts/neighborhood_router/validation_probabilities.npz")
    test10 = load("artifacts/neighborhood_router/test_probabilities.npz")

    labels = validation_frame["groundtruth"].to_numpy(np.int64)
    folds = validation_frame["inner_fold"].to_numpy(np.int64)
    baseline_validation = validation_frame["predicted"].to_numpy(np.int64)
    baseline_test = consensus_test["predictions"].astype(np.int64)
    ids = consensus_test["ids"].astype(np.int64)
    if not np.array_equal(labels, validation29["labels"]) or not np.array_equal(ids, test29["ids"]):
        raise ValueError("Consensus and multiview rows differ")

    validation_features = features(
        validation29["view_logits"],
        [
            validation11["calibrated_probabilities"],
            validation15["soup_probabilities"],
            validation24["siglip2_probabilities"],
            validation10["router_probabilities"],
            validation10["graph_probabilities"],
            validation10["router_graph_probabilities"],
        ],
    )
    test_features = features(
        test29["view_logits"],
        [
            test11["calibrated_probabilities"],
            test15["soup_probabilities"],
            test24["siglip2_probabilities"],
            test10["router_probabilities"],
            test10["graph_probabilities"],
            test10["router_graph_probabilities"],
        ],
    )
    binary = labels != 1
    baseline_metrics = score(labels, baseline_validation)
    baseline_folds = fold_scores(labels, folds, baseline_validation)
    selected = None
    selected_oof = None
    for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        oof = np.zeros(len(labels), dtype=np.float64)
        for fold in sorted(np.unique(folds)):
            train = binary & (folds != fold)
            valid = folds == fold
            model = classifier(c)
            model.fit(validation_features[train], (labels[train] == 0).astype(np.int64))
            oof[valid] = model.predict_proba(validation_features[valid])[:, 1]
        for threshold in np.arange(0.80, 0.951, 0.01):
            predictions = baseline_validation.copy()
            gate = (predictions == 2) & (oof >= threshold)
            predictions[gate] = 0
            metrics = score(labels, predictions)
            current_folds = fold_scores(labels, folds, predictions)
            non_degrading = sum(new >= old - 1e-12 for new, old in zip(current_folds, baseline_folds))
            changed = int(gate.sum())
            if (
                metrics["macro_f1"] > baseline_metrics["macro_f1"] + 1e-12
                and non_degrading == len(baseline_folds)
                and changed <= 1
            ):
                row = {
                    "c": c,
                    "threshold": float(threshold),
                    "changed_validation_rows": changed,
                    "fold_macro_f1": current_folds,
                    "mean_fold_macro_f1": float(np.mean(current_folds)),
                    "non_degrading_folds": int(non_degrading),
                    **metrics,
                }
                key = (
                    row["macro_f1"],
                    row["mean_fold_macro_f1"],
                    row["threshold"],
                    -row["c"],
                )
                if selected is None or key > selected[0]:
                    selected = (key, row)
                    selected_oof = oof.copy()
    if selected is None:
        raise RuntimeError("No strict OOF router candidate improved validation")
    selected = selected[1]

    final_router = classifier(selected["c"])
    final_router.fit(validation_features[binary], (labels[binary] == 0).astype(np.int64))
    test_router_probability = final_router.predict_proba(test_features)[:, 1]
    final_consensus = float(consensus_metrics["selected"]["threshold"])

    def make_predictions(anchor, router_probability, siglip2, hard_predictions, hard_probabilities, consensus_threshold):
        predictions = anchor.copy()
        router_gate = (anchor == 2) & (router_probability >= selected["threshold"])
        consensus_gate = (
            (anchor == 2)
            & (siglip2 == 0)
            & (hard_predictions == 0)
            & (hard_probabilities[:, 0] >= consensus_threshold)
        )
        predictions[router_gate | consensus_gate] = 0
        return predictions, router_gate, consensus_gate

    final_validation, _, _ = make_predictions(
        baseline_validation,
        selected_oof,
        validation_frame["siglip2_predicted"].to_numpy(np.int64),
        validation_frame["boundary_specialist_predicted"].to_numpy(np.int64),
        np.column_stack(
            [
                validation_frame["boundary_specialist_p0"].to_numpy(np.float64),
                np.zeros(len(labels)),
                1.0 - validation_frame["boundary_specialist_p0"].to_numpy(np.float64),
            ]
        ),
        final_consensus,
    )
    final_test, test_router_gate, test_consensus_gate = make_predictions(
        baseline_test,
        test_router_probability,
        consensus_test["siglip2_predictions"].astype(np.int64),
        consensus_test["boundary_specialist_probabilities"].argmax(axis=1).astype(np.int64),
        consensus_test["boundary_specialist_probabilities"].astype(np.float64),
        final_consensus,
    )

    template = pd.read_csv(ROOT / "BDC2026/submission.csv")[["id"]]

    def write_submission(name, predictions):
        output = template.copy()
        output["predicted"] = output["id"].astype(int).map(dict(zip(ids, predictions.astype(int), strict=True)))
        if len(output) != 1_458 or output["predicted"].isna().any():
            raise ValueError("Invalid submission mapping")
        output["predicted"] = output["predicted"].astype(int)
        output.to_csv(OUTPUT / name, index=False)

    write_submission("submission_final_router.csv", final_test)
    np.savez_compressed(
        OUTPUT / "router_probabilities.npz",
        ids=ids,
        validation_labels=labels,
        validation_folds=folds,
        validation_router_probability=selected_oof.astype(np.float32),
        test_router_probability=test_router_probability.astype(np.float32),
        validation_final_predictions=final_validation.astype(np.int64),
        final_predictions=final_test,
    )
    report = {
        "method_version": 1,
        "method": "OOF-selected boundary consensus + strict OOF binary router",
        "test_labels_used": False,
        "seed": SEED,
        "baseline_validation": baseline_metrics,
        "selected_router": selected,
        "final_validation": score(labels, final_validation),
        "final_consensus_threshold": final_consensus,
        "changed_test_ids": ids[test_router_gate | test_consensus_gate].astype(int).tolist(),
        "runtime_seconds": __import__("time").perf_counter() - started,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Selected router:", selected)
    print("Validation Macro-F1:", baseline_metrics["macro_f1"], "->", report["final_validation"]["macro_f1"])
    print("Changed IDs:", report["changed_test_ids"])
    print("Wrote final decision artifacts to", OUTPUT.resolve())


if __name__ == "__main__":
    main()
