"""Rebuild exact FINAL/01 decision layer from saved probability artifacts.

This script does not read test labels and does not run vision backbones. It
re-fits the deterministic five-fold logistic router from official validation
labels, applies guarded 2->0 consensus, and writes a competition-format CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd


SCRIPT = Path(__file__).resolve()
DEFAULT_ARTIFACT_ROOT = SCRIPT.parent / "workspace" / "rebuild"
DEFAULT_OUTPUT = SCRIPT.parent / "outputs" / "artifact_inference"

REQUIRED_ARTIFACTS = (
    "BDC2026/submission.csv",
    "artifacts/neighborhood_router/validation_probabilities.npz",
    "artifacts/neighborhood_router/test_probabilities.npz",
    "artifacts/class_calibration/validation_probabilities.npz",
    "artifacts/class_calibration/test_probabilities.npz",
    "artifacts/probability_ensemble/validation_probabilities.npz",
    "artifacts/probability_ensemble/test_probabilities.npz",
    "artifacts/high_resolution_context/validation_probabilities.npz",
    "artifacts/high_resolution_context/test_probabilities.npz",
    "artifacts/boundary_consensus/validation_predictions.csv",
    "artifacts/boundary_consensus/test_probabilities.npz",
    "artifacts/boundary_consensus/metrics.json",
    "artifacts/multiview_features/validation_probabilities.npz",
    "artifacts/multiview_features/test_probabilities.npz",
    "artifacts/hierarchical_patch_experts/hierarchical_metrics.json",
    "artifacts/hierarchical_patch_experts/hierarchical_oof_probabilities.npz",
    "artifacts/hierarchical_patch_experts/hierarchical_test_probabilities.npz",
    "artifacts/hierarchical_patch_experts/patch_mil_oof_probabilities.npz",
    "artifacts/hierarchical_patch_experts/patch_mil_test_probabilities.npz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(artifact_root: Path) -> dict[str, str]:
    missing = [relative for relative in REQUIRED_ARTIFACTS if not (artifact_root / relative).is_file()]
    if missing:
        raise FileNotFoundError("Missing inference artifacts:\n- " + "\n- ".join(missing))

    consensus_metrics = json.loads(
        (artifact_root / "artifacts/boundary_consensus/metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if consensus_metrics.get("test_labels_used") is not False:
        raise ValueError("Consensus artifact must state test_labels_used=false")

    validation = pd.read_csv(
        artifact_root / "artifacts/boundary_consensus/validation_predictions.csv"
    )
    required_columns = {
        "groundtruth",
        "inner_fold",
        "predicted",
        "siglip2_predicted",
        "boundary_specialist_predicted",
        "boundary_specialist_p0",
    }
    missing_columns = sorted(required_columns.difference(validation.columns))
    if missing_columns:
        raise ValueError(f"Consensus validation artifact missing columns: {missing_columns}")
    if len(validation) != 5_308 or sorted(validation["inner_fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("Unexpected validation population or fold assignment")

    return {relative: sha256(artifact_root / relative) for relative in REQUIRED_ARTIFACTS}


def load_decision_router_module():
    module_path = SCRIPT.parent / "stages" / "18_oof_decision_router" / "stage.py"
    spec = importlib.util.spec_from_file_location("final01_decision_router", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ["id", "predicted"]:
        raise ValueError(f"Invalid submission columns: {frame.columns.tolist()}")
    if len(frame) != 1_458 or frame["id"].duplicated().any():
        raise ValueError("Submission must contain 1,458 unique IDs")
    if not set(frame["predicted"].astype(int)).issubset({0, 1, 2}):
        raise ValueError("Submission contains an invalid class")
    return frame.astype({"id": int, "predicted": int})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional trusted submission used only for exact-output regression testing.",
    )
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = validate_inputs(artifact_root)

    decision_router = load_decision_router_module()
    decision_router.ROOT = artifact_root
    decision_router.OUTPUT = output_dir
    decision_router.main()

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT.parent / "stages/20_hierarchical_patch_consensus/stage.py"),
            "--root",
            str(artifact_root),
            "--router-root",
            str(output_dir),
            "--expert-root",
            str(artifact_root / "artifacts/hierarchical_patch_experts"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    generated = output_dir / "submission_final_expert_consensus.csv"
    final_path = output_dir / "submission_final_artifact_inference.csv"
    shutil.copy2(generated, final_path)
    final_frame = validate_submission(final_path)

    reference_match = None
    if args.reference is not None:
        if not args.reference.is_file():
            raise FileNotFoundError(f"Reference submission not found: {args.reference}")
        reference = validate_submission(args.reference)
        reference_match = final_frame.equals(reference)
        if not reference_match:
            changed = int((final_frame["predicted"] != reference["predicted"]).sum())
            raise RuntimeError(f"Artifact inference differs from reference on {changed} rows")

    final_metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = {
        "method": "FINAL/01 artifact-based decision-layer inference",
        "artifact_root": str(artifact_root),
        "test_labels_used": False,
        "seed": 2026,
        "selected_expert_consensus": final_metrics["selected"],
        "validation_macro_f1": final_metrics["final_validation"]["macro_f1"],
        "submission_rows": len(final_frame),
        "submission_sha256": sha256(final_path),
        "reference_exact_match": reference_match,
        "input_artifact_sha256": input_hashes,
    }
    (output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Artifact root:", artifact_root)
    print("Selected expert consensus:", summary["selected_expert_consensus"])
    print("Validation Macro-F1:", summary["validation_macro_f1"])
    print("Reference exact match:", reference_match)
    print("Submission:", final_path)


if __name__ == "__main__":
    main()
