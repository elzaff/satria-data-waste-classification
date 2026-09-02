"""Clean, standalone training and inference entrypoint.

Semantic stage modules live beside this orchestrator. Pipeline starts from
public pretrained weights and official data. It reads no historical experiment
scripts, checkpoints, probabilities, test labels, or submissions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import time


SCRIPT = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT.parent
MANIFEST_PATH = PACKAGE_ROOT / "pipeline_manifest.json"
DEFAULT_WORKSPACE = PACKAGE_ROOT / "workspace" / "rebuild"
DEFAULT_TEMPLATE = PACKAGE_ROOT / "submission_template.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("test_labels_used") is not False:
        raise ValueError("Manifest must state test_labels_used=false")
    stages = manifest.get("stages", [])
    names = [stage["name"] for stage in stages]
    if not stages or len(set(names)) != len(names):
        raise ValueError("Pipeline stages must be non-empty and unique")
    components = manifest.get("architecture_components", [])
    covered = [name for component in components for name in component["stages"]]
    if len(set(covered)) != len(covered) or set(covered) != set(names):
        raise ValueError("Architecture components must cover every stage exactly once")
    return manifest


def source_entries(manifest: dict):
    for relative, expected_hash in manifest["sources"].items():
        yield PACKAGE_ROOT / relative, Path(relative), expected_hash
    for item in manifest.get("shared_sources", []):
        yield (
            (PACKAGE_ROOT / item["source"]).resolve(),
            Path(item["destination"]),
            item["sha256"],
        )


def source_hashes(manifest: dict) -> dict[str, str]:
    return {
        destination.as_posix(): expected_hash
        for _, destination, expected_hash in source_entries(manifest)
    }


def validate_sources(manifest: dict) -> None:
    forbidden = ("gt-final.csv", "inshallah_groundtruth.csv")
    for path, destination, expected_hash in source_entries(manifest):
        if not path.is_file():
            raise FileNotFoundError(f"Missing clean source: {path}")
        if sha256(path) != expected_hash:
            raise RuntimeError(f"Clean source hash mismatch: {path}")
        source = path.read_text(encoding="utf-8")
        if any(name in source.lower() for name in forbidden):
            raise RuntimeError(f"Forbidden test-label dependency: {path}")
        compile(source, str(destination), "exec")


def copy_sources(workspace: Path, manifest: dict) -> None:
    for source, relative, expected_hash in source_entries(manifest):
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != expected_hash:
            raise RuntimeError(f"Copied source mismatch: {relative}")


def prepare_workspace(
    workspace: Path,
    profile: str,
    template: Path,
    manifest: dict,
    resume: bool,
) -> dict:
    state_path = workspace / "pipeline_state.json"
    if state_path.is_file():
        if not resume:
            raise RuntimeError(f"Workspace already has run. Add --resume: {workspace}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("profile") != profile:
            raise RuntimeError("Cannot resume with different Modal profile")
        if state.get("source_sha256") != source_hashes(manifest):
            raise RuntimeError("Cannot resume after clean source changed")
        return state

    if workspace.exists() and any(workspace.iterdir()):
        raise RuntimeError(f"Workspace not empty: {workspace}")
    if not template.is_file():
        raise FileNotFoundError(f"Missing official submission template: {template}")

    workspace.mkdir(parents=True, exist_ok=True)
    copy_sources(workspace, manifest)
    destination = workspace / "BDC2026" / "submission.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template, destination)
    (workspace / "logs").mkdir(exist_ok=True)

    state = {
        "method": manifest["name"],
        "version": manifest["version"],
        "seed": manifest["seed"],
        "profile": profile,
        "created_at_unix": time(),
        "source_sha256": source_hashes(manifest),
        "official_submission_sha256": sha256(destination),
        "test_labels_used": False,
        "stages": {},
    }
    save_json(state_path, state)
    return state


def environment(profile: str) -> dict[str, str]:
    values = os.environ.copy()
    values["MODAL_PROFILE"] = profile
    values["PYTHONIOENCODING"] = "utf-8"
    return values


def expand_command(command: list[str]) -> list[str]:
    return [sys.executable if value == "{python}" else value for value in command]


def run_command(
    name: str,
    command: list[str],
    workspace: Path,
    profile: str,
) -> None:
    command = expand_command(command)
    log_path = workspace / "logs" / f"{name}.log"
    print(f"\n[{name}] {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=environment(profile),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_stage(
    stage: dict,
    workspace: Path,
    profile: str,
    state: dict,
) -> None:
    state_path = workspace / "pipeline_state.json"
    name = stage["name"]
    expected = stage["expected"]
    previous = state["stages"].get(name, {})
    if previous.get("status") == "complete":
        if expected is None or (workspace / expected).is_file():
            print(f"[{name}] complete; skipped")
            return

    state["stages"][name] = {"status": "running", "started_at_unix": time()}
    save_json(state_path, state)
    try:
        run_command(name, stage["command"], workspace, profile)
        if expected is not None and not (workspace / expected).is_file():
            raise FileNotFoundError(f"Stage did not create: {expected}")
    except Exception as error:
        state["stages"][name] = {
            "status": "failed",
            "finished_at_unix": time(),
            "error": repr(error),
        }
        save_json(state_path, state)
        raise
    state["stages"][name] = {"status": "complete", "finished_at_unix": time()}
    save_json(state_path, state)


def validate_submission(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    if reader.fieldnames != ["id", "predicted"] or len(rows) != 1_458:
        raise ValueError(f"Invalid submission format: {path}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Submission IDs are not unique")
    if any(row["predicted"] not in {"0", "1", "2"} for row in rows):
        raise ValueError(f"Invalid predicted label: {path}")


def download_models(workspace: Path, profile: str, manifest: dict) -> None:
    for model in manifest["model_files"]:
        destination = workspace / "final_models" / model["name"] / "final_model.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"[download_{model['name']}] present; skipped")
            continue
        run_command(
            f"download_{model['name']}",
            [
                "modal", "volume", "get", "--force", "bdc2026-model-cache",
                model["remote"], str(destination),
            ],
            workspace,
            profile,
        )
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"Downloaded empty model: {destination}")


def finalize(
    workspace: Path,
    profile: str,
    manifest: dict,
    download_models_enabled: bool,
) -> Path:
    state_path = workspace / "pipeline_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing pipeline state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    incomplete = [
        stage["name"]
        for stage in manifest["stages"]
        if state["stages"].get(stage["name"], {}).get("status") != "complete"
    ]
    if incomplete:
        raise RuntimeError("Incomplete stages: " + ", ".join(incomplete))

    source = workspace / manifest["stages"][-1]["expected"]
    final = workspace / "submission_final01.csv"
    shutil.copy2(source, final)
    validate_submission(final)
    if download_models_enabled:
        download_models(workspace, profile, manifest)
    save_json(
        workspace / "rebuild_summary.json",
        {
            "profile": profile,
            "submission": str(final),
            "submission_sha256": sha256(final),
            "models_downloaded": download_models_enabled,
            "lightweight_artifacts_only": not download_models_enabled,
            "test_labels_used": False,
            "completed_at_unix": time(),
        },
    )
    print(f"\nPipeline complete: {final}")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--submission-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Opt in to downloading final checkpoint files from Modal Volume.",
    )
    parser.add_argument("--stage")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    validate_sources(manifest)
    stages = manifest["stages"]
    names = [stage["name"] for stage in stages]
    if args.stage and args.stage not in names:
        parser.error(f"unknown stage: {args.stage}")

    if args.dry_run:
        print(f"Readable source modules: {len(source_hashes(manifest))}")
        print("Historical experiment files read: 0")
        print("Test labels used: false")
        print(f"Profile: {args.profile}")
        print(f"Workspace: {args.workspace.resolve()}")
        for component in manifest["architecture_components"]:
            print(f"Component {component['name']}: {component['purpose']}")
        for stage in stages:
            command = " ".join(expand_command(stage["command"]))
            print(f"{stage['name']}: {command} -> {stage['expected'] or 'Modal manifest'}")
        return

    if shutil.which("modal") is None:
        raise FileNotFoundError("Modal CLI is not installed")
    workspace = args.workspace.resolve()
    state = prepare_workspace(
        workspace,
        args.profile,
        args.submission_template.resolve(),
        manifest,
        args.resume,
    )

    if args.stage:
        index = names.index(args.stage)
        incomplete = [
            stage["name"] for stage in stages[:index]
            if state["stages"].get(stage["name"], {}).get("status") != "complete"
        ]
        if incomplete:
            raise RuntimeError("Earlier stages incomplete: " + ", ".join(incomplete))
        run_stage(stages[index], workspace, args.profile, state)
    else:
        for stage in stages:
            run_stage(stage, workspace, args.profile, state)

    if args.finalize or not args.stage:
        finalize(workspace, args.profile, manifest, args.download_models)


if __name__ == "__main__":
    main()
