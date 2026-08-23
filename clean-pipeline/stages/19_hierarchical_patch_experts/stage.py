"""Train full-fold hierarchical and Patch-MIL experts from public weights."""

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

from modal_backbone_app import cache_volume, data_volume, hf_secret, image  # noqa: E402


RUNTIME = ROOT / "expert_runtime.py"
image = (
    image.pip_install("accelerate==1.10.1", "timm==1.0.20")
    .env({"HF_HUB_DOWNLOAD_TIMEOUT": "120", "HF_HUB_ETAG_TIMEOUT": "60"})
    .add_local_file(RUNTIME, "/root/expert_runtime.py")
    .add_local_file(ROOT / "modal_backbone_app.py", "/root/modal_backbone_app.py")
)
app = modal.App("bdc2026-clean-hierarchical-patch-experts")


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=8,
    memory=49152,
    timeout=24 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def train_experts(force: bool = False):
    from pathlib import Path

    import expert_runtime as runtime

    runtime.CACHE_ROOT = "/cache/final01_clean_experts_v1"
    outputs = {}
    for architecture in ("hierarchical_siglip2", "hierarchical_patch_mil_siglip2"):
        result = runtime.run_architecture(architecture, full=True, force=force)
        root = Path(result["run_root"])
        required = {
            "metrics": root / "metrics.json",
            "oof": root / "oof_probabilities.npz",
            "test": root / "test_probabilities.npz",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Incomplete clean expert artifacts: " + ", ".join(missing))
        outputs[architecture] = {
            "run_root": str(root),
            "metrics": required["metrics"].read_text(encoding="utf-8"),
            "oof": required["oof"].read_bytes(),
            "test": required["test"].read_bytes(),
        }
    cache_volume.commit()
    return outputs


@app.local_entrypoint()
def main(
    output_dir: str = "artifacts/hierarchical_patch_experts",
    force: bool = False,
):
    result = train_experts.remote(force=force)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = {
        "hierarchical_siglip2": "hierarchical",
        "hierarchical_patch_mil_siglip2": "patch_mil",
    }
    for architecture, prefix in names.items():
        values = result[architecture]
        (output / f"{prefix}_metrics.json").write_text(values["metrics"], encoding="utf-8")
        (output / f"{prefix}_oof_probabilities.npz").write_bytes(values["oof"])
        (output / f"{prefix}_test_probabilities.npz").write_bytes(values["test"])
        print(f"{architecture}: {values['run_root']}")
    print("Wrote expert artifacts to", output)
