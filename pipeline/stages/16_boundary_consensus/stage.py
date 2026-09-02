"""Semantic consensus + SigLIP2 + hard-negative boundary guard.

Only boundary-specialist validation inference runs on Modal. No retraining or test labels.

  modal run stages/16_boundary_consensus/stage.py
"""

from pathlib import Path
import sys

import modal


SCRIPT = Path(__file__).resolve()
ROOT = next(
    (
        path
        for path in (*SCRIPT.parents, Path.cwd())
        if (path / "modal_backbone_app.py").is_file()
    ),
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
app = modal.App("bdc2026-boundary-consensus")

MODEL = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
REVISION = "e959efa74c867491dcfe3ec3e4f97382e39025b3"
BOUNDARY_MODEL_ROOT = f"{CACHE_ROOT}/runs/dinov3_convnext_large_full_finetune_224_seed2026"
CONSENSUS_CACHE_ROOT = f"{CACHE_ROOT}/runs/boundary_consensus_validation_seed2026"
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 8
THRESHOLDS = tuple(round(0.34 + index * 0.01, 2) for index in range(62))


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def extract_validation(force: bool = False):
    import io
    import os
    import random
    import time

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageFile, ImageOps
    from sklearn.model_selection import StratifiedGroupKFold
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModel

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

    output = Path(CONSENSUS_CACHE_ROOT) / "boundary_specialist_validation.npz"
    if output.exists() and not force:
        return {"probabilities": output.read_bytes(), "cached": True, "seconds": 0.0}

    checkpoint = Path(BOUNDARY_MODEL_ROOT) / "ablations" / "hard_negative" / "best_validation_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing boundary-specialist checkpoint in this Modal workspace: {checkpoint}"
        )

    manifest = pd.read_csv(Path(DATA_ROOT) / "train_manifest.csv")
    manifest["label"] = manifest["label"].astype(int)
    if len(manifest) != 26_527 or set(manifest["label"]) != {0, 1, 2}:
        raise ValueError("Invalid training manifest")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, valid_indices = next(
        splitter.split(manifest, manifest["label"], manifest["group"])
    )
    valid_frame = manifest.iloc[valid_indices].reset_index(drop=True)
    labels = valid_frame["label"].to_numpy(dtype=np.int64)

    processor = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION)

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
                ((IMAGE_SIZE - resized.width) // 2, (IMAGE_SIZE - resized.height) // 2),
            )
            return canvas

    transform = transforms.Compose(
        [
            ResizePad(),
            transforms.ToTensor(),
            transforms.Normalize(processor.image_mean, processor.image_std),
        ]
    )

    class Images(Dataset):
        def __len__(self):
            return len(valid_frame)

        def __getitem__(self, index):
            with Image.open(valid_frame.iloc[index]["path"]) as source:
                return transform(ImageOps.exif_transpose(source).convert("RGB"))

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(MODEL, revision=REVISION)
            self.classifier = nn.Linear(self.backbone.config.hidden_sizes[-1], 3)

        def forward(self, images):
            return self.classifier(
                self.backbone(pixel_values=images).pooler_output
            )

    model = Classifier().to("cuda")
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    model.eval()
    loader = DataLoader(
        Images(),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    batches = []
    with torch.inference_mode():
        for images in loader:
            images = images.to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = 0.5 * (
                    model(images) + model(torch.flip(images, dims=[3]))
                )
            batches.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probabilities = np.concatenate(batches).astype(np.float32)
    if probabilities.shape != (5_308, 3):
        raise ValueError(f"Unexpected validation shape: {probabilities.shape}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, labels=labels, probabilities=probabilities)
    cache_volume.commit()
    buffer = io.BytesIO()
    np.savez(buffer, labels=labels, probabilities=probabilities)
    return {
        "probabilities": buffer.getvalue(),
        "cached": False,
        "seconds": time.perf_counter() - started,
    }


def build_consensus(validation_bytes, output):
    import io
    import json
    import time

    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    started = time.perf_counter()
    semantic_root = ROOT / "artifacts" / "semantic_consensus"
    boundary_root = ROOT / "artifacts" / "boundary_specialist"
    validation_frame = pd.read_csv(semantic_root / "validation_predictions.csv")
    semantic_test = np.load(semantic_root / "test_probabilities.npz", allow_pickle=False)
    hard_validation = np.load(io.BytesIO(validation_bytes), allow_pickle=False)
    hard_test = np.load(boundary_root / "test_probabilities.npz", allow_pickle=False)

    labels = validation_frame["groundtruth"].to_numpy(dtype=np.int64)
    folds = validation_frame["inner_fold"].to_numpy(dtype=np.int64)
    baseline_validation = validation_frame["predicted"].to_numpy(dtype=np.int64)
    siglip2_validation = validation_frame["siglip2_predicted"].to_numpy(dtype=np.int64)
    ids = semantic_test["ids"].astype(np.int64)
    baseline_test = semantic_test["predictions"].astype(np.int64)
    siglip2_test = semantic_test["siglip2_predictions"].astype(np.int64)
    hard_validation_probabilities = hard_validation["probabilities"].astype(np.float64)
    hard_test_probabilities = hard_test["probabilities"].astype(np.float64)
    if not np.array_equal(labels, hard_validation["labels"]):
        raise ValueError("Semantic and boundary validation labels differ")
    if not np.array_equal(ids, hard_test["ids"]):
        raise ValueError("Semantic and boundary test IDs differ")

    def score(predictions):
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

    def fold_scores(predictions):
        return [
            float(
                f1_score(
                    labels[folds == fold],
                    predictions[folds == fold],
                    average="macro",
                )
            )
            for fold in sorted(np.unique(folds))
        ]

    def apply(anchor, siglip2, hard_probabilities, threshold):
        predictions = anchor.copy()
        gate = (
            (anchor == 2)
            & (siglip2 == 0)
            & (hard_probabilities.argmax(axis=1) == 0)
            & (hard_probabilities[:, 0] >= threshold)
        )
        predictions[gate] = 0
        return predictions, gate

    baseline = score(baseline_validation)
    baseline_folds = fold_scores(baseline_validation)
    candidates = []
    for threshold in THRESHOLDS:
        predictions, gate = apply(
            baseline_validation,
            siglip2_validation,
            hard_validation_probabilities,
            threshold,
        )
        metrics = score(predictions)
        current_folds = fold_scores(predictions)
        metrics.update(
            {
                "threshold": threshold,
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
        and abs(row["class_f1"][1] - baseline["class_f1"][1]) <= 1e-12
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["macro_f1"],
                row["mean_fold_macro_f1"],
                row["non_degrading_folds"],
                -row["changed_validation_rows"],
                row["threshold"],
            ),
        )
    else:
        selected = {
            **baseline,
            "threshold": 1.01,
            "changed_validation_rows": 0,
            "fold_macro_f1": baseline_folds,
            "mean_fold_macro_f1": float(np.mean(baseline_folds)),
            "non_degrading_folds": len(baseline_folds),
            "fallback_to_anchor": True,
        }
    selected_threshold = float(selected["threshold"])

    validation_predictions, validation_gate = apply(
        baseline_validation,
        siglip2_validation,
        hard_validation_probabilities,
        selected_threshold,
    )
    test_predictions, test_gate = apply(
        baseline_test,
        siglip2_test,
        hard_test_probabilities,
        selected_threshold,
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

    output.mkdir(parents=True, exist_ok=True)
    submission(baseline_test).to_csv(
        output / "submission_boundary_consensus_semantic_baseline.csv", index=False
    )
    submission(test_predictions).to_csv(
        output / "submission_boundary_consensus_guarded_consensus.csv", index=False
    )
    pd.DataFrame(
        {
            "groundtruth": labels,
            "inner_fold": folds,
            "semantic_anchor_predicted": baseline_validation,
            "siglip2_predicted": siglip2_validation,
            "boundary_specialist_predicted": hard_validation_probabilities.argmax(axis=1),
            "boundary_specialist_p0": hard_validation_probabilities[:, 0],
            "consensus_gate": validation_gate,
            "predicted": validation_predictions,
        }
    ).to_csv(output / "validation_predictions.csv", index=False)
    np.savez(
        output / "validation_probabilities.npz",
        labels=labels,
        inner_folds=folds,
        boundary_specialist_probabilities=hard_validation_probabilities.astype(np.float32),
        predictions=validation_predictions,
    )
    np.savez(
        output / "test_probabilities.npz",
        ids=ids,
        semantic_anchor_predictions=baseline_test,
        siglip2_predictions=siglip2_test,
        boundary_specialist_probabilities=hard_test_probabilities.astype(np.float32),
        consensus_gate=test_gate,
        predictions=test_predictions,
    )
    report = {
        "method_version": 1,
        "method": "semantic anchor + high-resolution context + hard-negative boundary consensus",
        "test_labels_used": False,
        "seed": SEED,
        "selection": "threshold selected only from official-train OOF metrics and fold guards",
        "baseline_validation": baseline,
        "selected": selected,
        "selected_validation": score(validation_predictions),
        "changed_test_rows": int(test_gate.sum()),
        "changed_test_ids": ids[test_gate].astype(int).tolist(),
        "top_candidates": sorted(
            candidates,
            key=lambda row: (
                row["macro_f1"],
                row["mean_fold_macro_f1"],
                row["non_degrading_folds"],
            ),
            reverse=True,
        )[:15],
        "local_postprocessing_seconds": time.perf_counter() - started,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


@app.local_entrypoint()
def main(
    output_dir: str = "artifacts/boundary_consensus",
    force: bool = False,
):
    result = extract_validation.remote(force=force)
    report = build_consensus(result["probabilities"], Path(output_dir).resolve())
    print("Validation inference seconds:", result["seconds"])
    print("Selected threshold:", report["selected"]["threshold"])
    print(
        "Validation Macro-F1:",
        report["baseline_validation"]["macro_f1"],
        "->",
        report["selected_validation"]["macro_f1"],
    )
    print("Changed test IDs:", report["changed_test_ids"])
    print("Wrote artifacts to", Path(output_dir).resolve())
