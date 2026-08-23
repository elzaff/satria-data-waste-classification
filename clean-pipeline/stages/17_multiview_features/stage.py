"""Inference-only multi-crop boundary features for final consensus.

Run from project root through ``pipeline.py``.

  modal run stages/17_multiview_features/stage.py

No training, external data, or test labels are used.
"""

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
app = modal.App("bdc2026-multiview-features")

MODEL = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
REVISION = "e959efa74c867491dcfe3ec3e4f97382e39025b3"
RUN04_ROOT = f"{CACHE_ROOT}/runs/dinov3_convnext_large_full_finetune_224_seed2026"
RUN29_ROOT = f"{CACHE_ROOT}/runs/multiview_features_seed2026"
IMAGE_SIZE = 224
RESIZE_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 8
METHOD_VERSION = 1


def choose_gate(labels, folds, baseline, siglip2, full_predictions, view_logits):
    import numpy as np
    from sklearn.metrics import f1_score

    labels = np.asarray(labels)
    folds = np.asarray(folds)
    baseline = np.asarray(baseline)
    fold_values = sorted(np.unique(folds))

    def macro(predictions, mask=None):
        if mask is None:
            return float(f1_score(labels, predictions, average="macro"))
        return float(f1_score(labels[mask], predictions[mask], average="macro"))

    baseline_score = macro(baseline)
    baseline_folds = [macro(baseline, folds == fold) for fold in fold_values]
    mode_rank = {"full_and_multicrop": 2, "siglip2_and_multicrop": 1, "multicrop_only": 0}
    best = None
    for full_weight in (0.25, 0.40, 0.55, 0.70, 0.85):
        for center_weight in (0.25, 0.50, 0.75):
            crop_logits = center_weight * view_logits[:, 1] + (1.0 - center_weight) * view_logits[:, 2:].mean(axis=1)
            pooled = full_weight * view_logits[:, 0] + (1.0 - full_weight) * crop_logits
            for temperature in (0.7, 0.85, 1.0, 1.15, 1.3):
                scaled = pooled / temperature
                probabilities = np.exp(scaled - scaled.max(axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                multicrop_predictions = probabilities.argmax(axis=1)
                for mode in mode_rank:
                    agreement = multicrop_predictions == 0
                    if mode == "full_and_multicrop":
                        agreement &= full_predictions == 0
                    elif mode == "siglip2_and_multicrop":
                        agreement &= siglip2 == 0
                    for confidence in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975):
                        predictions = baseline.copy()
                        gate = (baseline == 2) & agreement & (probabilities[:, 0] >= confidence)
                        predictions[gate] = 0
                        score = macro(predictions)
                        fold_scores = [macro(predictions, folds == fold) for fold in fold_values]
                        non_degrading = sum(new >= old - 1e-12 for new, old in zip(fold_scores, baseline_folds))
                        mean_fold = float(np.mean(fold_scores))
                        if score > baseline_score + 1e-12 and mean_fold >= np.mean(baseline_folds) - 1e-12 and non_degrading >= 4:
                            row = {
                                "mode": mode,
                                "full_weight": full_weight,
                                "center_weight": center_weight,
                                "temperature": temperature,
                                "confidence_min": confidence,
                                "macro_f1": score,
                                "fold_macro_f1": fold_scores,
                                "mean_fold_macro_f1": mean_fold,
                                "non_degrading_folds": int(non_degrading),
                                "changed_validation_rows": int(gate.sum()),
                            }
                            key = (score, mean_fold, non_degrading, -int(gate.sum()), mode_rank[mode], confidence)
                            if best is None or key > best[0]:
                                best = (key, row, predictions, probabilities, gate)
    if best is None:
        return None, baseline.copy(), np.zeros((len(labels), 3)), np.zeros(len(labels), dtype=bool)
    return best[1], best[2], best[3], best[4]


def apply_gate(anchor, siglip2, full_predictions, view_logits, selected):
    import numpy as np

    if selected is None:
        return anchor.copy(), np.zeros(len(anchor), dtype=bool), np.zeros((len(anchor), 3))
    crop_logits = selected["center_weight"] * view_logits[:, 1] + (1.0 - selected["center_weight"]) * view_logits[:, 2:].mean(axis=1)
    pooled = selected["full_weight"] * view_logits[:, 0] + (1.0 - selected["full_weight"]) * crop_logits
    scaled = pooled / selected["temperature"]
    probabilities = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    agreement = probabilities.argmax(axis=1) == 0
    if selected["mode"] == "full_and_multicrop":
        agreement &= full_predictions == 0
    elif selected["mode"] == "siglip2_and_multicrop":
        agreement &= siglip2 == 0
    gate = (anchor == 2) & agreement & (probabilities[:, 0] >= selected["confidence_min"])
    predictions = anchor.copy()
    predictions[gate] = 0
    return predictions, gate, probabilities


def self_check():
    import numpy as np

    labels = np.tile([0, 2], 5)
    folds = np.repeat(np.arange(5), 2)
    baseline = np.tile([2, 2], 5)
    full = np.tile([0, 2], 5)
    logits = np.tile(np.asarray([[[5.0, 0.0, 0.0]] * 6, [[0.0, 0.0, 5.0]] * 6]), (5, 1, 1))
    selected, predictions, _, _ = choose_gate(labels, folds, baseline, full, full, logits)
    assert selected is not None and np.array_equal(predictions, labels)


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def infer_multicrop(force: bool = False):
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
    from torchvision.transforms import functional as vision_functional
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

    output_root = Path(RUN29_ROOT)
    validation_path = output_root / "validation_view_logits.npz"
    test_path = output_root / "test_view_logits.npz"
    if validation_path.exists() and test_path.exists() and not force:
        return {
            "validation": validation_path.read_bytes(),
            "test": test_path.read_bytes(),
            "cached": True,
            "seconds": 0.0,
        }

    checkpoint = Path(RUN04_ROOT) / "ablations" / "hard_negative" / "best_validation_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing Run04 checkpoint in active Modal workspace: {checkpoint}")
    manifest_path = Path(DATA_ROOT) / "train_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Missing train_manifest.csv; prepare it on CPU first")
    manifest = pd.read_csv(manifest_path)
    manifest["label"] = manifest["label"].astype(int)
    if len(manifest) != 26_527 or set(manifest["label"]) != {0, 1, 2}:
        raise ValueError("Invalid training manifest")
    _, valid_indices = next(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            manifest, manifest["label"], manifest["group"]
        )
    )
    valid_frame = manifest.iloc[valid_indices].reset_index(drop=True)
    valid_labels = valid_frame["label"].to_numpy(np.int64)
    test_files = sorted(
        (path for path in (Path(DATA_ROOT) / "test").iterdir() if path.is_file()),
        key=lambda path: int(path.stem),
    )
    if len(valid_frame) != 5_308 or len(test_files) != 1_458:
        raise ValueError("Unexpected validation/test size")
    test_frame = pd.DataFrame({"path": [str(path) for path in test_files]})
    test_ids = np.asarray([int(path.stem) for path in test_files], dtype=np.int64)

    processor = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION)
    normalize = transforms.Normalize(processor.image_mean, processor.image_std)
    to_tensor = transforms.Compose([transforms.ToTensor(), normalize])

    def full_view(source):
        resized = ImageOps.contain(source, (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (124, 116, 104))
        canvas.paste(resized, ((IMAGE_SIZE - resized.width) // 2, (IMAGE_SIZE - resized.height) // 2))
        return canvas

    def crop_views(source):
        resized = vision_functional.resize(
            source,
            RESIZE_SIZE,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )
        width, height = resized.size
        right, bottom = width - IMAGE_SIZE, height - IMAGE_SIZE
        center = (bottom // 2, right // 2)
        positions = (center, (0, 0), (0, right), (bottom, 0), (bottom, right))
        return [vision_functional.crop(resized, top, left, IMAGE_SIZE, IMAGE_SIZE) for top, left in positions]

    class Views(Dataset):
        def __init__(self, frame):
            self.paths = frame["path"].astype(str).tolist()

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            with Image.open(self.paths[index]) as source:
                image_value = ImageOps.exif_transpose(source).convert("RGB")
                values = [full_view(image_value), *crop_views(image_value)]
            return torch.stack([to_tensor(value) for value in values])

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(MODEL, revision=REVISION)
            self.classifier = nn.Linear(self.backbone.config.hidden_sizes[-1], 3)

        def forward(self, images):
            return self.classifier(self.backbone(pixel_values=images).pooler_output)

    model = Classifier().to("cuda")
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    model.eval()

    @torch.inference_mode()
    def predict(frame):
        loader = DataLoader(
            Views(frame),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )
        batches = []
        for views in loader:
            batch, count, channels, height, width = views.shape
            flat = views.reshape(batch * count, channels, height, width).to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = 0.5 * (model(flat) + model(torch.flip(flat, dims=[3])))
            batches.append(logits.float().cpu().numpy().reshape(batch, count, 3))
        return np.concatenate(batches).astype(np.float32)

    validation_logits = predict(valid_frame)
    test_logits = predict(test_frame)
    if validation_logits.shape != (5_308, 6, 3) or test_logits.shape != (1_458, 6, 3):
        raise ValueError("Unexpected multi-crop logits shape")
    output_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(validation_path, labels=valid_labels, logits=validation_logits)
    np.savez_compressed(test_path, ids=test_ids, logits=test_logits)
    cache_volume.commit()
    validation_buffer, test_buffer = io.BytesIO(), io.BytesIO()
    np.savez_compressed(validation_buffer, labels=valid_labels, logits=validation_logits)
    np.savez_compressed(test_buffer, ids=test_ids, logits=test_logits)
    return {
        "validation": validation_buffer.getvalue(),
        "test": test_buffer.getvalue(),
        "cached": False,
        "seconds": time.perf_counter() - started,
    }


def build_outputs(validation_bytes, test_bytes, output):
    import io
    import json
    import time

    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    started = time.perf_counter()
    consensus_root = ROOT / "artifacts" / "boundary_consensus"
    validation_frame = pd.read_csv(consensus_root / "validation_predictions.csv")
    consensus_test = np.load(consensus_root / "test_probabilities.npz", allow_pickle=False)
    validation_views = np.load(io.BytesIO(validation_bytes), allow_pickle=False)
    test_views = np.load(io.BytesIO(test_bytes), allow_pickle=False)
    labels = validation_frame["groundtruth"].to_numpy(np.int64)
    folds = validation_frame["inner_fold"].to_numpy(np.int64)
    baseline_validation = validation_frame["predicted"].to_numpy(np.int64)
    siglip2_validation = validation_frame["siglip2_predicted"].to_numpy(np.int64)
    full_validation = validation_frame["boundary_specialist_predicted"].to_numpy(np.int64)
    ids = consensus_test["ids"].astype(np.int64)
    baseline_test = consensus_test["predictions"].astype(np.int64)
    siglip2_test = consensus_test["siglip2_predictions"].astype(np.int64)
    full_test = consensus_test["boundary_specialist_probabilities"].argmax(axis=1).astype(np.int64)
    validation_logits = validation_views["logits"].astype(np.float64)
    test_logits = test_views["logits"].astype(np.float64)
    if not np.array_equal(labels, validation_views["labels"]) or not np.array_equal(ids, test_views["ids"]):
        raise ValueError("Consensus and multiview rows differ")

    selected, validation_predictions, validation_probabilities, validation_gate = choose_gate(
        labels, folds, baseline_validation, siglip2_validation, full_validation, validation_logits
    )
    test_predictions, test_gate, test_probabilities = apply_gate(
        baseline_test, siglip2_test, full_test, test_logits, selected
    )

    def score(predictions):
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "class_f1": f1_score(labels, predictions, labels=[0, 1, 2], average=None).tolist(),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
            "errors": int((labels != predictions).sum()),
        }

    template = pd.read_csv(ROOT / "BDC2026" / "submission.csv")[["id"]]
    template["predicted"] = template["id"].astype(int).map(dict(zip(ids, test_predictions.astype(int), strict=True)))
    if len(template) != 1_458 or template["predicted"].isna().any():
        raise ValueError("Invalid submission mapping")
    template["predicted"] = template["predicted"].astype(int)

    output.mkdir(parents=True, exist_ok=True)
    template.to_csv(output / "submission_multiview.csv", index=False)
    pd.DataFrame(
        {
            "groundtruth": labels,
            "inner_fold": folds,
            "consensus_anchor_predicted": baseline_validation,
            "multicrop_p0": validation_probabilities[:, 0],
            "gate": validation_gate,
            "predicted": validation_predictions,
        }
    ).to_csv(output / "validation_predictions.csv", index=False)
    np.savez_compressed(
        output / "validation_probabilities.npz",
        labels=labels,
        inner_folds=folds,
        view_logits=validation_logits.astype(np.float32),
        probabilities=validation_probabilities.astype(np.float32),
        predictions=validation_predictions,
    )
    np.savez_compressed(
        output / "test_probabilities.npz",
        ids=ids,
        view_logits=test_logits.astype(np.float32),
        probabilities=test_probabilities.astype(np.float32),
        predictions=test_predictions,
    )
    report = {
        "method_version": METHOD_VERSION,
        "method": "consensus anchor + inference-only six-view boundary specialist",
        "test_labels_used": False,
        "seed": SEED,
        "selection": "validation aggregate Macro-F1 gain, mean-fold non-loss, >=4/5 non-degrading folds",
        "baseline_validation": score(baseline_validation),
        "selected": selected,
        "selected_validation": score(validation_predictions),
        "changed_test_rows": int(test_gate.sum()),
        "changed_test_ids": ids[test_gate].astype(int).tolist(),
        "local_postprocessing_seconds": time.perf_counter() - started,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@app.local_entrypoint()
def main(output_dir: str = "artifacts/multiview_features", force: bool = False):
    self_check()
    result = infer_multicrop.remote(force=force)
    report = build_outputs(result["validation"], result["test"], Path(output_dir).resolve())
    print("Inference seconds:", result["seconds"])
    print("Selected:", report["selected"])
    print("Validation Macro-F1:", report["baseline_validation"]["macro_f1"], "->", report["selected_validation"]["macro_f1"])
    print("Changed test IDs:", report["changed_test_ids"])
    print("Wrote multiview artifacts to", Path(output_dir).resolve())
