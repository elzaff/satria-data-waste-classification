"""Shared Modal app for frozen visual backbones and linear probes."""

from pathlib import Path

import modal


APP_NAME = "bdc2026-frozen-backbones"
GPU = "L40S"
DATA_ROOT = "/data/BDC2026"
CACHE_ROOT = "/cache"
HF_CACHE = f"{CACHE_ROOT}/huggingface"

BACKBONES = {
    "dinov3": {
        "kind": "dinov3",
        "model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
        "run": "dinov3_vitl16_frozen_linear_256_seed2026",
        "image_size": 256,
        "batch_size": 64,
        "output": "submission_dinov3_linear.csv",
    },
    "convnextv2": {
        "kind": "convnextv2",
        "model": "facebook/convnextv2-large-22k-224",
        "revision": "e58a79c331e6c9acd20e3ba2de0e934c546f0eea",
        "run": "convnextv2_large_frozen_linear_224_seed2026",
        "image_size": 224,
        "batch_size": 128,
        "output": "submission_convnextv2_linear.csv",
    },
}

SEED = 2026
NUM_CLASSES = 3
NUM_FOLDS = 5
NUM_WORKERS = 8
LINEAR_C = 1.0

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers==4.56.2",
        "huggingface-hub==0.34.4",
        "safetensors==0.6.2",
        "scikit-learn==1.7.2",
        "pandas==2.3.2",
        "numpy==2.2.6",
        "pillow==11.3.0",
    )
    .env(
        {
            "HF_HOME": HF_CACHE,
            "TOKENIZERS_PARALLELISM": "false",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": str(SEED),
        }
    )
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("bdc2026-data", create_if_missing=True)
cache_volume = modal.Volume.from_name("bdc2026-model-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=60 * 60,
    volumes={"/data": data_volume},
)
def prepare_manifest():
    import hashlib

    import pandas as pd

    data_root = Path(DATA_ROOT)
    manifest_path = data_root / "train_manifest.csv"
    frame = None
    if manifest_path.exists():
        try:
            candidate = pd.read_csv(manifest_path)
            if (
                len(candidate) == 26_527
                and set(candidate.columns) == {"path", "label", "group"}
                and set(candidate["label"].astype(int)) == {0, 1, 2}
            ):
                frame = candidate
        except (OSError, ValueError):
            pass
    if frame is None:
        rows = []
        folders = {
            0: data_root / "train" / "0_Recyclable",
            1: data_root / "train" / "1_Electronic",
            2: data_root / "train" / "2_Organic",
        }
        for label, folder in folders.items():
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing dataset folder: {folder}")
            for path in sorted(p for p in folder.iterdir() if p.is_file()):
                rows.append(
                    {
                        "path": str(path),
                        "label": label,
                        "group": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        frame = pd.DataFrame(rows)
        frame.to_csv(manifest_path, index=False)
        data_volume.commit()

    required = {"path", "label", "group"}
    if (
        len(frame) != 26_527
        or set(frame.columns) != required
        or set(frame["label"].astype(int)) != {0, 1, 2}
    ):
        raise ValueError("Invalid training manifest")
    result = {
        "rows": len(frame),
        "duplicate_extras": int(frame.duplicated("group").sum()),
    }
    print(result, flush=True)
    return result


@app.function(
    image=image,
    gpu=GPU,
    timeout=3 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def extract_features(backbone: str, split: str, force: bool = False):
    import os
    import random

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as functional
    from PIL import Image, ImageFile, ImageOps
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModel

    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone: {backbone}")
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    config = BACKBONES[backbone]
    run_root = Path(CACHE_ROOT) / "runs" / config["run"]
    output_path = run_root / f"{split}_features.npz"
    if output_path.exists() and not force:
        saved = np.load(output_path, allow_pickle=False)
        result = {"split": split, "rows": int(len(saved["features"])), "cached": True}
        print(result, flush=True)
        return result

    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    device = torch.device("cuda")
    processor = AutoImageProcessor.from_pretrained(
        config["model"], revision=config["revision"]
    )
    model = AutoModel.from_pretrained(
        config["model"], revision=config["revision"]
    ).eval().to(device)
    mean, std = processor.image_mean, processor.image_std
    if config["kind"] == "dinov3":
        register_tokens = getattr(model.config, "num_register_tokens", 4)
    image_size = config["image_size"]

    class ResizePad:
        def __call__(self, source):
            resized = ImageOps.contain(
                source,
                (image_size, image_size),
                method=Image.Resampling.BICUBIC,
            )
            canvas = Image.new("RGB", (image_size, image_size), (124, 116, 104))
            offset = (
                (image_size - resized.width) // 2,
                (image_size - resized.height) // 2,
            )
            canvas.paste(resized, offset)
            return canvas

    transform = transforms.Compose(
        [
            ResizePad(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    data_root = Path(DATA_ROOT)
    if split == "train":
        manifest = pd.read_csv(data_root / "train_manifest.csv")
        paths = [Path(path) for path in manifest["path"]]
        labels = manifest["label"].astype(np.int64).to_numpy()
        ids = np.arange(len(paths), dtype=np.int64)
    else:
        paths = sorted(
            (path for path in (data_root / "test").iterdir() if path.is_file()),
            key=lambda path: int(path.stem),
        )
        labels = np.full(len(paths), -1, dtype=np.int64)
        ids = np.asarray([int(path.stem) for path in paths], dtype=np.int64)

    expected = 26_527 if split == "train" else 1_458
    if len(paths) != expected:
        raise ValueError(f"Expected {expected} {split} images, found {len(paths)}")

    class ImageDataset(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            with Image.open(paths[index]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                return transform(image)

    loader = DataLoader(
        ImageDataset(),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    batches = []
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if config["kind"] == "dinov3":
                    hidden = model(pixel_values=images).last_hidden_state
                    cls_token = hidden[:, 0]
                    patch_mean = hidden[:, 1 + register_tokens :].mean(dim=1)
                    features = torch.cat((cls_token, patch_mean), dim=1)
                else:
                    features = model(pixel_values=images).pooler_output
                features = functional.normalize(features, dim=1)
            batches.append(features.float().cpu().numpy())

    features = np.concatenate(batches).astype(np.float16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        features=features,
        labels=labels,
        ids=ids,
        paths=np.asarray([str(path) for path in paths]),
    )
    cache_volume.commit()
    result = {"split": split, "rows": len(features), "features": features.shape[1], "cached": False}
    print(result, flush=True)
    return result


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=3 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
)
def fit_linear_probe(backbone: str):
    import hashlib
    import json

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold

    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone: {backbone}")
    config = BACKBONES[backbone]
    run_root = Path(CACHE_ROOT) / "runs" / config["run"]
    saved = np.load(run_root / "train_features.npz", allow_pickle=False)
    features = saved["features"].astype(np.float32)
    labels = saved["labels"].astype(int)
    paths = saved["paths"].astype(str)
    manifest = pd.read_csv(Path(DATA_ROOT) / "train_manifest.csv")
    if not np.array_equal(paths, manifest["path"].astype(str).to_numpy()):
        raise ValueError("Feature order does not match training manifest")

    splitter = StratifiedGroupKFold(
        n_splits=NUM_FOLDS, shuffle=True, random_state=SEED
    )
    oof_probabilities = np.zeros((len(labels), NUM_CLASSES), dtype=np.float32)
    folds = np.full(len(labels), -1, dtype=np.int8)
    fold_metrics = []
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(features, labels, manifest["group"])
    ):
        classifier = LogisticRegression(
            C=LINEAR_C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=500,
            tol=1e-5,
            random_state=SEED + fold,
        )
        classifier.fit(features[train_idx], labels[train_idx])
        probabilities = classifier.predict_proba(features[valid_idx])
        predictions = probabilities.argmax(axis=1)
        oof_probabilities[valid_idx] = probabilities
        folds[valid_idx] = fold
        fold_metrics.append(
            {
                "fold": fold,
                "macro_f1": float(
                    f1_score(labels[valid_idx], predictions, average="macro")
                ),
                "accuracy": float(accuracy_score(labels[valid_idx], predictions)),
            }
        )
        joblib.dump(classifier, run_root / f"linear_fold_{fold}.joblib")

    if np.any(folds < 0):
        raise RuntimeError("OOF assignment incomplete")
    final_classifier = LogisticRegression(
        C=LINEAR_C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=500,
        tol=1e-5,
        random_state=SEED,
    ).fit(features, labels)
    joblib.dump(final_classifier, run_root / "linear_full.joblib")

    oof_predictions = oof_probabilities.argmax(axis=1)
    oof = pd.DataFrame(
        {
            "path": paths,
            "label": labels,
            "fold": folds,
            "predicted": oof_predictions,
            "confidence": oof_probabilities.max(axis=1),
        }
    )
    oof.to_csv(run_root / "oof_predictions.csv", index=False)
    summary = {
        "model": config["model"],
        "model_revision": config["revision"],
        "dataset_fingerprint": hashlib.sha256(
            "".join(
                f"{row.path}|{row.label}|{row.group}\n"
                for row in manifest.itertuples()
            ).encode()
        ).hexdigest(),
        "seed": SEED,
        "image_size": config["image_size"],
        "linear_c": LINEAR_C,
        "oof_macro_f1": float(f1_score(labels, oof_predictions, average="macro")),
        "oof_accuracy": float(accuracy_score(labels, oof_predictions)),
        "folds": fold_metrics,
    }
    (run_root / "cv_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    cache_volume.commit()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=30 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
)
def create_submission(backbone: str):
    import io
    import json

    import joblib
    import numpy as np
    import pandas as pd

    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone: {backbone}")
    config = BACKBONES[backbone]
    run_root = Path(CACHE_ROOT) / "runs" / config["run"]
    saved = np.load(run_root / "test_features.npz", allow_pickle=False)
    features = saved["features"].astype(np.float32)
    ids = saved["ids"].astype(int)

    probabilities = []
    for fold in range(NUM_FOLDS):
        classifier = joblib.load(run_root / f"linear_fold_{fold}.joblib")
        probabilities.append(classifier.predict_proba(features))
    full_classifier = joblib.load(run_root / "linear_full.joblib")
    fold_mean = np.mean(probabilities, axis=0)
    final_probabilities = 0.5 * fold_mean + 0.5 * full_classifier.predict_proba(features)
    predictions = final_probabilities.argmax(axis=1).astype(int)

    prediction_map = dict(zip(ids, predictions, strict=True))
    submission = pd.read_csv(Path(DATA_ROOT) / "submission.csv")[["id"]]
    submission["predicted"] = submission["id"].astype(int).map(prediction_map)
    if submission["predicted"].isna().any() or len(submission) != 1_458:
        raise ValueError("Submission IDs do not match test images")
    submission["predicted"] = submission["predicted"].astype(int)
    output_path = run_root / config["output"]
    submission.to_csv(output_path, index=False)
    metadata = {
        "rows": len(submission),
        "class_counts": {
            str(label): int(count)
            for label, count in submission["predicted"].value_counts().sort_index().items()
        },
    }
    (run_root / "inference.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    cache_volume.commit()
    buffer = io.StringIO()
    submission.to_csv(buffer, index=False)
    print(metadata, flush=True)
    return buffer.getvalue()


def run_pipeline(
    backbone: str,
    action: str = "all",
    output_dir: str = ".",
    force: bool = False,
):
    if action not in {"train", "infer", "all"}:
        raise ValueError("action must be train, infer, or all")
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of: {', '.join(BACKBONES)}")
    config = BACKBONES[backbone]
    if action in {"train", "all"}:
        manifest = prepare_manifest.remote()
        print(f"Manifest: {manifest['rows']} rows")
        print(extract_features.remote(backbone, "train", force=force))
        metrics = fit_linear_probe.remote(backbone)
        print(f"OOF Macro-F1: {metrics['oof_macro_f1']:.6f}")
    if action in {"infer", "all"}:
        print(extract_features.remote(backbone, "test", force=force))
        csv_text = create_submission.remote(backbone)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        output_path = destination / config["output"]
        output_path.write_text(csv_text, encoding="utf-8")
        print(f"Wrote {output_path.resolve()}")
