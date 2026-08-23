"""Inference-only selective expert router + transductive graph smoothing.

Requires completed runs 01, 04, 06, and 09 in the shared Modal cache.

Run:
  modal run artifacts/neighborhood_router/modal_run10_pipeline.py
"""

from pathlib import Path

import modal

from modal_backbone_app import (
    CACHE_ROOT,
    DATA_ROOT,
    SEED,
    cache_volume,
    data_volume,
    hf_secret,
    image,
)


shared_app = Path(__file__).with_name("modal_backbone_app.py")
if not shared_app.exists():
    shared_app = Path.cwd() / "modal_backbone_app.py"
image = image.add_local_file(shared_app, "/root/modal_backbone_app.py")

APP_NAME = "bdc2026-neighborhood-router"
GPU = "A100-80GB"

DINO_MODEL = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
DINO_REVISION = "e959efa74c867491dcfe3ec3e4f97382e39025b3"
CONVNEXT_MODEL = "facebook/convnextv2-large-22k-224"
CONVNEXT_REVISION = "e58a79c331e6c9acd20e3ba2de0e934c546f0eea"

BASE_DINO_ROOT = f"{CACHE_ROOT}/runs/dinov3_convnext_large_full_finetune_224_seed2026"
AUG_DINO_ROOT = f"{CACHE_ROOT}/runs/dinov3_convnext_large_bdc_augmentation_224_seed2026"
CONVNEXT_ROOT = f"{CACHE_ROOT}/runs/convnextv2_large_full_finetune_224_seed2026"
RUN09_ROOT = f"{CACHE_ROOT}/runs/dinov3_convnext_large_directional_hard_negative_seed2026"
RUN_ROOT = f"{CACHE_ROOT}/runs/run10_selective_router_graph_seed2026"

METHOD_VERSION = 1
NUM_CLASSES = 3
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 8
INNER_FOLDS = 5
ROUTER_C_VALUES = (0.01, 0.1, 1.0, 10.0)
GRAPH_K_VALUES = (5, 11, 21)
GRAPH_LAMBDAS = (0.10, 0.20, 0.30)
GRAPH_STEPS = (1, 3)
GRAPH_TEMPERATURE = 0.07
THRESHOLD_VALUES = tuple(round(0.35 + 0.01 * index, 2) for index in range(16))
MIN_IMPROVED_FOLDS = 3
MIN_MEAN_F1_GAIN = 1e-4

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=32768,
    timeout=6 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def run_pipeline(force: bool = False):
    import json
    import os
    import random
    import time

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageFile, ImageOps
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

    run_root = Path(RUN_ROOT)
    metrics_path = run_root / "metrics.json"
    validation_path = run_root / "validation_predictions.csv"
    probability_path = run_root / "test_probabilities.npz"
    validation_probability_path = run_root / "validation_probabilities.npz"
    submission_names = {
        "router": "submission_neighborhood_router.csv",
        "graph": "submission_neighborhood_graph.csv",
        "router_graph": "submission_neighborhood_router_graph.csv",
        "recommended": "submission_neighborhood_recommended.csv",
    }

    def payload(metrics):
        return {
            "metrics": metrics,
            "validation": validation_path.read_text(encoding="utf-8"),
            "probabilities": probability_path.read_bytes(),
            "validation_probabilities": validation_probability_path.read_bytes(),
            "submissions": {
                key: (run_root / name).read_text(encoding="utf-8")
                for key, name in submission_names.items()
            },
        }

    required_outputs = [
        metrics_path,
        validation_path,
        probability_path,
        validation_probability_path,
        *(run_root / name for name in submission_names.values()),
    ]
    if all(path.exists() for path in required_outputs) and not force:
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        if cached.get("method_version") == METHOD_VERSION:
            cached["cached"] = True
            return payload(cached)
    run_root.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    data_root = Path(DATA_ROOT)
    manifest_path = data_root / "train_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Missing train_manifest.csv; complete run 04 first")
    manifest = pd.read_csv(manifest_path)
    manifest["label"] = manifest["label"].astype(int)
    if (
        len(manifest) != 26_527
        or set(manifest.columns) != {"path", "label", "group"}
        or set(manifest["label"]) != {0, 1, 2}
    ):
        raise ValueError("Invalid training manifest")

    _, valid_indices = next(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            manifest, manifest["label"], manifest["group"]
        )
    )
    valid_frame = manifest.iloc[valid_indices].reset_index(drop=True)
    valid_labels = valid_frame["label"].to_numpy(dtype=np.int64)
    test_paths = sorted(
        (path for path in (data_root / "test").iterdir() if path.is_file()),
        key=lambda path: int(path.stem),
    )
    if len(test_paths) != 1_458:
        raise ValueError(f"Expected 1458 test images, found {len(test_paths)}")
    source_test_ids = np.asarray([int(path.stem) for path in test_paths], dtype=np.int64)
    test_frame = pd.DataFrame(
        {"path": [str(path) for path in test_paths], "label": [-1] * len(test_paths)}
    )

    for path in (
        Path(BASE_DINO_ROOT) / "best_validation_model.pt",
        Path(BASE_DINO_ROOT) / "final_model.pt",
        Path(AUG_DINO_ROOT) / "best_validation_model.pt",
        Path(AUG_DINO_ROOT) / "test_probabilities.npz",
        Path(CONVNEXT_ROOT) / "best_validation.pt",
        Path(CONVNEXT_ROOT) / "final_model.pt",
        Path(RUN09_ROOT) / "validation_probabilities.npz",
        Path(RUN09_ROOT) / "test_probabilities.npz",
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing prerequisite artifact: {path}")

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

    dino_processor = AutoImageProcessor.from_pretrained(
        DINO_MODEL, revision=DINO_REVISION
    )
    dino_transform = transforms.Compose(
        [
            ResizePad(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=dino_processor.image_mean, std=dino_processor.image_std
            ),
        ]
    )
    convnext_transform = transforms.Compose(
        [
            ResizePad(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    class WasteDataset(Dataset):
        def __init__(self, frame, transform, flip=False):
            self.paths = frame["path"].astype(str).tolist()
            self.labels = frame["label"].astype(int).tolist()
            self.transform = transform
            self.flip = flip

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            with Image.open(self.paths[index]) as source:
                value = ImageOps.exif_transpose(source).convert("RGB")
                if self.flip:
                    value = ImageOps.mirror(value)
                return self.transform(value), self.labels[index]

    class DINOClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(
                DINO_MODEL, revision=DINO_REVISION
            )
            self.classifier = nn.Linear(
                self.backbone.config.hidden_sizes[-1], NUM_CLASSES
            )

        def forward(self, pixel_values):
            pooled = self.backbone(pixel_values=pixel_values).pooler_output
            return self.classifier(pooled)

    device = torch.device("cuda")

    def loader(frame, transform, flip=False, batch_size=BATCH_SIZE):
        return DataLoader(
            WasteDataset(frame, transform, flip),
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )

    def load_dino(checkpoint):
        model = DINOClassifier().to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        return model

    @torch.inference_mode()
    def dino_probabilities(model, frame, flip_tta=False):
        outputs = []
        for images, _ in loader(frame, dino_transform):
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                if flip_tta:
                    logits = 0.5 * (logits + model(torch.flip(images, dims=[3])))
            outputs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(outputs)

    @torch.inference_mode()
    def dino_embeddings(model, frame):
        outputs = []
        for images, _ in loader(frame, dino_transform):
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pooled = model.backbone(pixel_values=images).pooler_output
            outputs.append(
                torch.nn.functional.normalize(pooled.float(), dim=1).cpu().numpy()
            )
        return np.concatenate(outputs)

    def load_convnext(checkpoint):
        model = AutoModelForImageClassification.from_pretrained(
            CONVNEXT_MODEL,
            revision=CONVNEXT_REVISION,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        ).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        model.eval()
        return model

    @torch.inference_mode()
    def convnext_probabilities(model, frame, flip_tta=True):
        logits_by_view = []
        for flip in ((False, True) if flip_tta else (False,)):
            outputs = []
            for images, _ in loader(
                frame, convnext_transform, flip=flip, batch_size=BATCH_SIZE * 2
            ):
                images = images.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(pixel_values=images).logits
                outputs.append(logits.float().cpu().numpy())
            logits_by_view.append(np.concatenate(outputs))
        mean_logits = np.mean(logits_by_view, axis=0)
        return torch.softmax(torch.from_numpy(mean_logits), dim=1).numpy()

    def reorder(values, ids, target_ids):
        index = {int(value): position for position, value in enumerate(ids)}
        if set(index) != set(int(value) for value in target_ids):
            raise ValueError("Probability IDs do not match test image IDs")
        return values[np.asarray([index[int(value)] for value in target_ids])]

    with np.load(
        Path(RUN09_ROOT) / "validation_probabilities.npz", allow_pickle=False
    ) as values:
        if not np.array_equal(values["labels"], valid_labels):
            raise ValueError("Run09 validation labels do not match current split")
        base_valid_probabilities = values["base_probabilities"].astype(np.float64)
        base_valid_predictions = values["base_predictions"].astype(np.int64)
    with np.load(Path(RUN09_ROOT) / "test_probabilities.npz", allow_pickle=False) as values:
        base_test_probabilities = reorder(
            values["base_probabilities"].astype(np.float64),
            values["ids"],
            source_test_ids,
        )
        base_test_predictions = reorder(
            values["base_predictions"].astype(np.int64),
            values["ids"],
            source_test_ids,
        )

    inference_started = time.perf_counter()
    aug_validation_model = load_dino(
        Path(AUG_DINO_ROOT) / "best_validation_model.pt"
    )
    aug_valid_probabilities = dino_probabilities(
        aug_validation_model, valid_frame, flip_tta=True
    )
    del aug_validation_model
    torch.cuda.empty_cache()
    with np.load(
        Path(AUG_DINO_ROOT) / "test_probabilities.npz", allow_pickle=False
    ) as values:
        aug_test_probabilities = reorder(
            values["probabilities"].astype(np.float64),
            values["ids"],
            source_test_ids,
        )

    conv_validation_model = load_convnext(
        Path(CONVNEXT_ROOT) / "best_validation.pt"
    )
    conv_valid_probabilities = convnext_probabilities(
        conv_validation_model, valid_frame, flip_tta=True
    )
    del conv_validation_model
    torch.cuda.empty_cache()
    conv_final_model = load_convnext(Path(CONVNEXT_ROOT) / "final_model.pt")
    conv_test_probabilities = convnext_probabilities(
        conv_final_model, test_frame, flip_tta=True
    )
    del conv_final_model
    torch.cuda.empty_cache()

    base_validation_model = load_dino(
        Path(BASE_DINO_ROOT) / "best_validation_model.pt"
    )
    valid_embeddings = dino_embeddings(base_validation_model, valid_frame)
    del base_validation_model
    torch.cuda.empty_cache()
    base_final_model = load_dino(Path(BASE_DINO_ROOT) / "final_model.pt")
    test_embeddings = dino_embeddings(base_final_model, test_frame)
    del base_final_model
    torch.cuda.empty_cache()
    expert_inference_seconds = time.perf_counter() - inference_started

    def metrics(labels, predictions):
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "class_f1": [
                float(value)
                for value in f1_score(
                    labels, predictions, labels=[0, 1, 2], average=None
                )
            ],
            "confusion_matrix": confusion_matrix(
                labels, predictions, labels=[0, 1, 2]
            ).tolist(),
        }

    def expert_features(base, augmented, convnext):
        experts = [np.clip(values, 1e-8, 1.0) for values in (base, augmented, convnext)]
        entropy = [-(values * np.log(values)).sum(axis=1, keepdims=True) for values in experts]
        margin = [
            (np.sort(values, axis=1)[:, -1] - np.sort(values, axis=1)[:, -2])[:, None]
            for values in experts
        ]
        return np.concatenate(
            [*experts, experts[1] - experts[0], experts[2] - experts[0], *entropy, *margin],
            axis=1,
        )

    def opposite_disagreement(base_predictions, augmented, convnext):
        opposite = np.where(base_predictions == 0, 2, 0)
        return (
            np.isin(base_predictions, [0, 2])
            & ((augmented.argmax(axis=1) == opposite) | (convnext.argmax(axis=1) == opposite))
        )

    def router_model(c_value):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=SEED,
            ),
        )

    valid_features = expert_features(
        base_valid_probabilities,
        aug_valid_probabilities,
        conv_valid_probabilities,
    )
    test_features = expert_features(
        base_test_probabilities,
        aug_test_probabilities,
        conv_test_probabilities,
    )
    valid_router_mask = opposite_disagreement(
        base_valid_predictions,
        aug_valid_probabilities,
        conv_valid_probabilities,
    )
    test_router_mask = opposite_disagreement(
        base_test_predictions,
        aug_test_probabilities,
        conv_test_probabilities,
    )

    inner_fold_ids = np.full(len(valid_frame), -1, dtype=np.int8)
    for fold, (_, fold_indices) in enumerate(
        StratifiedGroupKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED + 1010,
        ).split(valid_frame, valid_labels, valid_frame["group"])
    ):
        inner_fold_ids[fold_indices] = fold
    if np.any(inner_fold_ids < 0):
        raise RuntimeError("Inner fold assignment incomplete")

    baseline_fold_scores = [
        float(
            f1_score(
                valid_labels[inner_fold_ids == fold],
                base_valid_predictions[inner_fold_ids == fold],
                average="macro",
            )
        )
        for fold in range(INNER_FOLDS)
    ]
    router_grid = []
    router_oof_by_c = {}
    router_probability_by_c = {}
    for c_value in ROUTER_C_VALUES:
        predictions = base_valid_predictions.copy()
        routed_probabilities = base_valid_probabilities.copy()
        for fold in range(INNER_FOLDS):
            fit_rows = (inner_fold_ids != fold) & np.isin(valid_labels, [0, 2])
            query_rows = (inner_fold_ids == fold) & valid_router_mask
            model = router_model(c_value)
            model.fit(valid_features[fit_rows], valid_labels[fit_rows])
            if np.any(query_rows):
                pair_probabilities = model.predict_proba(valid_features[query_rows])
                pair_mass = routed_probabilities[query_rows][:, [0, 2]].sum(
                    axis=1, keepdims=True
                )
                routed_probabilities[np.ix_(query_rows, [0, 2])] = (
                    pair_probabilities * pair_mass
                )
                predictions[query_rows] = model.predict(valid_features[query_rows])
        fold_scores = [
            float(
                f1_score(
                    valid_labels[inner_fold_ids == fold],
                    predictions[inner_fold_ids == fold],
                    average="macro",
                )
            )
            for fold in range(INNER_FOLDS)
        ]
        row = {
            "C": c_value,
            **metrics(valid_labels, predictions),
            "routed_rows": int(valid_router_mask.sum()),
            "changed_predictions": int(np.sum(predictions != base_valid_predictions)),
            "fold_macro_f1": fold_scores,
            "improved_folds": int(
                sum(
                    score > baseline + 1e-12
                    for score, baseline in zip(
                        fold_scores, baseline_fold_scores, strict=True
                    )
                )
            ),
            "mean_fold_gain": float(
                np.mean(np.asarray(fold_scores) - baseline_fold_scores)
            ),
        }
        router_grid.append(row)
        router_oof_by_c[c_value] = predictions
        router_probability_by_c[c_value] = routed_probabilities

    stable_router_rows = [
        row
        for row in router_grid
        if row["improved_folds"] >= MIN_IMPROVED_FOLDS
        and row["mean_fold_gain"] >= MIN_MEAN_F1_GAIN
    ]
    router_selected = max(
        stable_router_rows or router_grid,
        key=lambda row: row["macro_f1"],
    )
    router_guard_passed = bool(stable_router_rows)
    selected_c = router_selected["C"]
    router_valid_predictions = router_oof_by_c[selected_c]
    router_valid_probabilities = router_probability_by_c[selected_c]

    final_router = router_model(selected_c)
    pair_rows = np.isin(valid_labels, [0, 2])
    final_router.fit(valid_features[pair_rows], valid_labels[pair_rows])
    router_test_probabilities = base_test_probabilities.copy()
    router_test_predictions = base_test_predictions.copy()
    if np.any(test_router_mask):
        pair_probabilities = final_router.predict_proba(test_features[test_router_mask])
        pair_mass = router_test_probabilities[test_router_mask][:, [0, 2]].sum(
            axis=1, keepdims=True
        )
        router_test_probabilities[np.ix_(test_router_mask, [0, 2])] = (
            pair_probabilities * pair_mass
        )
        router_test_predictions[test_router_mask] = final_router.predict(
            test_features[test_router_mask]
        )

    @torch.inference_mode()
    def graph_neighbors(features, maximum_k):
        reference = torch.from_numpy(features).to(device)
        all_indices, all_similarities = [], []
        for start in range(0, len(features), 512):
            query = reference[start : start + 512]
            similarities = query @ reference.T
            rows = torch.arange(len(query), device=device)
            similarities[rows, start + rows] = -torch.inf
            values, indices = similarities.topk(maximum_k, dim=1)
            all_indices.append(indices.cpu().numpy())
            all_similarities.append(values.float().cpu().numpy())
        return np.concatenate(all_indices), np.concatenate(all_similarities)

    graph_started = time.perf_counter()
    valid_neighbor_indices, valid_neighbor_similarities = graph_neighbors(
        valid_embeddings, max(GRAPH_K_VALUES)
    )
    test_neighbor_indices, test_neighbor_similarities = graph_neighbors(
        test_embeddings, max(GRAPH_K_VALUES)
    )

    def graph_smooth(probabilities, active_predictions, indices, similarities, k, lam, steps):
        unary = probabilities.copy()
        current = unary.copy()
        active = np.isin(active_predictions, [0, 2])
        weights = np.exp(
            (similarities[:, :k] - similarities[:, :k].max(axis=1, keepdims=True))
            / GRAPH_TEMPERATURE
        )
        weights /= weights.sum(axis=1, keepdims=True)
        pair_mass = unary[:, [0, 2]].sum(axis=1, keepdims=True)
        unary_pair = np.divide(
            unary[:, [0, 2]],
            pair_mass,
            out=np.full((len(unary), 2), 0.5, dtype=np.float64),
            where=pair_mass > 1e-12,
        )
        for _ in range(steps):
            neighbor = (
                current[indices[:, :k]] * weights[:, :, None]
            ).sum(axis=1)
            neighbor_mass = neighbor[:, [0, 2]].sum(axis=1, keepdims=True)
            neighbor_pair = np.divide(
                neighbor[:, [0, 2]],
                neighbor_mass,
                out=np.full((len(unary), 2), 0.5, dtype=np.float64),
                where=neighbor_mass > 1e-12,
            )
            mixed_pair = (1.0 - lam) * unary_pair + lam * neighbor_pair
            current = unary.copy()
            current[np.ix_(active, [0, 2])] = (
                mixed_pair[active] * pair_mass[active]
            )
        if not np.isfinite(current).all():
            raise FloatingPointError("Graph smoothing produced non-finite values")
        return current

    def directional_predictions(probabilities, threshold):
        predictions = probabilities.argmax(axis=1).astype(np.int64)
        pair_mass = probabilities[:, 0] + probabilities[:, 2]
        ratio = np.divide(
            probabilities[:, 0],
            pair_mass,
            out=np.full(len(probabilities), 0.5, dtype=np.float64),
            where=pair_mass > 1e-12,
        )
        predictions[(predictions == 2) & (ratio > threshold)] = 0
        return predictions

    def search_graph(unary, active_predictions, baseline_predictions):
        baseline_scores = [
            float(
                f1_score(
                    valid_labels[inner_fold_ids == fold],
                    baseline_predictions[inner_fold_ids == fold],
                    average="macro",
                )
            )
            for fold in range(INNER_FOLDS)
        ]
        grid = []
        cached_probabilities = {}
        for k in GRAPH_K_VALUES:
            for lam in GRAPH_LAMBDAS:
                for steps in GRAPH_STEPS:
                    smoothed = graph_smooth(
                        unary,
                        active_predictions,
                        valid_neighbor_indices,
                        valid_neighbor_similarities,
                        k,
                        lam,
                        steps,
                    )
                    cached_probabilities[(k, lam, steps)] = smoothed
                    for threshold in THRESHOLD_VALUES:
                        predictions = directional_predictions(smoothed, threshold)
                        fold_scores = [
                            float(
                                f1_score(
                                    valid_labels[inner_fold_ids == fold],
                                    predictions[inner_fold_ids == fold],
                                    average="macro",
                                )
                            )
                            for fold in range(INNER_FOLDS)
                        ]
                        grid.append(
                            {
                                "k": k,
                                "lambda": lam,
                                "steps": steps,
                                "threshold_0_vs_2": threshold,
                                **metrics(valid_labels, predictions),
                                "changed_predictions": int(
                                    np.sum(predictions != baseline_predictions)
                                ),
                                "fold_macro_f1": fold_scores,
                                "improved_folds": int(
                                    sum(
                                        score > base + 1e-12
                                        for score, base in zip(
                                            fold_scores, baseline_scores, strict=True
                                        )
                                    )
                                ),
                                "mean_fold_gain": float(
                                    np.mean(np.asarray(fold_scores) - baseline_scores)
                                ),
                            }
                        )
        stable_rows = [
            row
            for row in grid
            if row["improved_folds"] >= MIN_IMPROVED_FOLDS
            and row["mean_fold_gain"] >= MIN_MEAN_F1_GAIN
        ]
        selected = max(stable_rows or grid, key=lambda row: row["macro_f1"])
        selected_probabilities = cached_probabilities[
            (selected["k"], selected["lambda"], selected["steps"])
        ]
        selected_predictions = directional_predictions(
            selected_probabilities, selected["threshold_0_vs_2"]
        )
        guard_passed = bool(stable_rows)
        return selected, grid, selected_probabilities, selected_predictions, guard_passed

    graph_selected, graph_grid, graph_valid_probabilities, graph_valid_predictions, graph_guard = search_graph(
        base_valid_probabilities,
        base_valid_predictions,
        base_valid_predictions,
    )
    router_graph_selected, router_graph_grid, router_graph_valid_probabilities, router_graph_valid_predictions, router_graph_guard = search_graph(
        router_valid_probabilities,
        router_valid_predictions,
        router_valid_predictions,
    )

    def apply_graph(unary, active_predictions, selected):
        smoothed = graph_smooth(
            unary,
            active_predictions,
            test_neighbor_indices,
            test_neighbor_similarities,
            int(selected["k"]),
            float(selected["lambda"]),
            int(selected["steps"]),
        )
        return smoothed, directional_predictions(
            smoothed, float(selected["threshold_0_vs_2"])
        )

    graph_test_probabilities, graph_test_predictions = apply_graph(
        base_test_probabilities, base_test_predictions, graph_selected
    )
    router_graph_test_probabilities, router_graph_test_predictions = apply_graph(
        router_test_probabilities, router_test_predictions, router_graph_selected
    )
    graph_seconds = time.perf_counter() - graph_started

    candidates = {
        "base_calibrated": {
            **metrics(valid_labels, base_valid_predictions),
            "guard_passed": True,
        },
        "router": {
            **metrics(valid_labels, router_valid_predictions),
            "guard_passed": router_guard_passed,
        },
        "graph": {
            **metrics(valid_labels, graph_valid_predictions),
            "guard_passed": graph_guard,
        },
        "router_graph": {
            **metrics(valid_labels, router_graph_valid_predictions),
            "guard_passed": router_graph_guard and router_guard_passed,
        },
    }
    eligible_candidates = [
        name for name, row in candidates.items() if row["guard_passed"]
    ]
    recommended_name = max(
        eligible_candidates, key=lambda name: candidates[name]["macro_f1"]
    )

    test_predictions = {
        "base_calibrated": base_test_predictions,
        "router": router_test_predictions,
        "graph": graph_test_predictions,
        "router_graph": router_graph_test_predictions,
    }
    test_probabilities = {
        "base_calibrated": base_test_probabilities,
        "router": router_test_probabilities,
        "graph": graph_test_probabilities,
        "router_graph": router_graph_test_probabilities,
    }

    template = pd.read_csv(data_root / "submission.csv")[["id"]]
    template_ids = template["id"].astype(int).to_numpy()
    if len(template_ids) != 1_458 or set(template_ids) != set(source_test_ids):
        raise ValueError("Submission template IDs do not match test images")
    source_index = {
        value: index for index, value in enumerate(source_test_ids.tolist())
    }
    order = np.asarray([source_index[value] for value in template_ids], dtype=np.int64)
    submissions = {}
    for name in ("router", "graph", "router_graph"):
        frame = template.copy()
        frame["predicted"] = test_predictions[name][order].astype(int)
        frame.to_csv(run_root / submission_names[name], index=False)
        submissions[name] = frame
    recommended = template.copy()
    recommended["predicted"] = test_predictions[recommended_name][order].astype(int)
    recommended.to_csv(run_root / submission_names["recommended"], index=False)
    submissions["recommended"] = recommended

    validation_output = valid_frame[["path", "label", "group"]].copy()
    validation_output["inner_fold"] = inner_fold_ids
    validation_output["base_predicted"] = base_valid_predictions
    validation_output["router_eligible"] = valid_router_mask
    validation_output["router_predicted"] = router_valid_predictions
    validation_output["graph_predicted"] = graph_valid_predictions
    validation_output["router_graph_predicted"] = router_graph_valid_predictions
    validation_output.to_csv(validation_path, index=False)

    np.savez_compressed(
        validation_probability_path,
        labels=valid_labels,
        inner_folds=inner_fold_ids,
        base_probabilities=base_valid_probabilities.astype(np.float32),
        base_predictions=base_valid_predictions,
        router_probabilities=router_valid_probabilities.astype(np.float32),
        router_predictions=router_valid_predictions,
        graph_probabilities=graph_valid_probabilities.astype(np.float32),
        graph_predictions=graph_valid_predictions,
        router_graph_probabilities=router_graph_valid_probabilities.astype(np.float32),
        router_graph_predictions=router_graph_valid_predictions,
    )
    np.savez_compressed(
        probability_path,
        ids=template_ids,
        base_probabilities=base_test_probabilities[order].astype(np.float32),
        base_predictions=base_test_predictions[order],
        router_probabilities=router_test_probabilities[order].astype(np.float32),
        router_predictions=router_test_predictions[order],
        graph_probabilities=graph_test_probabilities[order].astype(np.float32),
        graph_predictions=graph_test_predictions[order],
        router_graph_probabilities=router_graph_test_probabilities[order].astype(np.float32),
        router_graph_predictions=router_graph_test_predictions[order],
        recommended_predictions=test_predictions[recommended_name][order],
    )

    report = {
        "method": "inference-only selective 0-vs-2 expert router + transductive graph smoothing",
        "method_version": METHOD_VERSION,
        "seed": SEED,
        "selection_source": "official train held-out validation with inner 5-fold OOF",
        "test_labels_used": False,
        "experts": [
            "run09 base calibrated DINOv3 ConvNeXt-L kNN",
            "run06 BDC augmentation DINOv3 ConvNeXt-L",
            "run01 ConvNeXtV2-L",
        ],
        "base_validation": candidates["base_calibrated"],
        "router_selected": router_selected,
        "router_guard_passed": router_guard_passed,
        "router_grid": router_grid,
        "router_validation_rows": int(valid_router_mask.sum()),
        "router_test_rows": int(test_router_mask.sum()),
        "graph_selected": graph_selected,
        "graph_guard_passed": graph_guard,
        "graph_grid": graph_grid,
        "router_graph_selected": router_graph_selected,
        "router_graph_guard_passed": router_graph_guard,
        "router_graph_grid": router_graph_grid,
        "selection_guard": {
            "minimum_improved_inner_folds": MIN_IMPROVED_FOLDS,
            "minimum_mean_macro_f1_gain": MIN_MEAN_F1_GAIN,
        },
        "validation_candidates": candidates,
        "recommended_submission": recommended_name,
        "test_class_counts": {
            name: {
                str(label): int(count)
                for label, count in frame["predicted"].value_counts().sort_index().items()
            }
            for name, frame in submissions.items()
        },
        "timing_seconds": {
            "expert_inference_and_embeddings": expert_inference_seconds,
            "graph_and_selection": graph_seconds,
            "end_to_end": time.perf_counter() - started_at,
        },
        "cached": False,
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cache_volume.commit()
    print(json.dumps(report, indent=2), flush=True)
    return payload(report)


@app.local_entrypoint()
def main(
    output_dir: str = "artifacts/neighborhood_router",
    force: bool = False,
):
    import json

    result = run_pipeline.remote(force=force)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in result["submissions"].items():
        (destination / f"submission_neighborhood_{name}.csv").write_text(
            content, encoding="utf-8"
        )
    (destination / "validation_predictions.csv").write_text(
        result["validation"], encoding="utf-8"
    )
    (destination / "test_probabilities.npz").write_bytes(result["probabilities"])
    (destination / "validation_probabilities.npz").write_bytes(
        result["validation_probabilities"]
    )
    (destination / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    print("Recommended:", result["metrics"]["recommended_submission"])
    print(
        "Validation Macro-F1:",
        result["metrics"]["validation_candidates"][
            result["metrics"]["recommended_submission"]
        ]["macro_f1"],
    )
    print(f"Wrote all run 10 artifacts to {destination.resolve()}")
