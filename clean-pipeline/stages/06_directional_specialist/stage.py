"""Directional class-0 hard negatives, kNN, and 0-vs-2 calibration.

Requires run 04 checkpoints in the shared Modal cache.

Run:
  modal run artifacts/directional_specialist/modal_run09_pipeline.py
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

APP_NAME = "bdc2026-directional-boundary-specialist"
GPU = "A100-80GB"
MODEL_NAME = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
MODEL_REVISION = "e959efa74c867491dcfe3ec3e4f97382e39025b3"
BASE_RUN_NAME = "dinov3_convnext_large_full_finetune_224_seed2026"
BASE_ROOT = f"{CACHE_ROOT}/runs/{BASE_RUN_NAME}"
RUN_NAME = "dinov3_convnext_large_directional_hard_negative_seed2026"
RUN_ROOT = f"{CACHE_ROOT}/runs/{RUN_NAME}"
METHOD_VERSION = 1

NUM_CLASSES = 3
IMAGE_SIZE = 224
BATCH_SIZE = 32
GRAD_ACCUMULATION = 2
NUM_WORKERS = 8
FINE_TUNE_EPOCHS = 2
BACKBONE_LR = 2e-6
HEAD_LR = 2e-5
LAYER_DECAY = 0.8
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
HARD_FRACTION_CLASS_0 = 0.10
HARD_WEIGHT = 2.0
K_VALUES = (5, 11, 21, 31)
ALPHA_VALUES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
THRESHOLD_VALUES = tuple(round(0.35 + 0.01 * index, 2) for index in range(16))
KNN_TEMPERATURE = 0.07

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=32768,
    timeout=12 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def run_pipeline(force: bool = False):
    import io
    import json
    import math
    import os
    import random
    import re
    import time

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageFile, ImageOps
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModel, get_cosine_schedule_with_warmup

    run_root = Path(RUN_ROOT)
    metrics_path = run_root / "metrics.json"
    validation_path = run_root / "validation_predictions.csv"
    probability_path = run_root / "test_probabilities.npz"
    validation_probability_path = run_root / "validation_probabilities.npz"
    submission_names = {
        "base_calibrated": "submission_directional_base_calibrated.csv",
        "hard_negative_knn": "submission_directional_hard_negative_knn.csv",
        "hard_negative_calibrated": "submission_directional_hard_negative_calibrated.csv",
        "recommended": "submission_directional_recommended.csv",
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

    required = [
        metrics_path,
        validation_path,
        probability_path,
        validation_probability_path,
        *(run_root / name for name in submission_names.values()),
    ]
    if all(path.exists() for path in required) and not force:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("method_version") == METHOD_VERSION:
            metrics["cached"] = True
            return payload(metrics)

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

    base_root = Path(BASE_ROOT)
    base_validation_checkpoint = base_root / "best_validation_model.pt"
    base_final_checkpoint = base_root / "final_model.pt"
    for checkpoint in (base_validation_checkpoint, base_final_checkpoint):
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing {checkpoint}; complete run 04 training first"
            )

    train_indices, valid_indices = next(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(
            manifest, manifest["label"], manifest["group"]
        )
    )
    train_frame = manifest.iloc[train_indices].reset_index(drop=True)
    valid_frame = manifest.iloc[valid_indices].reset_index(drop=True)
    test_paths = sorted(
        (path for path in (data_root / "test").iterdir() if path.is_file()),
        key=lambda path: int(path.stem),
    )
    if len(test_paths) != 1_458:
        raise ValueError(f"Expected 1458 test images, found {len(test_paths)}")
    test_frame = pd.DataFrame(
        {"path": [str(path) for path in test_paths], "label": [-1] * len(test_paths)}
    )

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)

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

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.15, 0.15, 0.15, 0.05)], p=0.5
            ),
            transforms.RandomRotation(8, fill=(124, 116, 104)),
            ResizePad(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08), value="random"),
        ]
    )
    eval_transform = transforms.Compose([ResizePad(), transforms.ToTensor(), normalize])

    class WasteDataset(Dataset):
        def __init__(self, frame, transform):
            self.paths = frame["path"].astype(str).tolist()
            self.labels = frame["label"].astype(int).tolist()
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            with Image.open(self.paths[index]) as source:
                value = ImageOps.exif_transpose(source).convert("RGB")
                return self.transform(value), self.labels[index]

    class DINOConvNextClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
            self.classifier = nn.Linear(
                self.backbone.config.hidden_sizes[-1], NUM_CLASSES
            )

        def forward(self, pixel_values):
            pooled = self.backbone(pixel_values=pixel_values).pooler_output
            return self.classifier(pooled)

    device = torch.device("cuda")

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    def loader(frame, transform, sampler=None, seed=SEED):
        return DataLoader(
            WasteDataset(frame, transform),
            batch_size=BATCH_SIZE,
            shuffle=False,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed),
        )

    def load_model(checkpoint):
        model = DINOConvNextClassifier().to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.eval()
        return model

    @torch.inference_mode()
    def predict(model, frame):
        probabilities, features = [], []
        model.eval()
        for images, _ in loader(frame, eval_transform, seed=SEED + 99):
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pooled = model.backbone(pixel_values=images).pooler_output
                logits = model.classifier(pooled)
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            features.append(
                torch.nn.functional.normalize(pooled.float(), dim=1).cpu().numpy()
            )
        return np.concatenate(probabilities), np.concatenate(features)

    @torch.inference_mode()
    def neighbor_probabilities(reference_features, reference_labels, query_features, k):
        reference = torch.from_numpy(reference_features).to(device)
        labels = torch.from_numpy(reference_labels.astype(np.int64)).to(device)
        outputs = []
        for start in range(0, len(query_features), 256):
            query = torch.from_numpy(query_features[start : start + 256]).to(device)
            similarities, indices = (query @ reference.T).topk(k, dim=1)
            weights = torch.softmax(similarities / KNN_TEMPERATURE, dim=1)
            values = (
                torch.nn.functional.one_hot(labels[indices], NUM_CLASSES)
                * weights.unsqueeze(-1)
            ).sum(dim=1)
            outputs.append(values.cpu().numpy())
        del reference, labels
        return np.concatenate(outputs)

    def metrics(labels, predictions):
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "class_f1": [
                float(value)
                for value in f1_score(
                    labels, predictions, average=None, labels=[0, 1, 2]
                )
            ],
            "confusion_matrix": confusion_matrix(
                labels, predictions, labels=[0, 1, 2]
            ).tolist(),
        }

    def directional_predictions(probabilities, threshold):
        predictions = probabilities.argmax(axis=1).astype(np.int64)
        pair_sum = probabilities[:, 0] + probabilities[:, 2]
        ratio = np.divide(
            probabilities[:, 0],
            pair_sum,
            out=np.full(len(probabilities), 0.5, dtype=np.float64),
            where=pair_sum > 1e-12,
        )
        predictions[(predictions == 2) & (ratio > threshold)] = 0
        return predictions

    def select_threshold(labels, probabilities):
        baseline = probabilities.argmax(axis=1)
        grid = []
        for threshold in THRESHOLD_VALUES:
            predictions = directional_predictions(probabilities, threshold)
            grid.append(
                {
                    "threshold_0_vs_2": threshold,
                    **metrics(labels, predictions),
                    "changed_predictions": int(np.sum(predictions != baseline)),
                }
            )
        selected = max(
            grid,
            key=lambda row: (
                row["macro_f1"],
                -row["changed_predictions"],
                row["threshold_0_vs_2"],
            ),
        )
        return selected, grid, directional_predictions(
            probabilities, selected["threshold_0_vs_2"]
        )

    def directional_hard_indices(labels, probabilities):
        labels = np.asarray(labels, dtype=np.int64)
        class_zero = np.flatnonzero(labels == 0)
        margin = probabilities[class_zero, 0] - probabilities[class_zero, 2]
        wrong_count = int(np.sum(margin <= 0.0))
        count = max(wrong_count, round(HARD_FRACTION_CLASS_0 * len(class_zero)))
        selected = class_zero[np.argsort(margin)[:count]]
        return selected, {
            "class_0_rows": int(len(class_zero)),
            "class_0_predicted_as_2": wrong_count,
            "selected_rows": int(len(selected)),
            "selected_fraction_of_class_0": float(len(selected) / len(class_zero)),
            "largest_selected_margin_0_minus_2": float(
                np.max(probabilities[selected, 0] - probabilities[selected, 2])
            ),
        }

    probe = np.asarray([[0.0, 1.0, 0.0], [0.44, 0.01, 0.55]])
    assert directional_predictions(probe, 0.43).tolist() == [1, 0]

    def optimizer_for(model):
        grouped = {}
        depth = model.backbone.config.num_stages
        for name, parameter in model.named_parameters():
            if name.startswith("classifier"):
                learning_rate = HEAD_LR
            else:
                match = re.search(r"backbone\.stages\.(\d+)\.", name)
                if match:
                    learning_rate = BACKBONE_LR * LAYER_DECAY ** (
                        depth - 1 - int(match.group(1))
                    )
                elif "backbone.embeddings" in name:
                    learning_rate = BACKBONE_LR * LAYER_DECAY**depth
                else:
                    learning_rate = BACKBONE_LR
            decay = 0.0 if parameter.ndim == 1 or name.endswith(".bias") else WEIGHT_DECAY
            grouped.setdefault((learning_rate, decay), []).append(parameter)
        return torch.optim.AdamW(
            [
                {"params": values, "lr": learning_rate, "weight_decay": decay}
                for (learning_rate, decay), values in grouped.items()
            ]
        )

    def fine_tune(model, frame, sample_weights, epochs, valid=None, checkpoint=None):
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(SEED + 909),
        )
        train_loader = loader(
            frame, train_transform, sampler=sampler, seed=SEED + 909
        )
        optimizer = optimizer_for(model)
        updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUMULATION)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, round(updates_per_epoch * epochs * 0.05)),
            num_training_steps=updates_per_epoch * epochs,
        )
        counts = np.bincount(frame["label"], minlength=NUM_CLASSES)
        class_weights = len(frame) / (NUM_CLASSES * counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=LABEL_SMOOTHING,
        )
        history, best = [], None
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for step, (images, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw_loss = criterion(model(images), targets)
                (raw_loss / GRAD_ACCUMULATION).backward()
                total_loss += raw_loss.item()
                if (step + 1) % GRAD_ACCUMULATION == 0 or step + 1 == len(train_loader):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            row = {"epoch": epoch, "train_loss": total_loss / len(train_loader)}
            if valid is not None:
                valid_probabilities, _ = predict(model, valid)
                row.update(
                    metrics(valid["label"].to_numpy(), valid_probabilities.argmax(axis=1))
                )
                if best is None or row["macro_f1"] > best["macro_f1"]:
                    best = row.copy()
                    torch.save(model.state_dict(), checkpoint)
            history.append(row)
            print(json.dumps(row), flush=True)
        del optimizer, scheduler
        return history, best

    valid_labels = valid_frame["label"].to_numpy(dtype=np.int64)
    train_labels = train_frame["label"].to_numpy(dtype=np.int64)

    validation_started = time.perf_counter()
    base_validation_model = load_model(base_validation_checkpoint)
    base_train_probabilities, base_train_features = predict(
        base_validation_model, train_frame
    )
    base_valid_probabilities, base_valid_features = predict(
        base_validation_model, valid_frame
    )
    base_neighbor = neighbor_probabilities(
        base_train_features, train_labels, base_valid_features, 5
    )
    base_blend = 0.70 * base_valid_probabilities + 0.30 * base_neighbor
    base_calibration, base_calibration_grid, base_calibrated_predictions = (
        select_threshold(valid_labels, base_blend)
    )
    base_knn_metrics = metrics(valid_labels, base_blend.argmax(axis=1))

    hard_train_indices, hard_train_stats = directional_hard_indices(
        train_labels, base_train_probabilities
    )
    train_weights = np.ones(len(train_frame), dtype=np.float64)
    train_weights[hard_train_indices] = HARD_WEIGHT
    run_root.mkdir(parents=True, exist_ok=True)
    hard_validation_checkpoint = run_root / "best_validation_model.pt"
    fine_tune_started = time.perf_counter()
    validation_history, best_epoch_row = fine_tune(
        base_validation_model,
        train_frame,
        train_weights,
        FINE_TUNE_EPOCHS,
        valid=valid_frame,
        checkpoint=hard_validation_checkpoint,
    )
    validation_fine_tune_seconds = time.perf_counter() - fine_tune_started
    selected_epoch = int(best_epoch_row["epoch"])
    del base_validation_model, base_train_features, base_valid_features, base_neighbor
    torch.cuda.empty_cache()

    hard_validation_model = load_model(hard_validation_checkpoint)
    _, hard_train_features = predict(hard_validation_model, train_frame)
    hard_valid_model_probabilities, hard_valid_features = predict(
        hard_validation_model, valid_frame
    )
    knn_grid = []
    hard_neighbor_by_k = {}
    for k in K_VALUES:
        hard_neighbor_by_k[k] = neighbor_probabilities(
            hard_train_features, train_labels, hard_valid_features, k
        )
        for alpha in ALPHA_VALUES:
            probabilities = (
                (1.0 - alpha) * hard_valid_model_probabilities
                + alpha * hard_neighbor_by_k[k]
            )
            predictions = probabilities.argmax(axis=1)
            knn_grid.append(
                {"k": k, "alpha": alpha, **metrics(valid_labels, predictions)}
            )
    selected_knn = max(
        knn_grid,
        key=lambda row: (row["macro_f1"], -row["alpha"], -row["k"]),
    )
    hard_blend = (
        (1.0 - selected_knn["alpha"]) * hard_valid_model_probabilities
        + selected_knn["alpha"] * hard_neighbor_by_k[selected_knn["k"]]
    )
    hard_knn_predictions = hard_blend.argmax(axis=1)
    hard_calibration, hard_calibration_grid, hard_calibrated_predictions = (
        select_threshold(valid_labels, hard_blend)
    )
    validation_seconds = time.perf_counter() - validation_started

    validation_candidates = {
        "base_calibrated": base_calibration,
        "hard_negative_knn": metrics(valid_labels, hard_knn_predictions),
        "hard_negative_calibrated": hard_calibration,
    }
    recommended_name = max(
        validation_candidates,
        key=lambda name: validation_candidates[name]["macro_f1"],
    )

    del hard_validation_model, hard_train_features, hard_valid_features, hard_neighbor_by_k
    torch.cuda.empty_cache()

    full_labels = manifest["label"].to_numpy(dtype=np.int64)
    base_final_model = load_model(base_final_checkpoint)
    full_started = time.perf_counter()
    base_reference_started = time.perf_counter()
    base_full_probabilities, base_full_features = predict(base_final_model, manifest)
    base_reference_seconds = time.perf_counter() - base_reference_started
    base_test_started = time.perf_counter()
    base_test_model_probabilities, base_test_features = predict(base_final_model, test_frame)
    base_test_neighbor = neighbor_probabilities(
        base_full_features, full_labels, base_test_features, 5
    )
    base_test_blend = 0.70 * base_test_model_probabilities + 0.30 * base_test_neighbor
    base_test_predictions = directional_predictions(
        base_test_blend, base_calibration["threshold_0_vs_2"]
    )
    base_test_inference_seconds = time.perf_counter() - base_test_started

    hard_full_indices, hard_full_stats = directional_hard_indices(
        full_labels, base_full_probabilities
    )
    full_weights = np.ones(len(manifest), dtype=np.float64)
    full_weights[hard_full_indices] = HARD_WEIGHT
    final_fine_tune_started = time.perf_counter()
    final_history, _ = fine_tune(
        base_final_model,
        manifest,
        full_weights,
        selected_epoch,
    )
    final_fine_tune_seconds = time.perf_counter() - final_fine_tune_started
    hard_final_checkpoint = run_root / "final_model.pt"
    torch.save(base_final_model.state_dict(), hard_final_checkpoint)

    hard_reference_started = time.perf_counter()
    _, hard_full_features = predict(base_final_model, manifest)
    hard_reference_seconds = time.perf_counter() - hard_reference_started
    hard_test_started = time.perf_counter()
    hard_test_model_probabilities, hard_test_features = predict(base_final_model, test_frame)
    hard_test_neighbor = neighbor_probabilities(
        hard_full_features,
        full_labels,
        hard_test_features,
        int(selected_knn["k"]),
    )
    hard_test_blend = (
        (1.0 - selected_knn["alpha"]) * hard_test_model_probabilities
        + selected_knn["alpha"] * hard_test_neighbor
    )
    hard_test_knn_predictions = hard_test_blend.argmax(axis=1)
    hard_test_calibrated_predictions = directional_predictions(
        hard_test_blend, hard_calibration["threshold_0_vs_2"]
    )
    hard_test_inference_seconds = time.perf_counter() - hard_test_started
    full_and_test_seconds = time.perf_counter() - full_started

    source_ids = np.asarray([int(path.stem) for path in test_paths], dtype=np.int64)
    source_index = {value: index for index, value in enumerate(source_ids)}
    template = pd.read_csv(data_root / "submission.csv")[["id"]]
    template_ids = template["id"].astype(int).to_numpy()
    if len(template_ids) != 1_458 or set(template_ids) != set(source_ids):
        raise ValueError("Submission template IDs do not match test images")
    order = np.asarray([source_index[value] for value in template_ids], dtype=np.int64)
    source_predictions = {
        "base_calibrated": base_test_predictions,
        "hard_negative_knn": hard_test_knn_predictions,
        "hard_negative_calibrated": hard_test_calibrated_predictions,
    }
    source_predictions["recommended"] = source_predictions[recommended_name]
    submissions = {}
    for name, predictions in source_predictions.items():
        submission = template.copy()
        submission["predicted"] = predictions[order].astype(int)
        submission.to_csv(run_root / submission_names[name], index=False)
        submissions[name] = submission

    validation_output = valid_frame[["path", "label", "group"]].copy()
    validation_output["base_knn_predicted"] = base_blend.argmax(axis=1)
    validation_output["base_calibrated_predicted"] = base_calibrated_predictions
    validation_output["hard_negative_knn_predicted"] = hard_knn_predictions
    validation_output["hard_negative_calibrated_predicted"] = hard_calibrated_predictions
    validation_output.to_csv(validation_path, index=False)
    np.savez_compressed(
        validation_probability_path,
        labels=valid_labels,
        base_probabilities=base_blend.astype(np.float32),
        base_predictions=base_calibrated_predictions,
        hard_negative_probabilities=hard_blend.astype(np.float32),
        hard_negative_predictions=hard_calibrated_predictions,
    )
    np.savez_compressed(
        probability_path,
        ids=template_ids,
        base_probabilities=base_test_blend[order].astype(np.float32),
        base_predictions=base_test_predictions[order],
        hard_negative_probabilities=hard_test_blend[order].astype(np.float32),
        hard_negative_knn_predictions=hard_test_knn_predictions[order],
        hard_negative_calibrated_predictions=hard_test_calibrated_predictions[order],
        recommended_predictions=source_predictions[recommended_name][order],
    )

    metrics_report = {
        "method": "directional class-0 hard-negative fine-tuning + kNN + 0-vs-2 calibration",
        "method_version": METHOD_VERSION,
        "base_run": BASE_RUN_NAME,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "selection_source": "official train held-out validation only",
        "test_labels_used": False,
        "hard_negative_rule": "label=0; include all p2>=p0, then lowest p0-p2 margin until at least 10% of class 0",
        "hard_weight": HARD_WEIGHT,
        "fine_tune_epochs_tested": FINE_TUNE_EPOCHS,
        "selected_epoch": selected_epoch,
        "validation_head_history": validation_history,
        "final_training_history": final_history,
        "hard_train_stats": hard_train_stats,
        "hard_full_stats": hard_full_stats,
        "base_knn": {
            "k": 5,
            "alpha": 0.30,
            **base_knn_metrics,
        },
        "base_calibration_selected": base_calibration,
        "base_calibration_grid": base_calibration_grid,
        "hard_negative_knn_selected": selected_knn,
        "hard_negative_knn_grid": knn_grid,
        "hard_negative_calibration_selected": hard_calibration,
        "hard_negative_calibration_grid": hard_calibration_grid,
        "validation_candidates": validation_candidates,
        "recommended_submission": recommended_name,
        "test_class_counts": {
            name: {
                str(label): int(count)
                for label, count in frame["predicted"].value_counts().sort_index().items()
            }
            for name, frame in submissions.items()
        },
        "timing_seconds": {
            "validation_pipeline": validation_seconds,
            "validation_fine_tune": validation_fine_tune_seconds,
            "final_fine_tune": final_fine_tune_seconds,
            "base_reference_embeddings": base_reference_seconds,
            "base_test_inference_and_knn": base_test_inference_seconds,
            "hard_negative_reference_embeddings": hard_reference_seconds,
            "hard_negative_test_inference_and_knn": hard_test_inference_seconds,
            "full_reference_and_test": full_and_test_seconds,
            "end_to_end": time.perf_counter() - started_at,
        },
        "cached": False,
    }
    metrics_path.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")
    cache_volume.commit()
    print(json.dumps(metrics_report, indent=2), flush=True)
    return payload(metrics_report)


@app.local_entrypoint()
def main(
    output_dir: str = "artifacts/directional_specialist",
    force: bool = False,
):
    import json

    result = run_pipeline.remote(force=force)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in result["submissions"].items():
        (destination / f"submission_directional_{name}.csv").write_text(
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
    print(f"Wrote all run 09 artifacts to {destination.resolve()}")
