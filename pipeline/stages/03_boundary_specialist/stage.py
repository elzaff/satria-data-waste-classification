"""Ablations for the trained DINOv3 ConvNeXt-L run.

Run one experiment at a time:
  modal run artifacts/shape_encoder/modal_dinov3_convnext_ablations.py --action tta
  modal run artifacts/shape_encoder/modal_dinov3_convnext_ablations.py --action knn
  modal run artifacts/shape_encoder/modal_dinov3_convnext_ablations.py --action hard-negative
  modal run artifacts/shape_encoder/modal_dinov3_convnext_ablations.py --action ensemble
  modal run artifacts/shape_encoder/modal_dinov3_convnext_ablations.py --action all
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

APP_NAME = "bdc2026-dinov3-convnext-ablations"
GPU = "A100-80GB"
MODEL_NAME = "facebook/dinov3-convnext-large-pretrain-lvd1689m"
MODEL_REVISION = "e959efa74c867491dcfe3ec3e4f97382e39025b3"
RUN_NAME = "dinov3_convnext_large_full_finetune_224_seed2026"
RUN_ROOT = f"{CACHE_ROOT}/runs/{RUN_NAME}"
NUM_CLASSES = 3
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 8

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    timeout=12 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def run_ablation(method: str, force: bool = False):
    import io
    import json
    import math
    import os
    import random
    import re

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

    if method not in {"tta", "knn", "hard-negative"}:
        raise ValueError("method must be tta, knn, or hard-negative")

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
    run_root = Path(RUN_ROOT)
    output_root = run_root / "ablations" / method.replace("-", "_")
    metrics_path = output_root / "metrics.json"
    submission_path = output_root / f"submission_{method.replace('-', '_')}.csv"
    probability_path = output_root / "test_probabilities.npz"

    def payload():
        return {
            "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
            "submission": submission_path.read_text(encoding="utf-8"),
            "probabilities": probability_path.read_bytes(),
        }

    if metrics_path.exists() and submission_path.exists() and probability_path.exists() and not force:
        return payload()

    manifest = pd.read_csv(Path(DATA_ROOT) / "train_manifest.csv")
    manifest["label"] = manifest["label"].astype(int)
    if len(manifest) != 26_527 or set(manifest["label"]) != {0, 1, 2}:
        raise ValueError("Invalid training manifest; run the base pipeline first")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, valid_idx = next(splitter.split(manifest, manifest["label"], manifest["group"]))
    train_frame = manifest.iloc[train_idx].reset_index(drop=True)
    valid_frame = manifest.iloc[valid_idx].reset_index(drop=True)
    test_paths = sorted(
        (path for path in (Path(DATA_ROOT) / "test").iterdir() if path.is_file()),
        key=lambda path: int(path.stem),
    )
    if len(test_paths) != 1_458:
        raise ValueError(f"Expected 1458 test images, found {len(test_paths)}")
    test_frame = pd.DataFrame({"path": [str(path) for path in test_paths], "label": -1})

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)

    class ResizePad:
        def __init__(self, size):
            self.size = size

        def __call__(self, source):
            resized = ImageOps.contain(
                source, (self.size, self.size), method=Image.Resampling.BICUBIC
            )
            canvas = Image.new("RGB", (self.size, self.size), (124, 116, 104))
            canvas.paste(
                resized,
                ((self.size - resized.width) // 2, (self.size - resized.height) // 2),
            )
            return canvas

    def eval_transform(size, flip=False):
        items = [ResizePad(size)]
        if flip:
            items.append(transforms.RandomHorizontalFlip(p=1.0))
        items.extend([transforms.ToTensor(), normalize])
        return transforms.Compose(items)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.15, 0.15, 0.15, 0.05)], p=0.5
            ),
            transforms.RandomRotation(8, fill=(124, 116, 104)),
            ResizePad(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.08), value="random"),
        ]
    )

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
            self.classifier = nn.Linear(self.backbone.config.hidden_sizes[-1], NUM_CLASSES)

        def forward(self, pixel_values):
            features = self.backbone(pixel_values=pixel_values).pooler_output
            return self.classifier(features)

    def load_model(checkpoint_name):
        checkpoint = run_root / checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing {checkpoint}; run the base pipeline first")
        model = DINOConvNextClassifier().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        return model

    def loader(frame, transform, sampler=None):
        return DataLoader(
            WasteDataset(frame, transform),
            batch_size=BATCH_SIZE,
            shuffle=False,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )

    @torch.inference_mode()
    def predict(
        model,
        frame,
        size=IMAGE_SIZE,
        flip=False,
        embeddings=False,
        logits_only=False,
    ):
        model.eval()
        probabilities, features = [], []
        batch_size = max(8, int(BATCH_SIZE * (IMAGE_SIZE / size) ** 2))
        data_loader = DataLoader(
            WasteDataset(frame, eval_transform(size, flip)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )
        for images, _ in data_loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pooled = model.backbone(pixel_values=images).pooler_output
                logits = model.classifier(pooled)
            values = logits.float() if logits_only else torch.softmax(logits.float(), dim=1)
            probabilities.append(values.cpu().numpy())
            if embeddings:
                features.append(torch.nn.functional.normalize(pooled.float(), dim=1).cpu().numpy())
        result = np.concatenate(probabilities)
        return (result, np.concatenate(features)) if embeddings else result

    def metrics(labels, probabilities):
        predictions = probabilities.argmax(axis=1)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "class_f1": [
                float(value)
                for value in f1_score(labels, predictions, average=None, labels=[0, 1, 2])
            ],
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        }

    def save_result(probabilities, result_metrics):
        source_ids = [int(path.stem) for path in test_paths]
        source_index = {value: index for index, value in enumerate(source_ids)}
        submission = pd.read_csv(Path(DATA_ROOT) / "submission.csv")[["id"]]
        ids = submission["id"].astype(int).to_numpy()
        ordered = np.stack([probabilities[source_index[value]] for value in ids])
        submission["predicted"] = ordered.argmax(axis=1).astype(int)
        if len(submission) != 1_458 or submission["predicted"].isna().any():
            raise ValueError("Invalid submission")
        output_root.mkdir(parents=True, exist_ok=True)
        submission.to_csv(submission_path, index=False)
        np.savez(probability_path, ids=ids, probabilities=ordered)
        metrics_path.write_text(json.dumps(result_metrics, indent=2), encoding="utf-8")
        cache_volume.commit()
        return payload()

    if method == "tta":
        validation_model = load_model("best_validation_model.pt")
        components = {}
        for size in (224, 288, 320):
            for flip in (False, True):
                components[(size, flip)] = predict(validation_model, valid_frame, size, flip)
        variants = {
            "baseline_224": [(224, False)],
            "tta_224": [(224, False), (224, True)],
            "tta_224_288": [(224, False), (224, True), (288, False), (288, True)],
            "tta_224_288_320": [
                (224, False), (224, True), (288, False),
                (288, True), (320, False), (320, True),
            ],
        }
        labels = valid_frame["label"].to_numpy()
        grid = {}
        for name, keys in variants.items():
            average = np.mean([components[key] for key in keys], axis=0)
            grid[name] = metrics(labels, average)
        best_name = max(grid, key=lambda name: grid[name]["macro_f1"])
        final_model = load_model("final_model.pt")
        test_probabilities = np.mean(
            [predict(final_model, test_frame, *key) for key in variants[best_name]], axis=0
        )
        result_metrics = {
            "method": method,
            "selected_variant": best_name,
            "validation": grid,
            "selection_source": "official train validation split only",
        }
        return save_result(test_probabilities, result_metrics)

    def neighbor_probabilities(reference_features, reference_labels, query_features, k, temperature=0.07):
        reference = torch.from_numpy(reference_features).to(device)
        reference_labels = torch.from_numpy(reference_labels.astype(np.int64)).to(device)
        outputs = []
        for start in range(0, len(query_features), 256):
            query = torch.from_numpy(query_features[start : start + 256]).to(device)
            similarities, indices = (query @ reference.T).topk(k, dim=1)
            weights = torch.softmax(similarities / temperature, dim=1)
            labels = reference_labels[indices]
            probabilities = (
                torch.nn.functional.one_hot(labels, NUM_CLASSES)
                * weights.unsqueeze(-1)
            ).sum(dim=1)
            outputs.append(probabilities.cpu().numpy())
        return np.concatenate(outputs)

    if method == "knn":
        validation_model = load_model("best_validation_model.pt")
        train_probabilities, train_features = predict(
            validation_model, train_frame, embeddings=True
        )
        valid_probabilities, valid_features = predict(
            validation_model, valid_frame, embeddings=True
        )
        del train_probabilities
        labels = valid_frame["label"].to_numpy()
        reference_labels = train_frame["label"].to_numpy()
        grid = []
        best = {"macro_f1": -1.0}
        for k in (5, 11, 21, 31):
            neighbor = neighbor_probabilities(train_features, reference_labels, valid_features, k)
            for alpha in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
                blended = (1.0 - alpha) * valid_probabilities + alpha * neighbor
                row = {"k": k, "alpha": alpha, **metrics(labels, blended)}
                grid.append(row)
                if row["macro_f1"] > best["macro_f1"]:
                    best = row

        final_model = load_model("final_model.pt")
        _, full_features = predict(final_model, manifest, embeddings=True)
        test_probabilities, test_features = predict(final_model, test_frame, embeddings=True)
        neighbor = neighbor_probabilities(
            full_features,
            manifest["label"].to_numpy(),
            test_features,
            int(best["k"]),
        )
        test_probabilities = (
            (1.0 - best["alpha"]) * test_probabilities + best["alpha"] * neighbor
        )
        result_metrics = {
            "method": method,
            "selected": best,
            "validation_grid": grid,
            "selection_source": "official train validation split only",
        }
        return save_result(test_probabilities, result_metrics)

    def hard_indices(labels, probabilities):
        labels = np.asarray(labels, dtype=np.int64)
        true_probability = probabilities[np.arange(len(labels)), labels]
        masked = probabilities.copy()
        masked[np.arange(len(labels)), labels] = -1.0
        margin = true_probability - masked.max(axis=1)
        candidates = np.flatnonzero(np.isin(labels, [0, 2]))
        count = max(1, round(0.10 * len(candidates)))
        return candidates[np.argsort(margin[candidates])[:count]]

    def optimizer_for(model):
        grouped = {}
        depth = model.backbone.config.num_stages
        for name, parameter in model.named_parameters():
            if name.startswith("classifier"):
                learning_rate = 2e-5
            else:
                match = re.search(r"backbone\.stages\.(\d+)\.", name)
                learning_rate = 2e-6 * (0.8 ** (depth - 1 - int(match.group(1)))) if match else 2e-6
            decay = 0.0 if parameter.ndim == 1 or name.endswith(".bias") else 0.05
            grouped.setdefault((learning_rate, decay), []).append(parameter)
        return torch.optim.AdamW(
            [
                {"params": values, "lr": learning_rate, "weight_decay": decay}
                for (learning_rate, decay), values in grouped.items()
            ]
        )

    def fine_tune(model, frame, sample_weights, epochs, valid=None, checkpoint=None):
        generator = torch.Generator().manual_seed(SEED + 404)
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        train_loader = loader(frame, train_transform, sampler)
        optimizer = optimizer_for(model)
        updates_per_epoch = math.ceil(len(train_loader) / 2)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, round(updates_per_epoch * epochs * 0.05)),
            num_training_steps=updates_per_epoch * epochs,
        )
        counts = np.bincount(frame["label"], minlength=NUM_CLASSES)
        class_weights = len(frame) / (NUM_CLASSES * counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=0.05,
        )
        history = []
        best = None
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            for step, (images, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw_loss = criterion(model(images), targets)
                (raw_loss / 2).backward()
                loss_sum += raw_loss.item()
                if (step + 1) % 2 == 0 or step + 1 == len(train_loader):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            row = {"epoch": epoch, "train_loss": loss_sum / len(train_loader)}
            if valid is not None:
                valid_probabilities = predict(model, valid)
                row.update(metrics(valid["label"].to_numpy(), valid_probabilities))
                if best is None or row["macro_f1"] > best["macro_f1"]:
                    best = row.copy()
                    torch.save(model.state_dict(), checkpoint)
            history.append(row)
            print(json.dumps(row), flush=True)
        return history, best

    validation_model = load_model("best_validation_model.pt")
    train_probabilities = predict(validation_model, train_frame)
    valid_probabilities = predict(validation_model, valid_frame)
    baseline = metrics(valid_frame["label"].to_numpy(), valid_probabilities)
    selected = hard_indices(train_frame["label"].to_numpy(), train_probabilities)
    sample_weights = np.ones(len(train_frame), dtype=np.float64)
    sample_weights[selected] = 2.0
    output_root.mkdir(parents=True, exist_ok=True)
    validation_checkpoint = output_root / "best_validation_model.pt"
    history, best = fine_tune(
        validation_model,
        train_frame,
        sample_weights,
        2,
        valid_frame,
        validation_checkpoint,
    )
    selected_epoch = int(best["epoch"]) if best["macro_f1"] > baseline["macro_f1"] else 0

    final_model = load_model("final_model.pt")
    full_probabilities = predict(final_model, manifest)
    full_selected = hard_indices(manifest["label"].to_numpy(), full_probabilities)
    full_weights = np.ones(len(manifest), dtype=np.float64)
    full_weights[full_selected] = 2.0
    if selected_epoch:
        fine_tune(final_model, manifest, full_weights, selected_epoch)
        torch.save(final_model.state_dict(), output_root / "final_model.pt")
    test_logits = 0.5 * (
        predict(final_model, test_frame, flip=False, logits_only=True)
        + predict(final_model, test_frame, flip=True, logits_only=True)
    )
    test_probabilities = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    result_metrics = {
        "method": method,
        "baseline_validation": baseline,
        "validation_history": history,
        "selected_epoch": selected_epoch,
        "hard_fraction": 0.10,
        "hard_weight": 2.0,
        "hard_train_samples": int(len(selected)),
        "hard_full_samples": int(len(full_selected)),
        "selection_source": "official train validation split only",
    }
    return save_result(test_probabilities, result_metrics)


def write_remote_result(result, destination):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(
        __import__("json").dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    method = result["metrics"]["method"].replace("-", "_")
    (destination / f"submission_{method}.csv").write_text(
        result["submission"], encoding="utf-8"
    )
    (destination / "test_probabilities.npz").write_bytes(result["probabilities"])


def ensemble_local(output_root):
    import json

    import numpy as np
    import pandas as pd

    root = Path.cwd()
    sources = {
        "dinov3_convnext": root / "artifacts/shape_encoder/test_probabilities.npz",
        "convnextv2_global_local": root / "run/02_convnextv2-L-global-local/test_probabilities.npz",
        "dinov3_vitb": root / "run/03_dinov3vitB/test_probabilities.npz",
        "tta": output_root.parent / "01_tta/test_probabilities.npz",
        "knn": output_root.parent / "02_knn/test_probabilities.npz",
    }
    loaded = {}
    ids = None
    for name, path in sources.items():
        if not path.exists():
            continue
        with np.load(path) as values:
            current_ids = values["ids"].astype(int)
            probabilities = values["probabilities"].astype(np.float64)
        if ids is None:
            ids = current_ids
        elif not np.array_equal(ids, current_ids):
            raise ValueError(f"ID order differs in {path}")
        loaded[name] = probabilities
    required = {"dinov3_convnext", "convnextv2_global_local", "dinov3_vitb"}
    missing = required - loaded.keys()
    if missing:
        raise FileNotFoundError(f"Missing probability files: {sorted(missing)}")

    variants = {
        "three_models": {
            "dinov3_convnext": 0.50,
            "convnextv2_global_local": 0.30,
            "dinov3_vitb": 0.20,
        }
    }
    if {"tta", "knn"} <= loaded.keys():
        variants["tta_knn_models"] = {
            "tta": 0.40,
            "knn": 0.20,
            "convnextv2_global_local": 0.25,
            "dinov3_vitb": 0.15,
        }

    output_root.mkdir(parents=True, exist_ok=True)
    template = pd.read_csv(root / "BDC2026/submission.csv")[["id"]]
    if not np.array_equal(template["id"].astype(int).to_numpy(), ids):
        raise ValueError("Submission template ID order differs from probability files")
    metadata = {"selection": "fixed weights; no test labels used", "variants": {}}
    for name, weights in variants.items():
        probabilities = sum(loaded[source] * weight for source, weight in weights.items())
        submission = template.copy()
        submission["predicted"] = probabilities.argmax(axis=1).astype(int)
        submission.to_csv(output_root / f"submission_ensemble_{name}.csv", index=False)
        np.savez(
            output_root / f"test_probabilities_{name}.npz",
            ids=ids,
            probabilities=probabilities,
        )
        metadata["variants"][name] = {"weights": weights}
    (output_root / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


@app.local_entrypoint()
def main(action: str, force: bool = False):
    output_root = Path("artifacts")
    folders = {
        "tta": output_root / "shape_tta",
        "knn": output_root / "shape_knn",
        "hard-negative": output_root / "boundary_specialist",
    }
    if action == "ensemble":
        print(ensemble_local(output_root / "shape_ensemble"))
        return
    if action == "all":
        for method, destination in folders.items():
            result = run_ablation.remote(method, force=force)
            write_remote_result(result, destination)
            print(result["metrics"])
        print(ensemble_local(output_root / "shape_ensemble"))
        return
    if action not in folders:
        raise ValueError("action must be tta, knn, hard-negative, ensemble, or all")
    result = run_ablation.remote(action, force=force)
    write_remote_result(result, folders[action])
    print(result["metrics"])
