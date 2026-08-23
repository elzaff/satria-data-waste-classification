"""SigLIP2 SO400M partial fine-tune + guarded material-context ensemble.

Run from project root with Modal profile ``fazle``:

  modal run --detach artifacts/high_resolution_context/modal_run24_pipeline.py

Completed epochs resume automatically. Use ``--force`` only for a clean retrain.
No test labels or local ground-truth files are read by this pipeline.
"""

from pathlib import Path
import sys

import modal


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (
        candidate
        for candidate in (SCRIPT_PATH.parent, *SCRIPT_PATH.parents, Path.cwd())
        if (candidate / "modal_backbone_app.py").is_file()
    ),
    Path.cwd(),
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modal_backbone_app import (  # noqa: E402
    CACHE_ROOT,
    DATA_ROOT,
    HF_CACHE,
    SEED,
    cache_volume,
    data_volume,
    hf_secret,
    image,
)


RUN15_LOCAL = PROJECT_ROOT / "artifacts" / "probability_ensemble"
RUN17_LOCAL = PROJECT_ROOT / "artifacts" / "material_context_fusion"
LOCAL_FILES = (
    (RUN15_LOCAL / "validation_probabilities.npz", "/root/run15_validation.npz"),
    (RUN15_LOCAL / "test_probabilities.npz", "/root/run15_test.npz"),
    (RUN17_LOCAL / "validation_probabilities.npz", "/root/run17_validation.npz"),
    (RUN17_LOCAL / "test_probabilities.npz", "/root/run17_test.npz"),
)
image = image.pip_install("accelerate==1.10.1").add_local_file(
    PROJECT_ROOT / "modal_backbone_app.py", "/root/modal_backbone_app.py"
)
for local_path, remote_path in LOCAL_FILES:
    image = image.add_local_file(local_path, remote_path)


APP_NAME = "bdc2026-high-resolution-context"
GPU = "A100-80GB"
MODEL_REPO = "google/siglip2-so400m-patch14-384"
MODEL_REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
RUN_NAME = "high_resolution_context_encoder_seed2026"
RUN_ROOT = f"{CACHE_ROOT}/runs/{RUN_NAME}"

NUM_CLASSES = 3
IMAGE_SIZE = 384
BATCH_SIZE = 24
GRAD_ACCUMULATION = 2
HEAD_WARMUP_EPOCHS = 1
MAX_PARTIAL_EPOCHS = 4
EARLY_STOPPING_PATIENCE = 2
UNFROZEN_BLOCKS = 4
HEAD_LR = 3e-5
BACKBONE_LR = 2e-6
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
NUM_WORKERS = 8
RUN17_CONVNEXT_TEMPERATURE = 1.1
RUN17_CONVNEXT_WEIGHT = 0.275
BLEND_WEIGHTS = tuple(round(index * 0.025, 3) for index in range(21))
TEMPERATURES = tuple(round(0.7 + index * 0.1, 1) for index in range(7))
METHOD_VERSION = 1

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=49152,
    timeout=6 * 60 * 60,
    volumes={"/data": data_volume, "/cache": cache_volume},
    secrets=[hf_secret],
)
def train_and_infer(force: bool = False):
    import hashlib
    import json
    import math
    import os
    import random
    import time

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageFile, ImageOps
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel

    started_at = time.perf_counter()
    os.environ["PYTHONHASHSEED"] = str(SEED)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    seed_all(SEED)
    device = torch.device("cuda")
    run_root = Path(RUN_ROOT)
    run_root.mkdir(parents=True, exist_ok=True)

    metrics_path = run_root / "metrics.json"
    standalone_submission_path = run_root / "submission_high_resolution_siglip2_standalone.csv"
    blend_submission_path = run_root / "submission_high_resolution_blend.csv"
    recommended_submission_path = run_root / "submission_high_resolution_recommended.csv"
    validation_csv_path = run_root / "validation_predictions.csv"
    validation_probability_path = run_root / "validation_probabilities.npz"
    test_probability_path = run_root / "test_probabilities.npz"
    validation_stage_path = run_root / "validation_stage.json"
    validation_raw_path = run_root / "siglip2_validation_probabilities.npz"
    best_validation_path = run_root / "best_validation_model.pt"
    validation_head_resume = run_root / "validation_head_resume.pt"
    validation_partial_resume = run_root / "validation_partial_resume.pt"
    final_model_path = run_root / "final_model.pt"
    full_head_resume = run_root / "full_head_resume.pt"
    full_partial_resume = run_root / "full_partial_resume.pt"

    output_paths = (
        metrics_path,
        standalone_submission_path,
        blend_submission_path,
        recommended_submission_path,
        validation_csv_path,
        validation_probability_path,
        test_probability_path,
    )

    def make_payload(metrics):
        return {
            "metrics": metrics,
            "standalone_submission": standalone_submission_path.read_text(encoding="utf-8"),
            "blend_submission": blend_submission_path.read_text(encoding="utf-8"),
            "recommended_submission": recommended_submission_path.read_text(encoding="utf-8"),
            "validation_csv": validation_csv_path.read_text(encoding="utf-8"),
            "validation_probabilities": validation_probability_path.read_bytes(),
            "test_probabilities": test_probability_path.read_bytes(),
        }

    if all(path.exists() for path in output_paths) and not force:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["cached"] = True
        print("Run24 artifacts already complete; returning cached result.", flush=True)
        return make_payload(metrics)

    if force:
        for path in (
            *output_paths,
            validation_stage_path,
            validation_raw_path,
            best_validation_path,
            validation_head_resume,
            validation_partial_resume,
            final_model_path,
            full_head_resume,
            full_partial_resume,
        ):
            path.unlink(missing_ok=True)
        cache_volume.commit()

    manifest_path = Path(DATA_ROOT) / "train_manifest.csv"
    if not manifest_path.exists():
        rows = []
        folders = {
            0: Path(DATA_ROOT) / "train" / "0_Recyclable",
            1: Path(DATA_ROOT) / "train" / "1_Electronic",
            2: Path(DATA_ROOT) / "train" / "2_Organic",
        }
        for label, folder in folders.items():
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing training folder: {folder}")
            for path in sorted(item for item in folder.iterdir() if item.is_file()):
                rows.append(
                    {
                        "path": str(path),
                        "label": label,
                        "group": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        data_volume.commit()

    manifest = pd.read_csv(manifest_path)
    manifest["label"] = manifest["label"].astype(int)
    if (
        len(manifest) != 26_527
        or set(manifest.columns) != {"path", "label", "group"}
        or set(manifest["label"]) != {0, 1, 2}
    ):
        raise ValueError("Invalid training manifest")
    fingerprint = hashlib.sha256(
        "".join(
            f"{row.path}|{row.label}|{row.group}\n" for row in manifest.itertuples()
        ).encode()
    ).hexdigest()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_indices, valid_indices = next(
        splitter.split(manifest, manifest["label"], manifest["group"])
    )
    train_frame = manifest.iloc[train_indices].reset_index(drop=True)
    valid_frame = manifest.iloc[valid_indices].reset_index(drop=True)
    valid_labels = valid_frame["label"].to_numpy(dtype=np.int64)

    test_paths = sorted(
        (path for path in (Path(DATA_ROOT) / "test").iterdir() if path.is_file()),
        key=lambda path: int(path.stem),
    )
    if len(test_paths) != 1_458:
        raise ValueError(f"Expected 1458 test images, found {len(test_paths)}")
    test_ids = np.asarray([int(path.stem) for path in test_paths], dtype=np.int64)
    test_frame = pd.DataFrame({"path": [str(path) for path in test_paths]})

    run15_validation = np.load("/root/run15_validation.npz", allow_pickle=False)
    run15_test = np.load("/root/run15_test.npz", allow_pickle=False)
    run17_validation = np.load("/root/run17_validation.npz", allow_pickle=False)
    run17_test = np.load("/root/run17_test.npz", allow_pickle=False)
    if not np.array_equal(run15_validation["labels"], valid_labels):
        raise ValueError("Run15 validation split does not match Run24")
    if not np.array_equal(run17_validation["labels"], valid_labels):
        raise ValueError("Run17 validation split does not match Run24")
    if not np.array_equal(run15_validation["inner_folds"], run17_validation["inner_folds"]):
        raise ValueError("Run15 and Run17 inner folds differ")
    if not np.array_equal(run15_test["ids"], test_ids):
        raise ValueError("Run15 test IDs do not match Run24")
    if not np.array_equal(run17_test["ids"], test_ids):
        raise ValueError("Run17 test IDs do not match Run24")
    inner_folds = run15_validation["inner_folds"].astype(np.int64)

    def temperature_scale(probabilities, temperature):
        logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        output = np.exp(logits)
        return output / output.sum(axis=1, keepdims=True)

    run15_validation_probabilities = run15_validation["soup_probabilities"].astype(np.float64)
    run15_test_probabilities = run15_test["soup_probabilities"].astype(np.float64)
    convnext_validation_probabilities = temperature_scale(
        run17_validation["probabilities"].astype(np.float64),
        RUN17_CONVNEXT_TEMPERATURE,
    )
    convnext_test_probabilities = temperature_scale(
        run17_test["probabilities"].astype(np.float64),
        RUN17_CONVNEXT_TEMPERATURE,
    )
    baseline_validation_probabilities = (
        (1.0 - RUN17_CONVNEXT_WEIGHT) * run15_validation_probabilities
        + RUN17_CONVNEXT_WEIGHT * convnext_validation_probabilities
    )
    baseline_test_probabilities = (
        (1.0 - RUN17_CONVNEXT_WEIGHT) * run15_test_probabilities
        + RUN17_CONVNEXT_WEIGHT * convnext_test_probabilities
    )
    baseline_validation_probabilities /= baseline_validation_probabilities.sum(
        axis=1, keepdims=True
    )
    baseline_test_probabilities /= baseline_test_probabilities.sum(axis=1, keepdims=True)

    processor = AutoImageProcessor.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        cache_dir=HF_CACHE,
        token=os.environ.get("HF_TOKEN"),
        use_fast=False,
    )
    mean, std = processor.image_mean, processor.image_std

    class ResizePad:
        def __call__(self, source):
            resized = ImageOps.contain(
                source,
                (IMAGE_SIZE, IMAGE_SIZE),
                method=Image.Resampling.BICUBIC,
            )
            canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (127, 127, 127))
            canvas.paste(
                resized,
                ((IMAGE_SIZE - resized.width) // 2, (IMAGE_SIZE - resized.height) // 2),
            )
            return canvas

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.75, 1.0),
                ratio=(0.80, 1.25),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.35
            ),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(
                p=0.10,
                scale=(0.02, 0.15),
                ratio=(0.5, 2.0),
                value="random",
            ),
        ]
    )
    eval_transform = transforms.Compose([ResizePad(), transforms.ToTensor(), normalize])

    class WasteDataset(Dataset):
        def __init__(self, frame, transform, with_labels=True):
            self.paths = frame["path"].astype(str).tolist()
            self.labels = frame["label"].astype(int).tolist() if with_labels else None
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            with Image.open(self.paths[index]) as source:
                image_value = ImageOps.exif_transpose(source).convert("RGB")
                value = self.transform(image_value)
            return (value, self.labels[index]) if self.labels is not None else value

    def make_loader(frame, transform, shuffle, seed, with_labels=True):
        return DataLoader(
            WasteDataset(frame, transform, with_labels),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=False,
            generator=torch.Generator().manual_seed(seed),
        )

    class Siglip2Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            config = AutoConfig.from_pretrained(
                MODEL_REPO,
                revision=MODEL_REVISION,
                cache_dir=HF_CACHE,
                token=os.environ.get("HF_TOKEN"),
            ).vision_config
            self.backbone = SiglipVisionModel.from_pretrained(
                MODEL_REPO,
                config=config,
                revision=MODEL_REVISION,
                cache_dir=HF_CACHE,
                token=os.environ.get("HF_TOKEN"),
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                torch_dtype=torch.float32,
            )
            self.classifier = nn.Linear(config.hidden_size, NUM_CLASSES)
            nn.init.trunc_normal_(self.classifier.weight, std=0.02)
            nn.init.zeros_(self.classifier.bias)

        def forward(self, images):
            pooled = self.backbone(pixel_values=images).pooler_output
            return self.classifier(pooled)

    def new_model():
        return Siglip2Classifier()

    def freeze_for_head(model):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_tail(model):
        freeze_for_head(model)
        vision = model.backbone.vision_model
        if len(vision.encoder.layers) < UNFROZEN_BLOCKS:
            raise ValueError("Unexpected SigLIP2 encoder layout")
        for block in vision.encoder.layers[-UNFROZEN_BLOCKS:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for module in (vision.post_layernorm, vision.head):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def optimizer_groups(model, partial):
        if not partial:
            return [{"params": model.classifier.parameters(), "lr": HEAD_LR}]
        head_ids = {id(parameter) for parameter in model.classifier.parameters()}
        backbone = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in head_ids
        ]
        return [
            {"params": backbone, "lr": BACKBONE_LR},
            {"params": model.classifier.parameters(), "lr": HEAD_LR},
        ]

    def score(labels, probabilities):
        predictions = probabilities.argmax(axis=1)
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
            "errors": int((predictions != labels).sum()),
        }

    @torch.inference_mode()
    def predict(model, loader, tta=True):
        model.eval()
        batches = []
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                if tta:
                    logits = 0.5 * (logits + model(torch.flip(images, dims=[3])))
            batches.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(batches)

    def atomic_torch_save(value, path):
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(value, temporary)
        os.replace(temporary, path)

    def save_resume(path, phase, epoch, model, optimizer, scheduler, loader, history, best, stale):
        atomic_torch_save(
            {
                "method_version": METHOD_VERSION,
                "dataset_fingerprint": fingerprint,
                "phase": phase,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loader_generator": loader.generator.get_state(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "history": history,
                "best": best,
                "stale": stale,
            },
            path,
        )
        cache_volume.commit()

    def train_phase(
        model,
        loader,
        class_weights,
        epochs,
        phase,
        partial,
        resume_path,
        valid_loader=None,
        checkpoint_path=None,
        initial_best=-1.0,
        patience=None,
    ):
        optimizer = torch.optim.AdamW(
            optimizer_groups(model, partial), weight_decay=WEIGHT_DECAY
        )
        updates_per_epoch = math.ceil(len(loader) / GRAD_ACCUMULATION)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, updates_per_epoch * epochs)
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=LABEL_SMOOTHING,
        )
        history = []
        best = initial_best
        stale = 0
        start_epoch = 1
        if resume_path.exists():
            resume = torch.load(resume_path, map_location=device, weights_only=False)
            if resume["dataset_fingerprint"] != fingerprint or resume["phase"] != phase:
                raise ValueError(f"Invalid resume state: {resume_path}")
            model.load_state_dict(resume["model"])
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            loader.generator.set_state(resume["loader_generator"])
            random.setstate(resume["python_rng"])
            np.random.set_state(resume["numpy_rng"])
            torch.set_rng_state(resume["torch_rng"])
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
            history = resume["history"]
            best = resume["best"]
            stale = resume["stale"]
            start_epoch = resume["epoch"] + 1
            print(f"Resuming {phase} from epoch {start_epoch}", flush=True)

        if patience is not None and stale >= patience:
            del optimizer, scheduler, criterion
            return history, best

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for step, (images, targets) in enumerate(loader):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw_loss = criterion(model(images), targets)
                (raw_loss / GRAD_ACCUMULATION).backward()
                total_loss += raw_loss.item()
                if (step + 1) % GRAD_ACCUMULATION == 0 or step + 1 == len(loader):
                    nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad), 1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            row = {
                "phase": phase,
                "epoch": epoch,
                "train_loss": total_loss / len(loader),
            }
            if valid_loader is not None:
                row.update(score(valid_labels, predict(model, valid_loader, tta=False)))
                if row["macro_f1"] > best:
                    best = row["macro_f1"]
                    stale = 0
                    atomic_torch_save(
                        {
                            "model": model.state_dict(),
                            "phase": phase,
                            "epoch": epoch,
                            "macro_f1": best,
                            "dataset_fingerprint": fingerprint,
                        },
                        checkpoint_path,
                    )
                else:
                    stale += 1
            history.append(row)
            print(json.dumps(row), flush=True)
            save_resume(
                resume_path,
                phase,
                epoch,
                model,
                optimizer,
                scheduler,
                loader,
                history,
                best,
                stale,
            )
            if patience is not None and stale >= patience:
                break
        del optimizer, scheduler, criterion
        return history, best

    training_started_at = time.perf_counter()
    if validation_stage_path.exists() and validation_raw_path.exists() and not force:
        validation_stage = json.loads(validation_stage_path.read_text(encoding="utf-8"))
        saved_validation = np.load(validation_raw_path, allow_pickle=False)
        siglip2_validation_probabilities = saved_validation["probabilities"].astype(np.float64)
        if not np.array_equal(saved_validation["labels"], valid_labels):
            raise ValueError("Cached SigLIP2 validation labels differ")
        print("Using completed validation stage.", flush=True)
    else:
        train_counts = np.bincount(train_frame["label"], minlength=NUM_CLASSES)
        train_weights = len(train_frame) / (NUM_CLASSES * train_counts)
        train_loader = make_loader(train_frame, train_transform, True, SEED)
        valid_loader = make_loader(valid_frame, eval_transform, False, SEED)

        validation_model = new_model().to(device)
        freeze_for_head(validation_model)
        head_history, head_best = train_phase(
            validation_model,
            train_loader,
            train_weights,
            HEAD_WARMUP_EPOCHS,
            "validation_head",
            False,
            validation_head_resume,
            valid_loader=valid_loader,
            checkpoint_path=best_validation_path,
        )
        unfreeze_tail(validation_model)
        partial_history, _ = train_phase(
            validation_model,
            train_loader,
            train_weights,
            MAX_PARTIAL_EPOCHS,
            "validation_partial",
            True,
            validation_partial_resume,
            valid_loader=valid_loader,
            checkpoint_path=best_validation_path,
            initial_best=head_best,
            patience=EARLY_STOPPING_PATIENCE,
        )

        best_checkpoint = torch.load(
            best_validation_path, map_location=device, weights_only=False
        )
        validation_model.load_state_dict(best_checkpoint["model"])
        siglip2_validation_probabilities = predict(
            validation_model, valid_loader, tta=True
        ).astype(np.float64)
        best_partial_epoch = (
            int(best_checkpoint["epoch"])
            if best_checkpoint["phase"] == "validation_partial"
            else 0
        )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in validation_model.parameters()
            if parameter.requires_grad
        )
        total_parameters = sum(
            parameter.numel() for parameter in validation_model.parameters()
        )
        validation_stage = {
            "head_history": head_history,
            "partial_history": partial_history,
            "best_phase": best_checkpoint["phase"],
            "best_partial_epoch": best_partial_epoch,
            "best_validation_macro_f1_without_tta": best_checkpoint["macro_f1"],
            "siglip2_validation": score(
                valid_labels, siglip2_validation_probabilities
            ),
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
        }
        np.savez(
            validation_raw_path,
            labels=valid_labels,
            probabilities=siglip2_validation_probabilities.astype(np.float32),
        )
        validation_stage_path.write_text(
            json.dumps(validation_stage, indent=2), encoding="utf-8"
        )
        validation_head_resume.unlink(missing_ok=True)
        validation_partial_resume.unlink(missing_ok=True)
        cache_volume.commit()
        del validation_model, train_loader, valid_loader
        torch.cuda.empty_cache()

    best_partial_epoch = int(validation_stage["best_partial_epoch"])
    if final_model_path.exists() and not force:
        final_model = new_model().to(device)
        final_checkpoint = torch.load(
            final_model_path, map_location=device, weights_only=False
        )
        if final_checkpoint["dataset_fingerprint"] != fingerprint:
            raise ValueError("Final checkpoint dataset mismatch")
        final_model.load_state_dict(final_checkpoint["model"])
        full_head_history = final_checkpoint["head_history"]
        full_partial_history = final_checkpoint["partial_history"]
        print("Using completed full-training checkpoint.", flush=True)
    else:
        seed_all(SEED + 100)
        full_counts = np.bincount(manifest["label"], minlength=NUM_CLASSES)
        full_weights = len(manifest) / (NUM_CLASSES * full_counts)
        full_loader = make_loader(manifest, train_transform, True, SEED + 100)
        final_model = new_model().to(device)
        freeze_for_head(final_model)
        full_head_history, _ = train_phase(
            final_model,
            full_loader,
            full_weights,
            HEAD_WARMUP_EPOCHS,
            "full_head",
            False,
            full_head_resume,
        )
        full_partial_history = []
        if best_partial_epoch > 0:
            unfreeze_tail(final_model)
            full_partial_history, _ = train_phase(
                final_model,
                full_loader,
                full_weights,
                best_partial_epoch,
                "full_partial",
                True,
                full_partial_resume,
            )
        atomic_torch_save(
            {
                "model": final_model.state_dict(),
                "dataset_fingerprint": fingerprint,
                "best_partial_epoch": best_partial_epoch,
                "head_history": full_head_history,
                "partial_history": full_partial_history,
            },
            final_model_path,
        )
        full_head_resume.unlink(missing_ok=True)
        full_partial_resume.unlink(missing_ok=True)
        cache_volume.commit()
        del full_loader

    training_seconds = time.perf_counter() - training_started_at
    test_loader = make_loader(test_frame, eval_transform, False, SEED, with_labels=False)
    inference_started_at = time.perf_counter()
    siglip2_test_probabilities = predict(final_model, test_loader, tta=True).astype(
        np.float64
    )
    inference_seconds = time.perf_counter() - inference_started_at
    del final_model, test_loader
    torch.cuda.empty_cache()

    baseline_metrics = score(valid_labels, baseline_validation_probabilities)
    folds = sorted(np.unique(inner_folds))
    baseline_fold_scores = [
        float(
            f1_score(
                valid_labels[inner_folds == fold],
                baseline_validation_probabilities[inner_folds == fold].argmax(axis=1),
                average="macro",
            )
        )
        for fold in folds
    ]
    grid = []
    for temperature in TEMPERATURES:
        scaled_validation = temperature_scale(
            siglip2_validation_probabilities, temperature
        )
        for weight in BLEND_WEIGHTS:
            probabilities = (
                (1.0 - weight) * baseline_validation_probabilities
                + weight * scaled_validation
            )
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            row = score(valid_labels, probabilities)
            fold_scores = [
                float(
                    f1_score(
                        valid_labels[inner_folds == fold],
                        probabilities[inner_folds == fold].argmax(axis=1),
                        average="macro",
                    )
                )
                for fold in folds
            ]
            row.update(
                {
                    "temperature": temperature,
                    "siglip2_weight": weight,
                    "fold_macro_f1": fold_scores,
                    "mean_fold_macro_f1": float(np.mean(fold_scores)),
                    "non_degrading_folds": int(
                        sum(
                            new >= old - 1e-12
                            for new, old in zip(fold_scores, baseline_fold_scores)
                        )
                    ),
                }
            )
            grid.append(row)

    baseline_mean_fold = float(np.mean(baseline_fold_scores))
    eligible = [
        row
        for row in grid
        if row["siglip2_weight"] > 0
        and row["macro_f1"] > baseline_metrics["macro_f1"] + 1e-12
        and row["mean_fold_macro_f1"] >= baseline_mean_fold - 1e-12
        and row["non_degrading_folds"] >= 4
    ]
    selected = max(
        eligible,
        key=lambda row: (
            row["macro_f1"],
            row["mean_fold_macro_f1"],
            -row["siglip2_weight"],
        ),
        default={
            **baseline_metrics,
            "temperature": 1.0,
            "siglip2_weight": 0.0,
            "fold_macro_f1": baseline_fold_scores,
            "mean_fold_macro_f1": baseline_mean_fold,
            "non_degrading_folds": 5,
        },
    )
    selected_temperature = selected["temperature"]
    selected_weight = selected["siglip2_weight"]
    scaled_validation = temperature_scale(
        siglip2_validation_probabilities, selected_temperature
    )
    scaled_test = temperature_scale(siglip2_test_probabilities, selected_temperature)
    blended_validation_probabilities = (
        (1.0 - selected_weight) * baseline_validation_probabilities
        + selected_weight * scaled_validation
    )
    blended_test_probabilities = (
        (1.0 - selected_weight) * baseline_test_probabilities
        + selected_weight * scaled_test
    )
    blended_validation_probabilities /= blended_validation_probabilities.sum(
        axis=1, keepdims=True
    )
    blended_test_probabilities /= blended_test_probabilities.sum(axis=1, keepdims=True)

    template = pd.read_csv(Path(DATA_ROOT) / "submission.csv")[["id"]]

    def make_submission(predictions):
        result = template.copy()
        prediction_map = dict(zip(test_ids, predictions.astype(int), strict=True))
        result["predicted"] = result["id"].astype(int).map(prediction_map)
        if len(result) != 1_458 or result["predicted"].isna().any():
            raise ValueError("Invalid submission mapping")
        result["predicted"] = result["predicted"].astype(int)
        if set(result.columns) != {"id", "predicted"}:
            raise ValueError("Invalid submission columns")
        return result

    standalone_predictions = siglip2_test_probabilities.argmax(axis=1)
    baseline_predictions = baseline_test_probabilities.argmax(axis=1)
    blended_predictions = blended_test_probabilities.argmax(axis=1)
    standalone_submission = make_submission(standalone_predictions)
    blend_submission = make_submission(blended_predictions)
    recommended_submission = blend_submission.copy()

    validation_output = valid_frame[["path", "label", "group"]].rename(
        columns={"label": "groundtruth"}
    )
    validation_output["inner_fold"] = inner_folds
    validation_output["run17_predicted"] = baseline_validation_probabilities.argmax(axis=1)
    validation_output["siglip2_predicted"] = siglip2_validation_probabilities.argmax(axis=1)
    validation_output["blended_predicted"] = blended_validation_probabilities.argmax(axis=1)
    for label in range(NUM_CLASSES):
        validation_output[f"run17_p{label}"] = baseline_validation_probabilities[:, label]
        validation_output[f"siglip2_p{label}"] = siglip2_validation_probabilities[:, label]
        validation_output[f"blended_p{label}"] = blended_validation_probabilities[:, label]

    metrics = {
        "method_version": METHOD_VERSION,
        "method": "SigLIP2 SO400M partial fine-tune + guarded Run17 probability ensemble",
        "test_labels_used": False,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
        "head_warmup_epochs": HEAD_WARMUP_EPOCHS,
        "best_partial_epochs": best_partial_epoch,
        "unfrozen_blocks": UNFROZEN_BLOCKS,
        "head_lr": HEAD_LR,
        "backbone_lr": BACKBONE_LR,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "dataset_fingerprint": fingerprint,
        "validation_training": validation_stage,
        "full_training_history": full_head_history + full_partial_history,
        "siglip2_validation": score(valid_labels, siglip2_validation_probabilities),
        "run17_reconstructed_validation": baseline_metrics,
        "selected_validation": score(valid_labels, blended_validation_probabilities),
        "selected_temperature": selected_temperature,
        "selected_siglip2_weight": selected_weight,
        "selection_guard": "aggregate Macro-F1 gain, mean-fold non-loss, >=4/5 non-degrading folds",
        "top_blends": sorted(
            grid,
            key=lambda row: (
                row["macro_f1"],
                row["mean_fold_macro_f1"],
                -row["siglip2_weight"],
            ),
            reverse=True,
        )[:15],
        "test_changed_rows_vs_run17": int(
            (blended_predictions != baseline_predictions).sum()
        ),
        "test_class_counts": {
            str(label): int((blended_predictions == label).sum())
            for label in range(NUM_CLASSES)
        },
        "timing_seconds": {
            "training_or_resume_load": training_seconds,
            "test_inference": inference_seconds,
            "end_to_end": time.perf_counter() - started_at,
        },
        "resume_policy": "epoch-level exact resume; interrupted epoch restarts",
        "cached": False,
    }

    standalone_submission.to_csv(standalone_submission_path, index=False)
    blend_submission.to_csv(blend_submission_path, index=False)
    recommended_submission.to_csv(recommended_submission_path, index=False)
    validation_output.to_csv(validation_csv_path, index=False)
    np.savez(
        validation_probability_path,
        labels=valid_labels,
        inner_folds=inner_folds,
        run17_probabilities=baseline_validation_probabilities.astype(np.float32),
        siglip2_probabilities=siglip2_validation_probabilities.astype(np.float32),
        blended_probabilities=blended_validation_probabilities.astype(np.float32),
    )
    np.savez(
        test_probability_path,
        ids=test_ids,
        run17_probabilities=baseline_test_probabilities.astype(np.float32),
        siglip2_probabilities=siglip2_test_probabilities.astype(np.float32),
        blended_probabilities=blended_test_probabilities.astype(np.float32),
        standalone_predictions=standalone_predictions.astype(np.int64),
        predictions=blended_predictions.astype(np.int64),
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cache_volume.commit()
    return make_payload(metrics)


@app.local_entrypoint()
def main(
    output_dir: str = "artifacts/high_resolution_context",
    force: bool = False,
):
    import json

    result = train_and_infer.remote(force=force)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "submission_high_resolution_siglip2_standalone.csv").write_text(
        result["standalone_submission"], encoding="utf-8"
    )
    (destination / "submission_high_resolution_blend.csv").write_text(
        result["blend_submission"], encoding="utf-8"
    )
    (destination / "submission_high_resolution_recommended.csv").write_text(
        result["recommended_submission"], encoding="utf-8"
    )
    (destination / "validation_predictions.csv").write_text(
        result["validation_csv"], encoding="utf-8"
    )
    (destination / "validation_probabilities.npz").write_bytes(
        result["validation_probabilities"]
    )
    (destination / "test_probabilities.npz").write_bytes(
        result["test_probabilities"]
    )
    (destination / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2), encoding="utf-8"
    )
    metrics = result["metrics"]
    print("SigLIP2 validation Macro-F1:", metrics["siglip2_validation"]["macro_f1"])
    print("Run17 validation Macro-F1:", metrics["run17_reconstructed_validation"]["macro_f1"])
    print("Selected SigLIP2 weight:", metrics["selected_siglip2_weight"])
    print("Selected validation Macro-F1:", metrics["selected_validation"]["macro_f1"])
    print("Changed test rows vs Run17:", metrics["test_changed_rows_vs_run17"])
    print("Timing seconds:", metrics["timing_seconds"])
    print(f"Wrote all artifacts to {destination.resolve()}")
