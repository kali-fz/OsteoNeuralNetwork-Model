"""Training loop with mixed precision, early stopping and clinical checkpointing.

Two choices here are deliberate and easy to get wrong:

**bf16 rather than fp16.** On RDNA3 (gfx1100) bf16 has the same exponent range as
fp32, so it needs no ``GradScaler`` and cannot silently underflow the small
gradients that a focal loss on a 9% class produces.

**Early stopping on malignant recall, not on validation loss.** Loss is dominated
by the 91% of images that are not malignant, so it keeps improving long after the
model has stopped getting better at the only call that matters.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .dataset import build_dataloader, build_records, class_weights
from .losses import build_loss
from .metrics import compute_metrics, format_report
from .model import build_model, model_summary
from .utils import (
    amp_dtype_from_str,
    configure_backend,
    get_logger,
    save_json,
    set_seed,
)

logger = get_logger(__name__)


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    name = str(cfg.train.optimizer).lower()
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(cfg.train.lr)
    weight_decay = float(cfg.train.weight_decay)

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg, steps_per_epoch: int):
    """Cosine schedule with linear warmup, stepped per batch.

    Warmup matters when fine-tuning a pretrained backbone: a full-rate first step
    into a randomly initialised head can wreck the pretrained features before
    they have contributed anything.
    """
    if str(cfg.train.scheduler).lower() != "cosine":
        return None

    epochs = int(cfg.train.epochs)
    warmup_steps = max(1, int(cfg.train.warmup_epochs) * steps_per_epoch)
    total_steps = max(warmup_steps + 1, epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    amp_dtype: torch.dtype | None = None,
    grad_clip: float = 0.0,
) -> dict[str, Any]:
    """Run one pass. Training when ``optimizer`` is given, evaluation otherwise."""
    is_train = optimizer is not None
    model.train(is_train)

    use_amp = amp_dtype is not None and device.type == "cuda"
    total_loss = 0.0
    n_seen = 0
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach()) * batch_size
        n_seen += batch_size
        # softmax in fp32: under bf16 autocast the logits carry ~3 decimal
        # digits, which is plenty for argmax but not for a calibrated AUC.
        probabilities.append(
            torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
        )
        targets.append(labels.detach().cpu().numpy())

    y_prob = np.concatenate(probabilities)
    y_true = np.concatenate(targets)
    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / max(n_seen, 1)
    metrics["_y_prob"] = y_prob
    metrics["_y_true"] = y_true
    return metrics


def train(cfg, output_dir: Path, device: torch.device | None = None) -> dict[str, Any]:
    """Fine-tune the configured model and return the best validation metrics."""
    from .utils import get_device

    set_seed(int(cfg.seed))
    device = device or get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Must happen before the first conv/BN call. See configure_backend for why
    # this exists; scripts/verify_env.py checks the same thing up front.
    backend = configure_backend(bool(cfg.train.get("miopen", True)))
    logger.info("compute backend: %s (cudnn/MIOpen enabled=%s)",
                backend["backend"], backend["cudnn_enabled"])

    train_records = build_records(cfg, split="train")
    val_records = build_records(cfg, split="val")

    train_loader = build_dataloader(cfg, "train", records=train_records)
    val_loader = build_dataloader(cfg, "val", records=val_records)

    model = build_model(cfg).to(device)
    logger.info("\n%s", model_summary(model, cfg))

    # alpha from the TRAIN split only -- deriving it from the full dataset would
    # leak the test set's class composition into training. `alpha_beta` tempers
    # how hard the correction pushes; see onnm.dataset.class_weights.
    alpha = class_weights(
        train_records,
        num_classes=int(cfg.model.num_classes),
        beta=float(cfg.loss.get("alpha_beta", 1.0)),
    ).to(device)
    criterion = build_loss(cfg, alpha=alpha).to(device)
    logger.info("loss: %s", criterion)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    amp_dtype = (
        amp_dtype_from_str(cfg.train.amp_dtype)
        if bool(cfg.train.amp) and device.type == "cuda"
        else None
    )

    monitor = str(cfg.train.early_stopping_metric)
    patience = int(cfg.train.early_stopping_patience)
    best_score = -float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []

    for epoch in range(int(cfg.train.epochs)):
        started = time.time()
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler,
            amp_dtype, float(cfg.train.grad_clip),
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, amp_dtype=amp_dtype)

        score = val_metrics.get(monitor, float("nan"))
        if not np.isfinite(score):
            score = -float("inf")

        malignant = val_metrics["per_class"]["malignant"]
        logger.info(
            "epoch %3d/%d | %5.1fs | loss %.4f/%.4f | ROC %.3f | PR(mal) %.3f | "
            "sens(mal) %.3f | spec(mal) %.3f | bal-acc %.3f | F1 %.3f%s",
            epoch + 1, int(cfg.train.epochs), time.time() - started,
            train_metrics["loss"], val_metrics["loss"],
            val_metrics["roc_auc_macro"],
            val_metrics["pr_auc"].get("malignant", float("nan")),
            malignant["sensitivity"], malignant["specificity"],
            val_metrics["balanced_accuracy"], val_metrics["f1_macro"],
            "  <- best" if score > best_score else "",
        )

        # Per-class specificity is recorded for normal as well as malignant:
        # "normal specificity" is the rate at which lesion films are correctly
        # not called normal, and it is the number that moves when the model
        # starts over-calling lesions.
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_roc_auc_macro": val_metrics["roc_auc_macro"],
                "val_roc_auc_malignant": val_metrics["roc_auc"].get("malignant"),
                "val_pr_auc_macro": val_metrics["pr_auc_macro"],
                "val_pr_auc_malignant": val_metrics["pr_auc"].get("malignant"),
                "val_f1_macro": val_metrics["f1_macro"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_malignant_recall": val_metrics["malignant_recall"],
                "val_malignant_specificity": malignant["specificity"],
                "val_malignant_ppv": malignant["ppv"],
                "val_normal_sensitivity": val_metrics["per_class"]["normal"]["sensitivity"],
                "val_normal_specificity": val_metrics["per_class"]["normal"]["specificity"],
                "val_normal_called_lesion": int(
                    val_metrics["confusion_matrix"][0][1]
                    + val_metrics["confusion_matrix"][0][2]
                ),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "config": cfg.to_dict(),
                    monitor: score,
                },
                output_dir / "best.pt",
            )
            logger.info("  new best %s = %.4f -> saved best.pt", monitor, score)

        if epoch - best_epoch >= patience:
            logger.info(
                "early stop: %s has not improved for %d epochs (best %.4f at epoch %d)",
                monitor, patience, best_score, best_epoch + 1,
            )
            break

    save_json(history, output_dir / "history.json")
    logger.info("best %s = %.4f at epoch %d", monitor, best_score, best_epoch + 1)
    return {"best_score": best_score, "best_epoch": best_epoch + 1, "history": history}


def overfit_batch(
    cfg,
    n_samples: int = 32,
    steps: int = 200,
    device: torch.device | None = None,
    target_accuracy: float = 0.95,
) -> dict[str, Any]:
    """Memorise a tiny batch. The cheapest way to prove the pipeline is wired up.

    A model that cannot reach ~100% training accuracy on 32 images has a broken
    pipeline, not a hard problem: labels misaligned with images, normalisation
    destroying the signal, or gradients not reaching the backbone. Two minutes
    here saves a day of staring at a loss curve that never moves.

    Augmentation is disabled -- the point is memorisation, and random crops would
    make the target legitimately unreachable.
    """
    from .utils import get_device

    set_seed(int(cfg.seed))
    device = device or get_device()

    records = build_records(cfg, split="train")
    # Stratify the toy batch so every class is represented; an all-normal sample
    # would be memorised by a constant predictor and prove nothing.
    by_class: dict[int, list] = {}
    for record in records:
        by_class.setdefault(record["label"], []).append(record)

    per_class = max(1, n_samples // max(1, len(by_class)))
    subset = [r for group in by_class.values() for r in group[:per_class]]

    loader = build_dataloader(cfg, "val", records=subset, shuffle=False)  # val => no augmentation
    batches = [
        {"image": b["image"].to(device), "label": b["label"].to(device)} for b in loader
    ]

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg, alpha=None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    model.train()
    accuracy = 0.0
    losses: list[float] = []

    for step in range(steps):
        correct = total = 0
        step_loss = 0.0
        for batch in batches:
            logits = model(batch["image"])
            loss = criterion(logits, batch["label"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step_loss += float(loss.detach()) * batch["label"].size(0)
            correct += int((logits.argmax(1) == batch["label"]).sum())
            total += batch["label"].size(0)

        accuracy = correct / max(total, 1)
        losses.append(step_loss / max(total, 1))
        if step % 20 == 0 or accuracy >= target_accuracy:
            logger.info("step %3d | loss %.4f | train acc %.3f", step, losses[-1], accuracy)
        if accuracy >= target_accuracy:
            break

    return {
        "n_samples": sum(b["label"].size(0) for b in batches),
        "steps_run": len(losses),
        "final_accuracy": accuracy,
        "final_loss": losses[-1] if losses else float("nan"),
        "passed": accuracy >= target_accuracy,
    }


def evaluate(cfg, checkpoint: Path, split: str = "test", device: torch.device | None = None):
    """Evaluate a saved checkpoint on one split, with bootstrap CIs.

    Applies the calibration fitted by ``scripts/calibrate.py`` when one exists
    beside the checkpoint. The metrics are then reported at the operating point
    the model would actually be used at, rather than at an argmax that no
    deployment uses.
    """
    from .calibrate import Calibration, lesion_scores
    from .metrics import threshold_for_sensitivity, with_confidence_intervals
    from .utils import get_device

    device = device or get_device()
    records = build_records(cfg, split=split)
    loader = build_dataloader(cfg, split, records=records, shuffle=False)

    model = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    criterion = build_loss(cfg, alpha=None).to(device)
    metrics = run_epoch(model, loader, criterion, device)

    y_true, y_prob = metrics.pop("_y_true"), metrics.pop("_y_prob")

    calibration = Calibration.for_checkpoint(checkpoint)
    if calibration is not None and calibration.temperature != 1.0:
        # Re-softmax at the fitted temperature. Monotone, so the confusion
        # matrix above is untouched; only the probability-valued metrics move.
        logits = np.log(np.clip(y_prob, 1e-12, 1.0))
        y_prob = torch.softmax(
            torch.as_tensor(logits / calibration.temperature), dim=1
        ).numpy()
        logger.info("applied fitted temperature T=%.4f", calibration.temperature)

    cis = with_confidence_intervals(
        y_true, y_prob,
        n_boot=int(cfg.eval.bootstrap_n),
        alpha=float(cfg.eval.bootstrap_alpha),
    )
    operating_point = threshold_for_sensitivity(
        y_true, y_prob, target=float(cfg.eval.target_sensitivity)
    )

    print(format_report(metrics, cis))

    result = {
        "metrics": metrics,
        "confidence_intervals": cis,
        "operating_point": operating_point,
    }

    if calibration is None:
        print(
            "\n".join(
                [
                    "",
                    "  Note: no calibration.json beside this checkpoint. Probabilities",
                    "  are uncalibrated and the binary decision falls back to a naive",
                    "  0.50 cut. Fit one with:",
                    f"    python scripts/calibrate.py --checkpoint {checkpoint}",
                ]
            )
        )
        return result

    # Report the binary normal-vs-lesion decision at the calibrated threshold --
    # this is the number the app shows, and it is not visible in the 3-way
    # confusion matrix above.
    scores = lesion_scores(y_prob, normal_index=0)
    predicted_lesion = scores >= calibration.lesion_threshold
    actually_lesion = y_true != 0
    binary = {
        "threshold": calibration.lesion_threshold,
        "temperature": calibration.temperature,
        "fitted_on": calibration.fitted_on,
        "sensitivity": float(predicted_lesion[actually_lesion].mean())
        if actually_lesion.any() else float("nan"),
        "specificity": float((~predicted_lesion[~actually_lesion]).mean())
        if (~actually_lesion).any() else float("nan"),
        "false_positives": int((predicted_lesion & ~actually_lesion).sum()),
        "n_normal": int((~actually_lesion).sum()),
    }
    print(
        "\n".join(
            [
                "",
                f"  Binary decision at the calibrated threshold "
                f"({binary['threshold']:.3f}, fitted on {binary['fitted_on']}):",
                f"    sensitivity   {binary['sensitivity']:.3f}",
                f"    specificity   {binary['specificity']:.3f}",
                f"    normal films called lesion: "
                f"{binary['false_positives']} / {binary['n_normal']}",
            ]
        )
    )
    result["binary_decision"] = binary
    return result
