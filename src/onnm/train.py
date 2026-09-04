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
from .losses import build_lesion_loss, build_loss, lesion_weight
from .metrics import compute_metrics, format_report
from .model import (
    build_model,
    build_model_for_checkpoint,
    model_summary,
    set_backbone_trainable,
)
from .thermal import build_governor
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
    name = str(cfg.train.scheduler).lower()

    if name in ("plateau", "reduce_on_plateau", "reducelronplateau"):
        # Stepped once per epoch on a monitored metric, not per batch. Useful
        # for a long run where the right schedule is not known in advance:
        # cosine commits to a fixed decay over a fixed horizon, so with early
        # stopping it may never reach its low-LR phase at all.
        plateau = cfg.train.get("plateau", None)

        def setting(key, default):
            return default if plateau is None else plateau.get(key, default)

        monitored = str(setting("metric", "val_loss"))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            # "min" for a loss, "max" for anything where higher is better.
            mode="min" if monitored.endswith("loss") else "max",
            factor=float(setting("factor", 0.5)),
            patience=int(setting("patience", 5)),
            threshold=float(setting("threshold", 1e-4)),
            min_lr=float(setting("min_lr", 1e-7)),
        )

    if name != "cosine":
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


def resolve_amp_dtype(cfg, device: torch.device) -> torch.dtype | None:
    """Choose the autocast dtype, falling back when the GPU has no bf16.

    bf16 is the project default because it shares fp32's exponent range, so it
    needs no ``GradScaler`` and cannot silently underflow a gradient. Not every
    card has it: Turing (Colab's free T4, sm_75) has none at all, and autocast
    there either refuses or emulates it at a large cost. Both failures are
    quiet and both waste an entire run, so the capability is checked up front
    rather than inferred later from a suspicious epoch time.

    Returns ``None`` when AMP is off or the device is CPU, which is what
    ``run_epoch`` treats as "run in fp32".
    """
    if not bool(cfg.train.amp) or device.type != "cuda":
        return None

    dtype = amp_dtype_from_str(cfg.train.amp_dtype)
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        logger.warning(
            "%s does not support bfloat16; falling back to float16 with a GradScaler. "
            "Set train.amp_dtype explicitly to silence this.",
            torch.cuda.get_device_name(device),
        )
        return torch.float16
    return dtype


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    amp_dtype: torch.dtype | None = None,
    grad_clip: float = 0.0,
    governor: Any = None,
    scaler: Any = None,
    lesion_criterion: Any = None,
    lesion_weight: float = 0.0,
) -> dict[str, Any]:
    """Run one pass. Training when ``optimizer`` is given, evaluation otherwise.

    ``governor`` is an optional :class:`onnm.thermal.ThermalGovernor`, polled
    once per step. It blocks the loop while the GPU is over temperature, so the
    only thing this function needs to do is call it.
    """
    is_train = optimizer is not None
    model.train(is_train)

    # The lesion head is opt-in per call, and `return_mask` is restored
    # afterwards. Leaving it True would hand a tuple to MONAI's Grad-CAM, to
    # `collect_logits` and to `predict`, all of which index the result as a
    # tensor -- so the flag is scoped to exactly the loop that wants both heads.
    want_mask = lesion_criterion is not None and lesion_weight > 0.0
    had_mask_flag = getattr(model, "return_mask", False)
    if want_mask:
        model.return_mask = True
    total_lesion_loss = 0.0

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
                output = model(images)
                if want_mask:
                    logits, mask_logits = output
                    classification_loss = criterion(logits, labels)
                    # float32 for the pixel loss: with ~1.7% positive pixels
                    # after downsampling, a bf16 mean over 4096 cells loses the
                    # signal it is meant to be measuring.
                    lesion_loss = lesion_criterion(
                        mask_logits.float(), batch["mask"].to(device, non_blocking=True).float()
                    )
                    loss = classification_loss + lesion_weight * lesion_loss
                else:
                    logits = output
                    lesion_loss = None
                    loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    # fp16 only. The scale factor keeps small gradients out of
                    # the subnormal range, and must be removed again before
                    # clipping -- clipping a scaled gradient applies the
                    # threshold at the wrong magnitude, which would either do
                    # nothing or clip everything depending on the current scale.
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                # ReduceLROnPlateau is driven per epoch from a metric, so it is
                # stepped by the caller. Stepping it here would advance its
                # patience counter once per batch and collapse the LR.
                if scheduler is not None and not isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step()
                if governor is not None:
                    governor.step()

        batch_size = labels.size(0)
        if lesion_loss is not None:
            total_lesion_loss += float(lesion_loss.detach()) * batch_size
        total_loss += float(loss.detach()) * batch_size
        n_seen += batch_size
        # softmax in fp32: under bf16 autocast the logits carry ~3 decimal
        # digits, which is plenty for argmax but not for a calibrated AUC.
        probabilities.append(
            torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
        )
        targets.append(labels.detach().cpu().numpy())

    model.return_mask = had_mask_flag

    y_prob = np.concatenate(probabilities)
    y_true = np.concatenate(targets)
    metrics = compute_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / max(n_seen, 1)
    if want_mask:
        # Reported separately from the combined loss so a run log shows whether
        # the decoder is actually learning, rather than only whether the sum fell.
        metrics["lesion_loss"] = total_lesion_loss / max(n_seen, 1)
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

    lesion_criterion = build_lesion_loss(cfg)
    if lesion_criterion is not None:
        lesion_criterion = lesion_criterion.to(device)
        logger.info("lesion loss: %s", lesion_criterion)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # Freeze *after* the optimizer is built so the backbone's parameter groups
    # exist and resume updating the moment they are unfrozen; a frozen
    # parameter simply produces no grad and is skipped by the step.
    freeze_epochs = int(cfg.train.get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0:
        counts = set_backbone_trainable(model, cfg, False)
        logger.info(
            "backbone frozen for the first %d epoch(s): %s of %s params trainable",
            freeze_epochs, f"{counts['trainable']:,}", f"{counts['total']:,}",
        )

    amp_dtype = resolve_amp_dtype(cfg, device)
    # Enabled only for fp16, so the bf16 path this project trains on locally is
    # bit-for-bit unchanged: a disabled GradScaler is a no-op wrapper.
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype is torch.float16))
    logger.info(
        "AMP: %s (GradScaler %s)",
        "off" if amp_dtype is None else str(amp_dtype).removeprefix("torch."),
        "on" if scaler.is_enabled() else "off",
    )

    monitor = str(cfg.train.early_stopping_metric)
    patience = int(cfg.train.early_stopping_patience)
    best_score = -float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []

    governor = build_governor(cfg)
    is_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(int(cfg.train.epochs)):
        started = time.time()
        if freeze_epochs > 0 and epoch == freeze_epochs:
            counts = set_backbone_trainable(model, cfg, True)
            logger.info(
                "backbone unfrozen at epoch %d: %s params trainable",
                epoch + 1, f"{counts['trainable']:,}",
            )
        # OHEM stays inert until its warmup elapses, and resets its counters
        # each epoch so the log shows what it mined in *this* epoch.
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)

        epoch_lesion_weight = (
            lesion_weight(cfg, epoch) if lesion_criterion is not None else 0.0
        )
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, scheduler,
            amp_dtype, float(cfg.train.grad_clip), governor=governor, scaler=scaler,
            lesion_criterion=lesion_criterion, lesion_weight=epoch_lesion_weight,
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, amp_dtype=amp_dtype,
            lesion_criterion=lesion_criterion, lesion_weight=epoch_lesion_weight,
        )

        score = val_metrics.get(monitor, float("nan"))
        if not np.isfinite(score):
            score = -float("inf")

        if is_plateau:
            plateau_cfg = cfg.train.get("plateau", None)
            key = "val_loss" if plateau_cfg is None else str(
                plateau_cfg.get("metric", "val_loss")
            )
            observed = (
                val_metrics["loss"] if key == "val_loss"
                else float(val_metrics.get(key.removeprefix("val_"), val_metrics["loss"]))
            )
            before = optimizer.param_groups[0]["lr"]
            scheduler.step(observed)
            after = optimizer.param_groups[0]["lr"]
            if after < before:
                logger.info("ReduceLROnPlateau: lr %.2e -> %.2e", before, after)

        malignant = val_metrics["per_class"]["malignant"]
        logger.info(
            "epoch %3d/%d | %5.1fs | loss %.4f/%.4f | ROC %.3f | PR(mal) %.3f | "
            "sens(mal) %.3f | spec(mal) %.3f | bal-acc %.3f | F1 %.3f%s%s",
            epoch + 1, int(cfg.train.epochs), time.time() - started,
            train_metrics["loss"], val_metrics["loss"],
            val_metrics["roc_auc_macro"],
            val_metrics["pr_auc"].get("malignant", float("nan")),
            malignant["sensitivity"], malignant["specificity"],
            val_metrics["balanced_accuracy"], val_metrics["f1_macro"],
            (
                f" | OHEM {criterion.n_mined}/{criterion.n_normal}"
                if getattr(criterion, "active", False) else ""
            ),
            "  <- best" if score > best_score else "",
        )
        if governor is not None and epoch % 5 == 0:
            logger.info("  GPU: %s", governor.snapshot())

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

    result: dict[str, Any] = {
        "best_score": best_score,
        "best_epoch": best_epoch + 1,
        "history": history,
        # The *effective* dtype, which is not always the configured one -- see
        # resolve_amp_dtype. A run record that says bfloat16 when fp16 was used
        # would make two runs look comparable when they are not.
        "amp_dtype": None if amp_dtype is None else str(amp_dtype).removeprefix("torch."),
        "grad_scaler": scaler.is_enabled(),
    }
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        result["peak_vram_gb"] = round(peak_gb, 2)
        result["vram_utilisation"] = round(peak_gb / total_gb, 3)
        logger.info(
            "peak VRAM: %.2f GB of %.1f GB (%.0f%%)", peak_gb, total_gb,
            100 * peak_gb / total_gb,
        )
    if governor is not None:
        result["thermal"] = governor.stats.as_dict()
        logger.info("thermal: %s", result["thermal"])
        governor.close()
    return result


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

    # Same call `train()` makes, and for the same reason -- it must happen before
    # the first conv/BN call or it has no effect.
    #
    # Without it this gate was unrunnable on the hardware it exists to check.
    # `train.miopen: false` is the workaround for a ROCm defect in the
    # training-mode BatchNorm kernel, and overfit_batch trains, so it walks
    # straight into `RuntimeError: miopenStatusUnknownError` -- while the config
    # says the workaround is enabled and every other training path honours it.
    # The failure reads as "gate 6 is broken", which is exactly backwards: the
    # gate was the only thing reporting that this path never applied the fix.
    configure_backend(bool(cfg.train.get("miopen", True)))

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

    # Load first, then build what the checkpoint says it is. Building from the
    # YAML on disk and hoping it matches is what breaks the moment a checkpoint
    # carries a lesion head -- see model.build_model_for_checkpoint.
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model_for_checkpoint(state, cfg).to(device)
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
