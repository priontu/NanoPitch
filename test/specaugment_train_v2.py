#!/usr/bin/env python3
"""Run a faster experiment-side NanoPitch training loop with curriculum."""

import argparse
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def parse_wrapper_args(argv):
    parser = argparse.ArgumentParser(
        add_help=True,
        description="NanoPitch v2 training wrapper with curriculum and early stopping.",
    )
    parser.add_argument("--specaugment-band", type=int, required=True)
    parser.add_argument("--time-mask", type=int, default=12)
    parser.add_argument("--use-f0-voicing", action="store_true")
    parser.add_argument("--skip-train-eval", action="store_true")
    parser.add_argument("--save-best-only", action="store_true")
    parser.add_argument("--min-lr", type=float, default=2e-4)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--curriculum-step-epochs", type=int, default=20)
    parser.add_argument("--curriculum-start-min-snr", type=float, default=10.0)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_known_args(argv)


def apply_frequency_mask(mel, band):
    if band <= 0:
        return mel
    n_mels = mel.size(-1)
    band = min(int(band), n_mels)
    batch_size = mel.size(0)
    starts = torch.randint(0, n_mels - band + 1, (batch_size,), device=mel.device)
    augmented = mel.clone()
    fill = mel.mean(dim=(1, 2), keepdim=True)
    for batch_idx, start in enumerate(starts.tolist()):
        augmented[batch_idx, :, start:start + band] = fill[batch_idx]
    return augmented


def apply_time_mask(mel, width):
    if width <= 0:
        return mel
    seq_len = mel.size(1)
    width = min(int(width), seq_len)
    batch_size = mel.size(0)
    starts = torch.randint(0, seq_len - width + 1, (batch_size,), device=mel.device)
    augmented = mel.clone()
    fill = mel.mean(dim=(1, 2), keepdim=True)
    for batch_idx, start in enumerate(starts.tolist()):
        augmented[batch_idx, start:start + width, :] = fill[batch_idx]
    return augmented


def curriculum_snr_range(epoch, total_epochs, target_range, start_min_snr, step_epochs):
    target_min, target_max = float(target_range[0]), float(target_range[1])
    if step_epochs <= 0 or total_epochs <= step_epochs:
        return [target_min, target_max]
    n_stages = max(1, math.ceil(total_epochs / step_epochs))
    stage_idx = min((epoch - 1) // step_epochs, n_stages - 1)
    progress = 1.0 if n_stages == 1 else stage_idx / (n_stages - 1)
    current_min = start_min_snr + progress * (target_min - start_min_snr)
    return [float(current_min), target_max]


def summarize_results(results):
    def sm(key):
        values = [row[key] for row in results if np.isfinite(row[key])]
        return float(np.mean(values)) if values else float("nan")
    summary = {
        "vad_acc": sm("vad_acc"),
        "offline_rpa": sm("offline_rpa"),
        "offline_rca": sm("offline_rca"),
        "offline_median_cents": sm("offline_median_cents"),
        "realtime_rpa": sm("realtime_rpa"),
        "realtime_rca": sm("realtime_rca"),
        "realtime_median_cents": sm("realtime_median_cents"),
    }
    summary["score"] = float(np.nanmean([summary["offline_rpa"], summary["realtime_rpa"]]))
    return summary


def save_summary(path, payload):
    if path is None:
        return
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def main():
    wrapper_args, remaining = parse_wrapper_args(sys.argv[1:])
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(root, "training"))
    sys.path.insert(0, os.path.join(root, "test"))

    import train as train_module
    import evaluate_consistent

    args = train_module.parser.parse_args(remaining)
    if not wrapper_args.use_f0_voicing:
        wrapper_args.use_f0_voicing = True

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    model = train_module.NanoPitch(cond_size=args.cond_size, gru_size=args.gru_size)
    start_epoch = 1
    if args.resume:
        warnings.warn(
            "Loading checkpoint via torch.load() executes Python deserialization. "
            "Only use checkpoints from trusted sources.",
            RuntimeWarning,
        )
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
    model.to(device)

    dataset = train_module.NanoPitchDataset(data_dir, seq_len=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.98), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=wrapper_args.min_lr
    )
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))

    original_augment = train_module.augment_mel_batch
    original_getitem = train_module.NanoPitchDataset.__getitem__
    if wrapper_args.use_f0_voicing:
        def getitem_with_f0_voicing(self, idx):
            mel_clean, mel_noise, _vad, f0 = original_getitem(self, idx)
            voiced = (np.asarray(f0) > 0).astype(np.float32)
            return mel_clean, mel_noise, voiced, f0
        train_module.NanoPitchDataset.__getitem__ = getitem_with_f0_voicing

    current_snr_range = list(args.snr_range)

    def augment_with_curriculum(mel_clean, mel_noise, _snr_range, _device):
        mel_mix = original_augment(mel_clean, mel_noise, current_snr_range, _device)
        mel_mix = apply_frequency_mask(mel_mix, wrapper_args.specaugment_band)
        mel_mix = apply_time_mask(mel_mix, wrapper_args.time_mask)
        return mel_mix

    train_module.augment_mel_batch = augment_with_curriculum

    best_train_loss = float("inf")
    best_eval_score = float("-inf")
    best_epoch = None
    last_improvement_epoch = None
    stopped_early = False
    latest_eval_summary = None

    for epoch in range(start_epoch, start_epoch + args.epochs):
        current_snr_range = curriculum_snr_range(
            epoch=epoch - start_epoch + 1,
            total_epochs=args.epochs,
            target_range=args.snr_range,
            start_min_snr=wrapper_args.curriculum_start_min_snr,
            step_epochs=wrapper_args.curriculum_step_epochs,
        )
        writer.add_scalar("train/snr_min", current_snr_range[0], epoch)
        writer.add_scalar("train/snr_max", current_snr_range[1], epoch)
        t0 = time.time()
        train_loss = train_module.train_one_epoch(
            model, dataloader, optimizer, scheduler, writer, epoch, device, args
        )
        scheduler.step()
        dt = time.time() - t0
        print(
            f"  Epoch {epoch} done in {dt:.1f}s, loss={train_loss:.5f}, "
            f"snr_range=[{current_snr_range[0]:.2f}, {current_snr_range[1]:.2f}], "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "model_kwargs": {"cond_size": args.cond_size, "gru_size": args.gru_size},
            "loss": train_loss,
        }
        if not wrapper_args.save_best_only:
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
        if train_loss < best_train_loss:
            best_train_loss = train_loss

        should_eval = (epoch - start_epoch + 1) % wrapper_args.eval_interval == 0 or epoch == start_epoch
        if should_eval and not wrapper_args.skip_train_eval:
            test_path = os.path.join(data_dir, "test.npz")
            results = evaluate_consistent.evaluate_model_consistent(model, test_path, device=device)
            latest_eval_summary = summarize_results(results)
            writer.add_scalar("eval/overall_offline_rpa", latest_eval_summary["offline_rpa"], epoch)
            writer.add_scalar("eval/overall_realtime_rpa", latest_eval_summary["realtime_rpa"], epoch)
            writer.add_scalar("eval/overall_score", latest_eval_summary["score"], epoch)
            improved = latest_eval_summary["score"] > best_eval_score + wrapper_args.early_stopping_min_delta
            if improved:
                best_eval_score = latest_eval_summary["score"]
                best_epoch = epoch
                last_improvement_epoch = epoch
                torch.save(ckpt, os.path.join(ckpt_dir, "best.pth"))
            elif last_improvement_epoch is None:
                last_improvement_epoch = epoch
            if (epoch - last_improvement_epoch) >= wrapper_args.early_stopping_patience:
                stopped_early = True
                break
        elif not os.path.exists(os.path.join(ckpt_dir, "best.pth")):
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pth"))
            best_epoch = epoch

    writer.close()
    summary = {
        "epochs_requested": args.epochs,
        "stop_epoch": epoch,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "best_train_loss": best_train_loss,
        "best_eval_score": best_eval_score if math.isfinite(best_eval_score) else None,
        "latest_eval_summary": latest_eval_summary,
        "final_snr_range": current_snr_range,
        "time_mask": wrapper_args.time_mask,
        "specaugment_band": wrapper_args.specaugment_band,
        "scheduler": "CosineAnnealingLR",
        "min_lr": wrapper_args.min_lr,
        "use_f0_voicing": wrapper_args.use_f0_voicing,
    }
    save_summary(wrapper_args.summary_json, summary)


if __name__ == "__main__":
    main()
