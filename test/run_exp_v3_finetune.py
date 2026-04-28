#!/usr/bin/env python3
"""One-off fine-tuning run starting from the best exp1 checkpoint."""

from pathlib import Path
import argparse
import json
import os
import sys
import time
import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def find_project_root(start=None):
    path = Path.cwd() if start is None else Path(start)
    for candidate in [path, *path.parents]:
        if (candidate / "training" / "train.py").exists():
            return candidate
    raise FileNotFoundError("Could not find NanoPitch project root")


def build_parser():
    parser = argparse.ArgumentParser(description="Fine-tune the best exp1 checkpoint into exp_v3_1.")
    parser.add_argument("--source-checkpoint", default="test/runs/exp1/checkpoints/best.pth")
    parser.add_argument("--output-dir", default="test/runs/exp_v3_1")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--save-best-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def score_results(results):
    def mean_metric(key):
        vals = [row[key] for row in results if row[key] == row[key]]
        return sum(vals) / len(vals) if vals else float("nan")
    offline_rpa = mean_metric("offline_rpa")
    realtime_rpa = mean_metric("realtime_rpa")
    return {"offline_rpa": offline_rpa, "realtime_rpa": realtime_rpa, "score": (offline_rpa + realtime_rpa) / 2.0}


def write_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def main():
    args = build_parser().parse_args()
    root = find_project_root()
    sys.path.insert(0, str(root / "training"))
    sys.path.insert(0, str(root / "test"))
    import train as train_module
    import evaluate as eval_module

    data_dir = (root / args.data_dir).resolve() if not os.path.isabs(args.data_dir) else Path(args.data_dir)
    source_checkpoint = (root / args.source_checkpoint).resolve() if not os.path.isabs(args.source_checkpoint) else Path(args.source_checkpoint)
    output_dir = (root / args.output_dir).resolve() if not os.path.isabs(args.output_dir) else Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    warnings.warn(
        "Loading checkpoint via torch.load() executes Python deserialization. "
        "Only use checkpoints from trusted sources.",
        RuntimeWarning,
    )
    ckpt = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model_kwargs = ckpt.get("model_kwargs", {"cond_size": 64, "gru_size": 96})
    model = train_module.NanoPitch(**model_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    device = torch.device(args.device)
    model.to(device)

    def no_noise_augment(mel_clean, mel_noise, snr_range, device):
        return mel_clean
    train_module.augment_mel_batch = no_noise_augment

    dataset = train_module.NanoPitchDataset(str(data_dir), seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True,
                            pin_memory=(device.type == "cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.98), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    best_score = float("-inf")
    best_epoch = None
    best_summary = None
    history = []
    train_args = argparse.Namespace(snr_range=[-5.0, 20.0], w_vad=0.1, w_pitch=1.0)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        train_loss = train_module.train_one_epoch(model, dataloader, optimizer, scheduler, writer, epoch, device, train_args)
        dt = time.time() - t0
        eval_summary = None
        if epoch % args.eval_interval == 0 or epoch == start_epoch:
            results = eval_module.evaluate_model(model=model, test_path=os.path.join(data_dir, "test.npz"), device=device)
            eval_summary = score_results(results)
            writer.add_scalar("eval/overall_offline_rpa", eval_summary["offline_rpa"], epoch)
            writer.add_scalar("eval/overall_realtime_rpa", eval_summary["realtime_rpa"], epoch)
            writer.add_scalar("eval/overall_score", eval_summary["score"], epoch)
        history.append({"epoch": epoch, "train_loss": train_loss, "eval": eval_summary})
        model_ckpt = {"epoch": epoch, "state_dict": model.state_dict(), "model_kwargs": model_kwargs, "loss": train_loss}
        if not args.save_best_only:
            torch.save(model_ckpt, ckpt_dir / f"epoch_{epoch:03d}.pth")
        if eval_summary is not None and eval_summary["score"] > best_score:
            best_score = eval_summary["score"]
            best_epoch = epoch
            best_summary = eval_summary
            torch.save(model_ckpt, ckpt_dir / "best.pth")
        elif best_epoch is None:
            best_epoch = epoch
            torch.save(model_ckpt, ckpt_dir / "best.pth")
        print(f"  Epoch {epoch} done in {dt:.1f}s, loss={train_loss:.5f}")

    writer.close()
    run_summary = {
        "name": output_dir.name,
        "source_checkpoint": str(source_checkpoint),
        "model_kwargs": model_kwargs,
        "epochs_requested": args.epochs,
        "epochs_started_from": start_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seq_len": args.seq_len,
        "num_workers": args.num_workers,
        "augmentation": "none_clean_only",
        "best_epoch": best_epoch,
        "best_eval": best_summary,
        "history": history,
    }
    write_json(output_dir / "run_summary.json", run_summary)


if __name__ == "__main__":
    main()
