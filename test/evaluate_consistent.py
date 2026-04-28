#!/usr/bin/env python3
"""Experiment-side evaluation with f0-consistent voiced targets."""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAINING_DIR = os.path.join(ROOT, "training")
sys.path.insert(0, TRAINING_DIR)

import evaluate as base_evaluate


def evaluate_model_consistent(model, test_path, device="cpu"):
    """Mirror training/evaluate.py, but use f0 > 0 for voiced accuracy."""
    model.eval()
    test = np.load(test_path)

    clips = test["clips"]
    f0_gt = test["f0"]
    snrs = test["snr"]
    results = []

    for i in base_evaluate.tqdm(range(clips.shape[0]), desc="Evaluating", unit="clip"):
        mel = torch.from_numpy(clips[i].astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            v, p, _ = model(mel)

        pred_vad = v.squeeze(0).cpu().numpy().squeeze(-1)
        pred_pitch = p.squeeze(0).cpu().numpy()
        T = pred_vad.shape[0]

        f0_ref = f0_gt[i, :T].astype(np.float32)
        voiced_ref = (f0_ref > 0).astype(np.float32)
        f0_offline = base_evaluate.viterbi_decode(pred_pitch)
        f0_realtime = base_evaluate.viterbi_decode_realtime(pred_pitch)

        row = {
            "clip": i,
            "snr": float(snrs[i]),
            "vad_acc": float(np.mean((pred_vad > 0.5) == (voiced_ref > 0.5))),
        }
        for prefix, f0_dec in [("offline", f0_offline), ("realtime", f0_realtime)]:
            metrics = base_evaluate._pitch_metrics(f0_dec, f0_ref)
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser(description="NanoPitch evaluation (f0-consistent)")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--csv", default=None, help="Save per-clip CSV")
    parser.add_argument("--json", default=None, help="Save summary JSON")
    args = parser.parse_args()

    warnings.warn(
        "Loading checkpoint via torch.load() executes Python deserialization. "
        "Only evaluate checkpoints from trusted sources.",
        RuntimeWarning,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kwargs = ckpt.get("model_kwargs", {"cond_size": 64, "gru_size": 96})
    model = base_evaluate.NanoPitch(**kwargs)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    test_path = os.path.join(args.data_dir, "test.npz")
    t0 = time.time()
    results = evaluate_model_consistent(model, test_path, device=args.device)
    dt = time.time() - t0

    print("  [eval] using f0 > 0 as voiced reference in test/evaluate_consistent.py")
    base_evaluate.print_report(results)
    print(f"  Evaluated in {dt:.1f}s")

    if args.csv:
        base_evaluate.save_csv(results, args.csv)
    if args.json:
        base_evaluate.save_json(results, args.json)


if __name__ == "__main__":
    main()
