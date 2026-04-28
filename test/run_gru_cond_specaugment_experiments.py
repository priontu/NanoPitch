#!/usr/bin/env python3
"""Run GRU/conditioning-size SpecAugment experiments from the terminal."""

from pathlib import Path
import argparse
import csv
import json
import math
import subprocess
import sys
from statistics import mean


BASE_EXPERIMENTS = [
    {"name": "small", "cond_size": 32, "gru_size": 64},
    {"name": "baseline", "cond_size": 32, "gru_size": 96},
    {"name": "large", "cond_size": 32, "gru_size": 128},
    {"name": "small", "cond_size": 64, "gru_size": 64},
    {"name": "baseline", "cond_size": 64, "gru_size": 96},
    {"name": "large", "cond_size": 64, "gru_size": 128},
    {"name": "small", "cond_size": 96, "gru_size": 64},
    {"name": "baseline", "cond_size": 96, "gru_size": 96},
    {"name": "large", "cond_size": 96, "gru_size": 128},
]

SPEC_AUGMENT_BANDS = [1, 2, 3, 4, 5]


def find_project_root(start=None):
    path = Path.cwd() if start is None else Path(start)
    for candidate in [path, *path.parents]:
        if (candidate / "training" / "train.py").exists():
            return candidate
    raise FileNotFoundError("Could not find NanoPitch project root")


def build_experiments():
    experiments = []
    exp_num = 1
    for band in SPEC_AUGMENT_BANDS:
        for base in BASE_EXPERIMENTS:
            experiments.append({
                **base,
                "exp_num": exp_num,
                "name": f"{base['name']}_specband{band}",
                "specaugment_band": band,
            })
            exp_num += 1
    return experiments


def run_cmd(cmd, cwd):
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=str(cwd), check=True)


def as_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def mean_ignore_nan(values):
    clean = [value for value in values if math.isfinite(value)]
    return mean(clean) if clean else None


def load_eval_rows(csv_path):
    with open(csv_path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def write_overall_json(csv_path, overall_path, experiment, checkpoint_path, args):
    rows = load_eval_rows(csv_path)
    metric_fields = [
        "vad_acc", "offline_vdr", "offline_rpa", "offline_rca",
        "offline_gross_err", "offline_median_cents",
        "realtime_vdr", "realtime_rpa", "realtime_rca",
        "realtime_gross_err", "realtime_median_cents",
    ]
    overall = {
        "exp_num": experiment["exp_num"],
        "name": experiment["name"],
        "cond_size": experiment["cond_size"],
        "gru_size": experiment["gru_size"],
        "specaugment_band": experiment["specaugment_band"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seq_len": args.seq_len,
        "n_clips": len(rows),
        "checkpoint": str(checkpoint_path),
        "train_device": args.train_device,
        "eval_device": args.eval_device,
    }
    for field in metric_fields:
        overall[field] = mean_ignore_nan([as_float(row.get(field)) for row in rows])
    write_json(overall_path, overall)
    return overall


def save_experiment_config(experiment, exp_dir, args):
    config = {
        **experiment,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seq_len": args.seq_len,
        "num_workers": args.num_workers,
        "train_device": args.train_device,
        "eval_device": args.eval_device,
        "snr_range": list(args.snr_range),
        "w_vad": args.w_vad,
        "w_pitch": args.w_pitch,
        "voiced_definition": "f0_positive" if args.use_f0_voicing else "dataset_vad",
        "eval_script": "test/evaluate_consistent.py" if args.use_f0_voicing else "training/evaluate.py",
    }
    write_json(exp_dir / "experiment_config.json", config)


def run_single_experiment(experiment, paths, args):
    exp_dir = paths["runs_dir"] / f"exp_{experiment['exp_num']}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(experiment, exp_dir, args)

    checkpoint = exp_dir / "checkpoints" / "best.pth"
    eval_csv = exp_dir / "eval.csv"
    eval_json = exp_dir / "eval.json"
    overall_json = exp_dir / "overall.json"

    print("\n" + "=" * 80, flush=True)
    print(
        f"Experiment {experiment['exp_num']}: {experiment['name']} "
        f"(cond_size={experiment['cond_size']}, gru_size={experiment['gru_size']}, "
        f"specaugment_band={experiment['specaugment_band']}) on {args.train_device}",
        flush=True,
    )
    print("=" * 80, flush=True)

    if args.force_retrain or not checkpoint.exists():
        train_cmd = [
            sys.executable, paths["train_script"],
            "--specaugment-band", experiment["specaugment_band"],
            "--data-dir", paths["data_dir"],
            "--output-dir", exp_dir,
            "--device", args.train_device,
            "--cond-size", experiment["cond_size"],
            "--gru-size", experiment["gru_size"],
            "--epochs", args.epochs,
            "--batch-size", args.batch_size,
            "--lr", args.lr,
            "--seq-len", args.seq_len,
            "--num-workers", args.num_workers,
            "--snr-range", args.snr_range[0], args.snr_range[1],
            "--w-vad", args.w_vad,
            "--w-pitch", args.w_pitch,
        ]
        if args.use_f0_voicing:
            train_cmd.append("--use-f0-voicing")
        if args.skip_train_eval:
            train_cmd.append("--skip-train-eval")
        if args.save_best_only:
            train_cmd.append("--save-best-only")
        run_cmd(train_cmd, cwd=paths["root"])
    else:
        print(f"Skipping training because checkpoint already exists: {checkpoint}")

    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected checkpoint was not created: {checkpoint}")

    if args.force_reevaluate or not (eval_csv.exists() and eval_json.exists()):
        eval_cmd = [
            sys.executable, paths["eval_script"],
            "--checkpoint", checkpoint,
            "--data-dir", paths["data_dir"],
            "--device", args.eval_device,
            "--csv", eval_csv,
            "--json", eval_json,
        ]
        run_cmd(eval_cmd, cwd=paths["root"])
    else:
        print(f"Skipping evaluation because outputs already exist: {eval_csv}, {eval_json}")

    overall = write_overall_json(eval_csv, overall_json, experiment, checkpoint, args)
    print(f"Saved overall summary: {overall_json}", flush=True)
    return overall


def parse_args():
    parser = argparse.ArgumentParser(description="Run NanoPitch SpecAugment sweep.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--snr-range", type=float, nargs=2, default=(-5.0, 20.0))
    parser.add_argument("--w-vad", type=float, default=0.1)
    parser.add_argument("--w-pitch", type=float, default=1.0)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--force-retrain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-reevaluate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-train-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-best-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-f0-voicing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.eval_device is None:
        args.eval_device = args.train_device
    return args


def main():
    args = parse_args()
    root = find_project_root()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else root / "data"
    runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else root / "test" / "runs"
    paths = {
        "root": root,
        "data_dir": data_dir,
        "runs_dir": runs_dir,
        "train_script": root / "test" / "specaugment_train.py",
        "eval_script": root / "test" / "evaluate_consistent.py" if args.use_f0_voicing else root / "training" / "evaluate.py",
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments()
    for experiment in experiments:
        run_single_experiment(experiment, paths, args)


if __name__ == "__main__":
    main()
