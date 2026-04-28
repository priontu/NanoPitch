#!/usr/bin/env python3
"""Run a focused v2 sweep with faster defaults and curriculum training."""

from pathlib import Path
import argparse
import csv
import json
import math
import subprocess
import sys
from statistics import mean


def find_project_root(start=None):
    path = Path.cwd() if start is None else Path(start)
    for candidate in [path, *path.parents]:
        if (candidate / "training" / "train.py").exists():
            return candidate
    raise FileNotFoundError("Could not find NanoPitch project root")


def run_cmd(cmd, cwd):
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=str(cwd), check=True)


def pick_python_cmd():
    candidates = [
        [sys.executable],
        ["conda", "run", "-n", "torch_it", "python"],
        ["/home/DREXEL/pc833/miniconda3/envs/torch_it/bin/python"],
    ]
    for candidate in candidates:
        try:
            subprocess.run([*candidate, "-c", "import numpy, torch"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return candidate
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    raise RuntimeError("Could not find a Python interpreter with numpy and torch.")


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


def load_completed_summaries(runs_dir):
    summaries = []
    for path in sorted(runs_dir.glob("exp_*/overall.json")):
        with open(path) as handle:
            item = json.load(handle)
        item["score"] = (item["offline_rpa"] + item["realtime_rpa"]) / 2.0
        summaries.append(item)
    return summaries


def pick_v2_experiments(runs_dir):
    summaries = load_completed_summaries(runs_dir)
    if not summaries:
        raise FileNotFoundError("No completed exp_*/overall.json summaries were found.")
    exact_baseline = [item for item in summaries if item["cond_size"] == 64 and item["gru_size"] == 96]
    baseline = max(exact_baseline, key=lambda item: (item["score"], -item["realtime_median_cents"]))
    ranked = sorted(summaries, key=lambda item: (-item["score"], item["realtime_median_cents"], item["exp_num"]))
    selected = []
    baseline_key = (baseline["cond_size"], baseline["gru_size"], baseline["specaugment_band"])
    for item in ranked:
        key = (item["cond_size"], item["gru_size"], item["specaugment_band"])
        if key == baseline_key:
            continue
        selected.append(item)
        if len(selected) == 2:
            break
    experiments = []
    for exp_num, source in enumerate([selected[0], selected[1], baseline], start=1):
        tag = "baseline" if source is baseline else "top"
        experiments.append({
            "exp_num": exp_num,
            "name": f"{tag}_{source['name']}_v2",
            "source_exp_num": source["exp_num"],
            "source_name": source["name"],
            "source_score": source["score"],
            "cond_size": source["cond_size"],
            "gru_size": source["gru_size"],
            "specaugment_band": source["specaugment_band"],
        })
    return experiments


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
        "source_exp_num": experiment["source_exp_num"],
        "source_name": experiment["source_name"],
        "source_score": experiment["source_score"],
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
        "time_mask": args.time_mask,
        "early_stopping_patience": args.early_stopping_patience,
        "curriculum_start_min_snr": args.curriculum_start_min_snr,
        "curriculum_step_epochs": args.curriculum_step_epochs,
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
        "use_f0_voicing": args.use_f0_voicing,
        "time_mask": args.time_mask,
        "min_lr": args.min_lr,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "curriculum_start_min_snr": args.curriculum_start_min_snr,
        "curriculum_step_epochs": args.curriculum_step_epochs,
        "eval_script": "test/evaluate_consistent.py",
        "train_script": "test/specaugment_train_v2.py",
    }
    write_json(exp_dir / "experiment_config.json", config)


def run_single_experiment(experiment, paths, args):
    exp_dir = paths["runs_dir"] / f"exp_v2_{experiment['exp_num']}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(experiment, exp_dir, args)
    checkpoint = exp_dir / "checkpoints" / "best.pth"
    eval_csv = exp_dir / "eval.csv"
    eval_json = exp_dir / "eval.json"
    overall_json = exp_dir / "overall.json"
    training_summary_json = exp_dir / "training_summary.json"
    if args.force_retrain or not checkpoint.exists():
        train_cmd = [
            *paths["python_cmd"], paths["train_script"],
            "--specaugment-band", experiment["specaugment_band"],
            "--time-mask", args.time_mask,
            "--use-f0-voicing",
            "--save-best-only",
            "--min-lr", args.min_lr,
            "--eval-interval", args.eval_interval,
            "--early-stopping-patience", args.early_stopping_patience,
            "--early-stopping-min-delta", args.early_stopping_min_delta,
            "--curriculum-step-epochs", args.curriculum_step_epochs,
            "--curriculum-start-min-snr", args.curriculum_start_min_snr,
            "--summary-json", training_summary_json,
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
        run_cmd(train_cmd, cwd=paths["root"])
    if args.force_reevaluate or not (eval_csv.exists() and eval_json.exists()):
        eval_cmd = [
            *paths["python_cmd"], paths["eval_script"],
            "--checkpoint", checkpoint,
            "--data-dir", paths["data_dir"],
            "--device", args.eval_device,
            "--csv", eval_csv,
            "--json", eval_json,
        ]
        run_cmd(eval_cmd, cwd=paths["root"])
    return write_overall_json(eval_csv, overall_json, experiment, checkpoint, args)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the focused NanoPitch v2 sweep.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--min-lr", type=float, default=2e-4)
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--snr-range", type=float, nargs=2, default=(-5.0, 20.0))
    parser.add_argument("--curriculum-start-min-snr", type=float, default=10.0)
    parser.add_argument("--curriculum-step-epochs", type=int, default=20)
    parser.add_argument("--w-vad", type=float, default=0.1)
    parser.add_argument("--w-pitch", type=float, default=1.5)
    parser.add_argument("--time-mask", type=int, default=12)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--force-retrain", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-reevaluate", action=argparse.BooleanOptionalAction, default=True)
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
        "train_script": root / "test" / "specaugment_train_v2.py",
        "eval_script": root / "test" / "evaluate_consistent.py",
        "python_cmd": pick_python_cmd(),
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    experiments = pick_v2_experiments(runs_dir)
    for experiment in experiments:
        run_single_experiment(experiment, paths, args)


if __name__ == "__main__":
    main()
