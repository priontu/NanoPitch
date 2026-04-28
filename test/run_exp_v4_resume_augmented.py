#!/usr/bin/env python3
"""Resume from exp1 and train with the full v2 augmentation stack."""

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


def write_overall_json(csv_path, overall_path, checkpoint_path, args, model_kwargs):
    rows = load_eval_rows(csv_path)
    metric_fields = [
        "vad_acc", "offline_vdr", "offline_rpa", "offline_rca",
        "offline_gross_err", "offline_median_cents",
        "realtime_vdr", "realtime_rpa", "realtime_rca",
        "realtime_gross_err", "realtime_median_cents",
    ]
    overall = {
        "exp_num": 1,
        "name": "exp_v4_1",
        "source_checkpoint": str(args.source_checkpoint),
        "cond_size": model_kwargs["cond_size"],
        "gru_size": model_kwargs["gru_size"],
        "specaugment_band": args.specaugment_band,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "seq_len": args.seq_len,
        "n_clips": len(rows),
        "checkpoint": str(checkpoint_path),
        "train_device": args.train_device,
        "eval_device": args.eval_device,
        "time_mask": args.time_mask,
        "curriculum_start_min_snr": args.curriculum_start_min_snr,
        "curriculum_step_epochs": args.curriculum_step_epochs,
        "w_vad": args.w_vad,
        "w_pitch": args.w_pitch,
    }
    for field in metric_fields:
        overall[field] = mean_ignore_nan([as_float(row.get(field)) for row in rows])
    write_json(overall_path, overall)
    return overall


def load_model_kwargs(source_checkpoint, python_cmd, root):
    cmd = [
        *python_cmd, "-c",
        ("import json, torch; "
         f"ckpt=torch.load(r'''{source_checkpoint}''', map_location='cpu', weights_only=False); "
         "print(json.dumps(ckpt.get('model_kwargs', {'cond_size':64,'gru_size':96})))")
    ]
    result = subprocess.run(cmd, cwd=str(root), check=True, capture_output=True, text=True)
    return json.loads(result.stdout.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="Resume exp1 with the full augmented v2 recipe as exp_v4_1.")
    parser.add_argument("--source-checkpoint", default="test/runs/exp1/checkpoints/best.pth")
    parser.add_argument("--output-dir", default="test/runs/exp_v4_1")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--train-device", default="cuda")
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--snr-range", type=float, nargs=2, default=(-5.0, 20.0))
    parser.add_argument("--curriculum-start-min-snr", type=float, default=10.0)
    parser.add_argument("--curriculum-step-epochs", type=int, default=20)
    parser.add_argument("--w-vad", type=float, default=0.1)
    parser.add_argument("--w-pitch", type=float, default=1.5)
    parser.add_argument("--specaugment-band", type=int, default=2)
    parser.add_argument("--time-mask", type=int, default=12)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--force-retrain", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-reevaluate", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.eval_device is None:
        args.eval_device = args.train_device
    return args


def main():
    args = parse_args()
    root = find_project_root()
    python_cmd = pick_python_cmd()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else root / "data"
    output_dir = Path(args.output_dir).resolve() if Path(args.output_dir).is_absolute() else root / args.output_dir
    source_checkpoint = Path(args.source_checkpoint).resolve() if Path(args.source_checkpoint).is_absolute() else root / args.source_checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoints" / "best.pth"
    eval_csv = output_dir / "eval.csv"
    eval_json = output_dir / "eval.json"
    overall_json = output_dir / "overall.json"
    training_summary_json = output_dir / "training_summary.json"
    model_kwargs = load_model_kwargs(source_checkpoint, python_cmd, root)

    if args.force_retrain or not checkpoint.exists():
        train_cmd = [
            *python_cmd, root / "test" / "specaugment_train_v2.py",
            "--specaugment-band", args.specaugment_band,
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
            "--resume", source_checkpoint,
            "--data-dir", data_dir,
            "--output-dir", output_dir,
            "--device", args.train_device,
            "--cond-size", model_kwargs["cond_size"],
            "--gru-size", model_kwargs["gru_size"],
            "--epochs", args.epochs,
            "--batch-size", args.batch_size,
            "--lr", args.lr,
            "--seq-len", args.seq_len,
            "--num-workers", args.num_workers,
            "--snr-range", args.snr_range[0], args.snr_range[1],
            "--w-vad", args.w_vad,
            "--w-pitch", args.w_pitch,
        ]
        run_cmd(train_cmd, cwd=root)
    if args.force_reevaluate or not (eval_csv.exists() and eval_json.exists()):
        eval_cmd = [
            *python_cmd, root / "test" / "evaluate_consistent.py",
            "--checkpoint", checkpoint,
            "--data-dir", data_dir,
            "--device", args.eval_device,
            "--csv", eval_csv,
            "--json", eval_json,
        ]
        run_cmd(eval_cmd, cwd=root)
    write_overall_json(eval_csv, overall_json, checkpoint, args, model_kwargs)


if __name__ == "__main__":
    main()
