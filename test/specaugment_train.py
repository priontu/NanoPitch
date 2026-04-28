#!/usr/bin/env python3
"""Run NanoPitch training with test-side SpecAugment."""

import argparse
import os
import sys

import numpy as np
import torch


def parse_wrapper_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--specaugment-band", type=int, required=True)
    parser.add_argument("--use-f0-voicing", action="store_true")
    parser.add_argument("--skip-train-eval", action="store_true")
    parser.add_argument("--save-best-only", action="store_true")
    return parser.parse_known_args(argv)


def apply_frequency_mask(mel, band):
    if band <= 0:
        return mel
    n_mels = mel.size(-1)
    band = min(int(band), n_mels)
    batch_size = mel.size(0)
    starts = torch.randint(
        low=0,
        high=n_mels - band + 1,
        size=(batch_size,),
        device=mel.device,
    )
    augmented = mel.clone()
    fill = mel.mean(dim=(1, 2), keepdim=True)
    for batch_idx, start in enumerate(starts.tolist()):
        augmented[batch_idx, :, start:start + band] = fill[batch_idx]
    return augmented


def main():
    wrapper_args, remaining = parse_wrapper_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    training_dir = os.path.join(root, "training")
    sys.path.insert(0, training_dir)

    import train as train_module

    original_augment = train_module.augment_mel_batch
    original_torch_save = train_module.torch.save
    original_getitem = train_module.NanoPitchDataset.__getitem__

    def augment_with_specaugment(mel_clean, mel_noise, snr_range, device):
        mel_mix = original_augment(mel_clean, mel_noise, snr_range, device)
        return apply_frequency_mask(mel_mix, wrapper_args.specaugment_band)

    train_module.augment_mel_batch = augment_with_specaugment

    if wrapper_args.use_f0_voicing:
        def getitem_with_f0_voicing(self, idx):
            mel_clean, mel_noise, vad, f0 = original_getitem(self, idx)
            voiced = (np.asarray(f0) > 0).astype(np.float32)
            return mel_clean, mel_noise, voiced, f0
        train_module.NanoPitchDataset.__getitem__ = getitem_with_f0_voicing
        print("  [train] using f0 > 0 as voiced target in test/specaugment_train.py")

    if wrapper_args.skip_train_eval:
        def skip_evaluate(model, data_dir, writer, epoch, device, args):
            print("  [eval] skipped by test/specaugment_train.py")
            return None
        train_module.evaluate = skip_evaluate

    if wrapper_args.save_best_only:
        def save_best_only(obj, f, *args, **kwargs):
            if os.path.basename(os.fspath(f)) == "best.pth":
                return original_torch_save(obj, f, *args, **kwargs)
            return None
        train_module.torch.save = save_best_only

    train_module.main()


if __name__ == "__main__":
    main()
