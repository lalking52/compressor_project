"""
Split a source audio set into train/test lists by creating symlinks in data/ and data_test/.
Supports wav/flac; preserves subdirectory structure; avoids duplicating audio.
Example:
    python src/prepare_split.py --source LibriSpeech/test-clean --train-ratio 0.5
"""
import argparse
import glob
import os
from pathlib import Path
import random


def find_audio(root):
    root = Path(root)
    wavs = glob.glob(str(root / "**/*.wav"), recursive=True)
    flacs = glob.glob(str(root / "**/*.flac"), recursive=True)
    return [Path(p) for p in sorted(wavs + flacs)]


def make_link(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return
    # Use symlinks to avoid duplicating large audio files
    dst_path.symlink_to(src_path.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="LibriSpeech/test-clean", help="Root folder with wav/flac files")
    parser.add_argument("--train-ratio", type=float, default=0.5, help="Fraction of files for train (rest for test)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducibility")
    args = parser.parse_args()

    files = find_audio(args.source)
    if not files:
        raise SystemExit(f"No audio found under {args.source} (looked for wav/flac).")

    random.seed(args.seed)
    random.shuffle(files)
    split = int(len(files) * args.train_ratio)
    train_files = files[:split]
    test_files = files[split:]

    train_root = Path("data")
    test_root = Path("data_test")
    for src in train_files:
        rel = src.relative_to(args.source)
        make_link(src, train_root / rel)
    for src in test_files:
        rel = src.relative_to(args.source)
        make_link(src, test_root / rel)

    print(f"Linked {len(train_files)} files to {train_root} and {len(test_files)} to {test_root}.")


if __name__ == "__main__":
    main()
