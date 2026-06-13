import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.da_akt_project.data import build_sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--n_difficulty_bins", type=int, default=10)
    args = parser.parse_args()
    meta = build_sequences(
        args.raw_csv,
        args.processed_dir,
        args.seq_len,
        args.min_seq_len,
        args.valid_ratio,
        args.test_ratio,
        args.seed,
        args.n_difficulty_bins,
        rebuild=True,
    )
    print(meta)
