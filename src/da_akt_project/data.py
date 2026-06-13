import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {"user_id", "question_id", "concept_id", "correct", "timestamp"}


def _normalise_id_column(series: pd.Series, column_name: str) -> pd.Series:
    """Convert an ID column to clean strings and reject empty values."""
    cleaned = series.astype(str).str.strip()
    invalid = cleaned.eq("") | cleaned.str.lower().isin({"nan", "none", "null"})
    if invalid.any():
        raise ValueError(f"Column {column_name!r} contains empty/null-like IDs after cleaning.")
    return cleaned


def _timestamp_sort_key(series: pd.Series) -> pd.Series:
    """Return a stable sortable timestamp key.

    Numeric timestamps are sorted numerically; datetime-like timestamps are sorted
    chronologically; otherwise the original string order is used. This prevents
    common mistakes such as lexicographic sorting where "10" appears before
    "2".
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float)
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().all():
        return dt.view("int64")
    return series.astype(str)


@dataclass
class KTMeta:
    n_questions: int
    n_concepts: int
    n_users: int
    q2idx: Dict[str, int]
    c2idx: Dict[str, int]
    u2idx: Dict[str, int]
    n_difficulty_bins: int = 10
    seq_len: int = 100
    target_protocol: str = "strict_previous_interactions"


class KTDataset(Dataset):
    def __init__(self, seqs: List[Dict[str, np.ndarray]]):
        self.seqs = seqs

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.seqs[idx]
        return {k: torch.as_tensor(v, dtype=torch.long) for k, v in item.items()}


def _remap(values: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    uniq = sorted(values.astype(str).unique().tolist())
    mp = {v: i + 1 for i, v in enumerate(uniq)}  # 0 is reserved for padding.
    return values.astype(str).map(mp).astype(int), mp


def _split_users(users: List[int], valid_ratio: float, test_ratio: float, seed: int) -> Tuple[set, set, set]:
    if not 0 <= valid_ratio < 1 or not 0 <= test_ratio < 1 or valid_ratio + test_ratio >= 1:
        raise ValueError("valid_ratio and test_ratio must be in [0, 1), and their sum must be < 1.")
    rng = np.random.default_rng(seed)
    users = list(users)
    rng.shuffle(users)
    n = len(users)
    if n < 3:
        raise ValueError("At least 3 users are required for train/valid/test splitting.")
    n_test = max(1, int(round(n * test_ratio)))
    n_valid = max(1, int(round(n * valid_ratio)))
    if n_test + n_valid >= n:
        n_test = max(1, min(n_test, n - 2))
        n_valid = 1
    test_users = set(users[:n_test])
    valid_users = set(users[n_test:n_test + n_valid])
    train_users = set(users[n_test + n_valid:])
    if not train_users:
        raise ValueError("Train split is empty. Decrease valid_ratio/test_ratio or add more users.")
    return train_users, valid_users, test_users


def _difficulty_bins(train_df: pd.DataFrame, all_questions: np.ndarray, n_bins: int) -> Tuple[Dict[int, int], pd.DataFrame]:
    """Estimate item difficulty from train split only.

    difficulty(q) = 1 - correct_rate_train(q).
    The value is discretized to 1..n_bins. 0 is padding.
    Questions unseen in the training split use the global training difficulty and are marked as global_default.
    """
    if n_bins < 2:
        raise ValueError("n_difficulty_bins must be at least 2.")
    if len(train_df) == 0:
        raise ValueError("Cannot estimate difficulty from an empty training split.")

    global_correct_rate = float(train_df["correct"].mean())
    global_difficulty = 1.0 - global_correct_rate
    grouped = train_df.groupby("q")["correct"].agg(train_count="count", train_correct_rate="mean").reset_index()
    stats_by_q = {int(row.q): row for row in grouped.itertuples(index=False)}

    rows = []
    bin_map: Dict[int, int] = {}
    for q in sorted(int(x) for x in all_questions):
        row = stats_by_q.get(q)
        if row is None:
            count = 0
            correct_rate = global_correct_rate
            source = "global_default_unseen_in_train"
        else:
            count = int(row.train_count)
            correct_rate = float(row.train_correct_rate)
            source = "train_estimated"
        difficulty = float(np.clip(1.0 - correct_rate, 0.0, 0.999999))
        diff_bin = int(np.floor(difficulty * n_bins)) + 1
        bin_map[q] = diff_bin
        rows.append({
            "question_idx": q,
            "train_count": count,
            "train_correct_rate": correct_rate,
            "train_difficulty": difficulty,
            "difficulty_bin": diff_bin,
            "source": source,
            "global_train_correct_rate": global_correct_rate,
            "global_train_difficulty": global_difficulty,
        })
    return bin_map, pd.DataFrame(rows)


def _pad_sequence(values: np.ndarray, seq_len: int) -> np.ndarray:
    values = values.astype(np.int64)
    if len(values) > seq_len:
        raise ValueError("_pad_sequence received a sequence longer than seq_len.")
    return np.pad(values, (0, seq_len - len(values)), mode="constant", constant_values=0)


def build_sequences(raw_csv: str, processed_dir: str, seq_len: int = 100, min_seq_len: int = 3,
                    valid_ratio: float = 0.1, test_ratio: float = 0.2, seed: int = 3407,
                    n_difficulty_bins: int = 10, rebuild: bool = True) -> KTMeta:
    if seq_len < 3:
        raise ValueError("seq_len must be at least 3.")
    if min_seq_len < 2:
        raise ValueError("min_seq_len must be at least 2 because the first position is used as history only.")
    if rebuild and os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    if not os.path.exists(raw_csv):
        raise FileNotFoundError(
            f"Raw data file not found: {raw_csv}. Please place a CSV file with columns "
            f"{sorted(REQUIRED_COLUMNS)} at this path or update raw_csv in the config."
        )

    df = pd.read_csv(raw_csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Required columns: {sorted(REQUIRED_COLUMNS)}")

    before_drop = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    if len(df) < before_drop:
        print(f"WARNING: dropped {before_drop - len(df)} rows with missing required fields.")

    for col in ["user_id", "question_id", "concept_id"]:
        df[col] = _normalise_id_column(df[col], col)

    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    invalid_correct = df["correct"].isna() | ~df["correct"].isin([0, 1])
    if invalid_correct.any():
        print(f"WARNING: dropped {int(invalid_correct.sum())} rows whose correct value is not 0/1.")
        df = df.loc[~invalid_correct].copy()
    df["correct"] = df["correct"].astype(int)
    if df.empty:
        raise ValueError("No valid interactions after filtering correct to {0, 1}.")

    df["_row_order"] = np.arange(len(df))
    df["_timestamp_sort"] = _timestamp_sort_key(df["timestamp"])
    df["u"], u2idx = _remap(df["user_id"])
    df["q"], q2idx = _remap(df["question_id"])
    df["c"], c2idx = _remap(df["concept_id"])
    df = df.sort_values(["u", "_timestamp_sort", "_row_order"]).reset_index(drop=True)

    train_users, valid_users, test_users = _split_users(df["u"].unique().tolist(), valid_ratio, test_ratio, seed)
    split_of_user = {u: "train" for u in train_users}
    split_of_user.update({u: "valid" for u in valid_users})
    split_of_user.update({u: "test" for u in test_users})
    df["split"] = df["u"].map(split_of_user)

    train_df = df[df["split"].eq("train")].copy()
    q_bin_map, difficulty_stats = _difficulty_bins(train_df, df["q"].unique(), n_difficulty_bins)
    difficulty_stats["question_id"] = difficulty_stats["question_idx"].map({v: k for k, v in q2idx.items()})
    difficulty_stats.to_csv(os.path.join(processed_dir, "question_difficulty_stats.csv"), index=False)

    split_info = pd.DataFrame({
        "user_idx": list(split_of_user.keys()),
        "split": list(split_of_user.values()),
    })
    split_info["user_id"] = split_info["user_idx"].map({v: k for k, v in u2idx.items()})
    split_info.to_csv(os.path.join(processed_dir, "user_split.csv"), index=False)

    splits: Dict[str, List[Dict[str, np.ndarray]]] = {"train": [], "valid": [], "test": []}
    for u, g in df.groupby("u", sort=False):
        qs = g["q"].to_numpy(dtype=np.int64)
        cs = g["c"].to_numpy(dtype=np.int64)
        rs = g["correct"].to_numpy(dtype=np.int64)
        ds = np.array([q_bin_map[int(q)] for q in qs], dtype=np.int64)
        us = np.full(shape=len(qs), fill_value=int(u), dtype=np.int64)
        if len(qs) < min_seq_len:
            continue
        split = split_of_user[int(u)]
        for start in range(0, len(qs), seq_len):
            end = min(start + seq_len, len(qs))
            if end - start < min_seq_len:
                continue
            q_chunk, c_chunk, r_chunk, d_chunk, u_chunk = qs[start:end], cs[start:end], rs[start:end], ds[start:end], us[start:end]
            L = len(q_chunk)
            item = {
                "useq": _pad_sequence(u_chunk, seq_len),
                "qseqs": _pad_sequence(q_chunk, seq_len),
                "cseqs": _pad_sequence(c_chunk, seq_len),
                "rseqs": _pad_sequence(r_chunk, seq_len),
                "dseqs": _pad_sequence(d_chunk, seq_len),
                "masks": np.array([1] * L + [0] * (seq_len - L), dtype=np.int64),
            }
            splits[split].append(item)

    keys = ["useq", "qseqs", "cseqs", "rseqs", "dseqs", "masks"]
    split_summary_rows = []
    for name, seqs in splits.items():
        arrays = {k: np.stack([x[k] for x in seqs]).astype(np.int64) if seqs else np.zeros((0, seq_len), dtype=np.int64)
                  for k in keys}
        np.savez_compressed(os.path.join(processed_dir, f"{name}.npz"), **arrays)
        n_targets = int(arrays["masks"][:, 1:].sum()) if arrays["masks"].size else 0
        split_summary_rows.append({"split": name, "n_sequences": len(seqs), "n_target_interactions": n_targets})
    pd.DataFrame(split_summary_rows).to_csv(os.path.join(processed_dir, "split_summary.csv"), index=False)

    meta = KTMeta(
        n_questions=len(q2idx),
        n_concepts=len(c2idx),
        n_users=len(u2idx),
        q2idx=q2idx,
        c2idx=c2idx,
        u2idx=u2idx,
        n_difficulty_bins=n_difficulty_bins,
        seq_len=seq_len,
    )
    with open(os.path.join(processed_dir, "id_maps.json"), "w", encoding="utf-8") as f:
        json.dump(meta.__dict__, f, ensure_ascii=False, indent=2)
    return meta


def load_npz_dataset(path: str) -> KTDataset:
    data = np.load(path)
    keys = [k for k in ["useq", "qseqs", "cseqs", "rseqs", "dseqs", "masks"] if k in data.files]
    seqs = [{k: data[k][i] for k in keys} for i in range(data["qseqs"].shape[0])]
    return KTDataset(seqs)


def load_meta(processed_dir: str) -> KTMeta:
    with open(os.path.join(processed_dir, "id_maps.json"), "r", encoding="utf-8") as f:
        obj = json.load(f)
    return KTMeta(**obj)
