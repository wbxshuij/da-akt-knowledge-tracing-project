import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_sequences, load_meta, load_npz_dataset
from .diagnosis import collect_predictions, generate_diagnosis, write_experiment_report
from .model import DifficultyAwareAKT
from .utils import count_parameters, ensure_dir, get_device, load_config, safe_torch_load, save_json, set_seed


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"auc": 0.5, "acc": 0.0}
    auc = 0.5 if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
    return {"auc": auc, "acc": acc}


def run_epoch(model, loader, optimizer, device, train: bool, grad_clip: float = 1.0) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    model.train(train)
    total_loss = 0.0
    total_count = 0
    ys, ps = [], []

    iterator = tqdm(loader, leave=False, desc="train" if train else "eval")
    for batch in iterator:
        q = batch["qseqs"].to(device)
        c = batch["cseqs"].to(device)
        r = batch["rseqs"].to(device)
        m = batch["masks"].to(device)
        d = batch.get("dseqs")
        d = d.to(device) if d is not None else None

        logits = model(q, c, r, m, d)
        target_mask = m.bool().clone()
        # Under strict-previous protocol, the first position has no history and is not evaluated.
        target_mask[:, 0] = False
        loss_mat = criterion(logits, r.float())
        if target_mask.sum() == 0:
            continue
        loss = loss_mat[target_mask].mean()

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            prob = torch.sigmoid(logits)[target_mask].detach().cpu().numpy()
            y = r[target_mask].detach().cpu().numpy()
        ys.append(y)
        ps.append(prob)
        total_loss += float(loss.item()) * len(y)
        total_count += int(len(y))

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_prob = np.concatenate(ps) if ps else np.array([], dtype=np.float32)
    metrics = compute_metrics(y_true, y_prob)
    return total_loss / max(total_count, 1), metrics, y_true, y_prob


def _build_model(cfg: Dict, meta) -> DifficultyAwareAKT:
    return DifficultyAwareAKT(
        n_questions=meta.n_questions,
        n_concepts=meta.n_concepts,
        n_difficulty_bins=meta.n_difficulty_bins,
        seq_len=meta.seq_len,
        d_model=cfg["d_model"],
        n_blocks=cfg["n_blocks"],
        n_heads=cfg["num_attn_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        separate_qa=cfg.get("separate_qa", True),
        use_rasch=cfg.get("use_rasch", True),
        use_difficulty=cfg.get("use_difficulty", True),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/assist2009_da_akt.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.environ.setdefault("TORCH_NUM_THREADS", str(cfg.get("torch_num_threads", 1)))
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    set_seed(cfg.get("seed", 3407), deterministic=cfg.get("deterministic", False))
    device = get_device(cfg.get("device", "auto"))
    ensure_dir(cfg["save_dir"])
    save_json(cfg, os.path.join(cfg["save_dir"], "used_config.json"))

    meta = build_sequences(
        raw_csv=cfg["raw_csv"],
        processed_dir=cfg["processed_dir"],
        seq_len=cfg["seq_len"],
        min_seq_len=cfg.get("min_seq_len", 3),
        valid_ratio=cfg.get("valid_ratio", 0.1),
        test_ratio=cfg.get("test_ratio", 0.2),
        seed=cfg.get("seed", 3407),
        n_difficulty_bins=cfg.get("n_difficulty_bins", 10),
        rebuild=cfg.get("rebuild_processed", True),
    )
    meta = load_meta(cfg["processed_dir"])

    train_ds = load_npz_dataset(os.path.join(cfg["processed_dir"], "train.npz"))
    valid_ds = load_npz_dataset(os.path.join(cfg["processed_dir"], "valid.npz"))
    test_ds = load_npz_dataset(os.path.join(cfg["processed_dir"], "test.npz"))
    if len(train_ds) == 0:
        raise RuntimeError("No training sequences were generated. Check min_seq_len and data size.")
    if len(valid_ds) == 0:
        print("WARNING: validation split has no usable sequences; early stopping will monitor train AUC.")
    if len(test_ds) == 0:
        print("WARNING: test split has no usable sequences; reported test metrics will not be meaningful.")
    if cfg["d_model"] % cfg["num_attn_heads"] != 0:
        raise ValueError("d_model must be divisible by num_attn_heads. Please fix the config.")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 0))
    valid_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 0))
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 0))

    model = _build_model(cfg, meta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))

    history = []
    best_auc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(cfg["save_dir"], "best_model.pt")

    print(f"Device: {device}")
    print(f"Model: DA-AKT | strict_previous_interactions=True | use_difficulty={cfg.get('use_difficulty', True)}")
    print(f"Sequences: train={len(train_ds)}, valid={len(valid_ds)}, test={len(test_ds)}")
    print(f"Items={meta.n_questions}, concepts={meta.n_concepts}, users={meta.n_users}, parameters={count_parameters(model):,}")

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_met, _, _ = run_epoch(model, train_loader, optimizer, device, True, cfg.get("grad_clip", 1.0))
        va_loss, va_met, _, _ = run_epoch(model, valid_loader, optimizer, device, False)
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "valid_loss": va_loss,
            "train_auc": tr_met["auc"],
            "train_acc": tr_met["acc"],
            "valid_auc": va_met["auc"],
            "valid_acc": va_met["acc"],
        }
        history.append(row)
        print(row)

        monitor = va_met["auc"] if len(valid_ds) > 0 else tr_met["auc"]
        if monitor > best_auc:
            best_auc = monitor
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "config": cfg, "meta": meta.__dict__}, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.get("patience", 10):
                print("Early stopping triggered.")
                break

    ckpt = safe_torch_load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_met, y_true, y_prob = run_epoch(model, test_loader, optimizer, device, False)
    best_epoch = int(pd.DataFrame(history).sort_values("valid_auc", ascending=False).iloc[0]["epoch"]) if history else -1
    metrics = {
        "best_epoch": best_epoch,
        "best_valid_auc": float(best_auc),
        "test_loss": float(te_loss),
        "test_auc": float(te_met["auc"]),
        "test_acc": float(te_met["acc"]),
        "n_questions": meta.n_questions,
        "n_concepts": meta.n_concepts,
        "n_users": meta.n_users,
        "n_train_seq": len(train_ds),
        "n_valid_seq": len(valid_ds),
        "n_test_seq": len(test_ds),
        "n_difficulty_bins": meta.n_difficulty_bins,
        "use_difficulty": bool(cfg.get("use_difficulty", True)),
        "target_protocol": meta.target_protocol,
        "trainable_parameters": count_parameters(model),
    }
    save_json(metrics, os.path.join(cfg["save_dir"], "metrics.json"))
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(cfg["save_dir"], "history.csv"), index=False)
    pd.DataFrame({"label": y_true.astype(int), "prob": y_prob}).to_csv(os.path.join(cfg["save_dir"], "test_predictions.csv"), index=False)

    if cfg.get("generate_diagnosis", True):
        pred_df = collect_predictions(model, test_loader, device)
        generate_diagnosis(pred_df, meta, cfg["save_dir"])
        write_experiment_report(metrics, hist_df, cfg["save_dir"])

    print("Final test metrics:", metrics)
    print("Outputs saved to:", cfg["save_dir"])


if __name__ == "__main__":
    main()
