import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from .utils import ensure_dir


def _inverse_map(mp: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in mp.items()}


@torch.no_grad()
def collect_predictions(model, loader, device) -> pd.DataFrame:
    """Collect per-target predictions for diagnosis.

    The first position of each sequence is excluded because it has no previous
    interaction history under the strict previous-interaction protocol.
    """
    model.eval()
    rows = []
    for batch in loader:
        u = batch.get("useq")
        q = batch["qseqs"].to(device)
        c = batch["cseqs"].to(device)
        r = batch["rseqs"].to(device)
        m = batch["masks"].to(device)
        d = batch.get("dseqs")
        d_dev = d.to(device) if d is not None else None
        logits = model(q, c, r, m, d_dev)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        q_np = q.detach().cpu().numpy()
        c_np = c.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        m_np = m.detach().cpu().numpy()
        d_np = d.numpy() if d is not None else np.zeros_like(q_np)
        u_np = u.numpy() if u is not None else np.zeros_like(q_np)
        for i in range(q_np.shape[0]):
            for t in range(1, q_np.shape[1]):
                if m_np[i, t] == 0:
                    continue
                p = float(prob[i, t])
                rows.append({
                    "sequence_pos": int(t),
                    "user_idx": int(u_np[i, t]),
                    "question_idx": int(q_np[i, t]),
                    "concept_idx": int(c_np[i, t]),
                    "difficulty_bin": int(d_np[i, t]),
                    "label": int(r_np[i, t]),
                    "prob_correct": p,
                    "pred_label": int(p >= 0.5),
                    "risk_score": float(1.0 - p),
                    "is_high_risk": int(p < 0.5),
                })
    return pd.DataFrame(rows)


def _weak_concepts_for_students(sc: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    rows = []
    for (user_idx, user_id), group in sc.groupby(["user_idx", "user_id"], sort=False):
        group = group.sort_values(["predicted_mastery", "samples"], ascending=[True, False]).head(top_k)
        text = "; ".join(
            f"{row.concept_id}(mastery={row.predicted_mastery:.3f}, risk={row.high_risk_rate:.3f})"
            for row in group.itertuples(index=False)
        )
        rows.append({"user_idx": user_idx, "user_id": user_id, "weak_concepts": text})
    return pd.DataFrame(rows)


def generate_diagnosis(pred_df: pd.DataFrame, meta, save_dir: str) -> Tuple[str, str, str]:
    """Generate teaching-oriented CSV diagnosis files."""
    ensure_dir(save_dir)
    q_inv = _inverse_map(meta.q2idx)
    c_inv = _inverse_map(meta.c2idx)
    u_inv = _inverse_map(meta.u2idx)
    df = pred_df.copy()
    if df.empty:
        pred_path = os.path.join(save_dir, "diagnosis_predictions.csv")
        concept_path = os.path.join(save_dir, "concept_mastery.csv")
        student_path = os.path.join(save_dir, "student_diagnosis.csv")
        df.to_csv(pred_path, index=False)
        pd.DataFrame().to_csv(concept_path, index=False)
        pd.DataFrame().to_csv(student_path, index=False)
        return pred_path, concept_path, student_path

    df["user_id"] = df["user_idx"].map(u_inv)
    df["question_id"] = df["question_idx"].map(q_inv)
    df["concept_id"] = df["concept_idx"].map(c_inv)
    pred_path = os.path.join(save_dir, "diagnosis_predictions.csv")
    df.to_csv(pred_path, index=False)

    concept = df.groupby(["concept_idx", "concept_id"], as_index=False).agg(
        samples=("label", "count"),
        real_correct_rate=("label", "mean"),
        predicted_mastery=("prob_correct", "mean"),
        avg_risk_score=("risk_score", "mean"),
        high_risk_rate=("is_high_risk", "mean"),
    ).sort_values(["predicted_mastery", "samples"], ascending=[True, False])
    concept_path = os.path.join(save_dir, "concept_mastery.csv")
    concept.to_csv(concept_path, index=False)

    sc = df.groupby(["user_idx", "user_id", "concept_idx", "concept_id"], as_index=False).agg(
        samples=("label", "count"),
        predicted_mastery=("prob_correct", "mean"),
        avg_risk_score=("risk_score", "mean"),
        high_risk_rate=("is_high_risk", "mean"),
    )
    weak = _weak_concepts_for_students(sc, top_k=3)
    student = df.groupby(["user_idx", "user_id"], as_index=False).agg(
        samples=("label", "count"),
        real_correct_rate=("label", "mean"),
        avg_predicted_mastery=("prob_correct", "mean"),
        avg_risk_score=("risk_score", "mean"),
        high_risk_rate=("is_high_risk", "mean"),
    ).merge(weak, on=["user_idx", "user_id"], how="left")
    student_path = os.path.join(save_dir, "student_diagnosis.csv")
    student.to_csv(student_path, index=False)
    return pred_path, concept_path, student_path


def write_experiment_report(metrics: Dict, history: pd.DataFrame, save_dir: str) -> str:
    ensure_dir(save_dir)
    best_epoch = int(history.sort_values("valid_auc", ascending=False).iloc[0]["epoch"]) if len(history) else -1
    report = f"""# DA-AKT 实验自动报告

## 1. 项目说明

本项目在 AKT 的基础上加入题目难度感知模块，形成 DA-AKT（Difficulty-Aware Attentive Knowledge Tracing）。系统不仅输出 AUC、ACC 等模型指标，还会生成学生学习诊断和知识点掌握情况分析。

## 2. 关键实现检查

1. **防止标签泄露**：预测位置 `t` 时，注意力层采用严格下三角掩码，只允许访问 `0..t-1` 的历史交互，不允许访问包含当前答案 `r_t` 的交互嵌入。
2. **难度特征不泄露**：题目难度只使用训练集学生的历史正确率估计，验证集和测试集答案不会参与难度统计。
3. **全流程工程化**：包含数据预处理、模型训练、验证早停、测试评估、诊断输出和报告生成。

## 3. 核心改动

1. 题目难度估计：`difficulty(q)=1-correct_rate_train(q)`。
2. 难度感知嵌入：将题目难度离散为若干区间，并通过门控方式融合到知识点表示和答题交互表示中。
3. 学习诊断输出：测试结束后自动生成学生层面的薄弱知识点、知识点层面的掌握度和逐题预测结果。
4. 自动实验报告：训练结束后自动保存指标、历史曲线数据和 Markdown 报告，方便课程作业整理。

## 4. 实验结果

| 指标 | 数值 |
|---|---:|
| Best Valid AUC | {metrics.get('best_valid_auc', 0):.4f} |
| Test AUC | {metrics.get('test_auc', 0):.4f} |
| Test ACC | {metrics.get('test_acc', 0):.4f} |
| Test Loss | {metrics.get('test_loss', 0):.4f} |
| Best Epoch | {best_epoch} |
| Trainable Parameters | {metrics.get('trainable_parameters', 0)} |

## 5. 输出文件

- `metrics.json`：最终实验指标。
- `history.csv`：每轮训练和验证结果。
- `test_predictions.csv`：测试集标签和预测概率。
- `diagnosis_predictions.csv`：学生、题目、知识点、难度和预测结果明细。
- `concept_mastery.csv`：知识点掌握度分析。
- `student_diagnosis.csv`：学生学习状态与薄弱知识点诊断。
"""
    path = os.path.join(save_dir, "experiment_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    return path
