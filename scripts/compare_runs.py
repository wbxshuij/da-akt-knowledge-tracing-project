import argparse
import json
import os
from typing import Dict

import pandas as pd


def read_metrics(run_dir: str) -> Dict:
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown without optional tabulate dependency."""
    headers = [str(col) for col in df.columns]
    rows = []
    for row in df.itertuples(index=False, name=None):
        rows.append([
            f"{value:.4f}" if isinstance(value, float) else str(value)
            for value in row
        ])

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_row(values):
        return "| " + " | ".join(str(values[i]).ljust(widths[i]) for i in range(len(values))) + " |"

    lines = [fmt_row(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two DA-AKT/AKT experiment runs.")
    parser.add_argument("--run_a", required=True)
    parser.add_argument("--name_a", default="DA-AKT")
    parser.add_argument("--run_b", required=True)
    parser.add_argument("--name_b", default="Plain-AKT")
    parser.add_argument("--output", default="outputs/ablation_comparison.md")
    args = parser.parse_args()

    a = read_metrics(args.run_a)
    b = read_metrics(args.run_b)
    rows = []
    for name, m in [(args.name_a, a), (args.name_b, b)]:
        rows.append({
            "run": name,
            "use_difficulty": m.get("use_difficulty"),
            "best_valid_auc": m.get("best_valid_auc"),
            "test_auc": m.get("test_auc"),
            "test_acc": m.get("test_acc"),
            "test_loss": m.get("test_loss"),
            "trainable_parameters": m.get("trainable_parameters"),
        })
    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.splitext(args.output)[0] + ".csv"
    df.to_csv(csv_path, index=False)

    delta_auc = float(a.get("test_auc", 0)) - float(b.get("test_auc", 0))
    delta_acc = float(a.get("test_acc", 0)) - float(b.get("test_acc", 0))
    md = "# 消融实验对比报告\n\n"
    md += "本报告用于对比加入题目难度感知模块的 DA-AKT 与关闭难度模块的普通 AKT。\n\n"
    md += dataframe_to_markdown(df)
    md += "\n\n"
    md += f"- Test AUC 差值（{args.name_a} - {args.name_b}）：{delta_auc:.4f}\n"
    md += f"- Test ACC 差值（{args.name_a} - {args.name_b}）：{delta_acc:.4f}\n\n"
    md += "说明：压缩包中的结果来自快速样例数据，主要用于证明工程流程可运行；正式课程报告可替换为 ASSIST2009 等真实数据集后的结果。\n"
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
