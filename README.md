# DA-AKT：难度感知知识追踪与学习诊断系统

本项目是面向“高级软件工程”课程作业的完整工程项目。项目参考 AKT（Attentive Knowledge Tracing）和 pyKT 的知识追踪实验思想，但不是直接复制 pyKT 的多模型评测库，而是将 AKT 改造为一个可运行、可检查、可复现的学生学习状态预测与诊断系统。

项目核心目标：根据学生历史答题记录预测未来答题正确概率，并进一步输出学生薄弱知识点、知识点掌握度、逐题预测明细和自动实验报告。

---

## 1. 项目亮点

| 模块 | 说明 |
|---|---|
| AKT 主干模型 | 包含知识点嵌入、答题交互嵌入、因果多头注意力、单调距离惩罚、残差连接、LayerNorm、FFN 和预测层 |
| 题目难度感知 | 只使用训练集统计每道题正确率，计算 `difficulty(q)=1-correct_rate_train(q)`，避免验证/测试答案泄露 |
| 严格防泄露协议 | 预测第 `t` 个位置时，只允许访问 `0..t-1` 的历史交互，不能访问包含当前答案 `r_t` 的当前位置交互嵌入 |
| 学习诊断输出 | 生成学生层面的平均掌握度、高风险比例和薄弱知识点列表 |
| 知识点分析输出 | 生成知识点层面的预测掌握度、真实正确率和风险率 |
| 自动实验报告 | 训练结束后自动输出 `experiment_report.md`、`metrics.json`、`history.csv` 等文件 |
| 消融实验 | 提供 DA-AKT 与 Plain-AKT 两套配置，方便说明“我在 AKT 基础上改了什么” |

---

## 2. 与 pyKT / 原 AKT 的区别

| 对比点 | pyKT / 原 AKT | 本项目 DA-AKT |
|---|---|---|
| 项目定位 | 通用知识追踪模型评测框架 | 面向课程作业的学习预测与诊断系统 |
| 模型范围 | 多个 KT baseline | 以 AKT 为主干做工程化增强 |
| 题目难度 | 主要依赖模型内部题目参数 | 增加训练集统计型题目难度特征 |
| 防泄露设计 | 依赖原框架协议 | 显式实现严格历史交互掩码，并提供检查脚本 |
| 输出结果 | 主要输出 AUC、ACC 等指标 | 指标 + 学生诊断 + 知识点分析 + 自动报告 |
| 软件工程 | 研究库结构较复杂 | 课程项目结构清晰，运行命令和报告模板完整 |

---

## 3. 项目结构

```text
da_akt_advanced_se_project/
├── README.md
├── requirements.txt
├── run.sh
├── run_ablation.sh
├── CHECK_REPORT.md
├── GITHUB_UPLOAD_GUIDE.md
├── configs/
│   ├── test.yaml
│   ├── test_plain_akt_ablation.yaml
│   ├── assist2009_da_akt.yaml
│   └── assist2009_plain_akt_ablation.yaml
├── data/
│   └── sample_interactions.csv
├── scripts/
│   ├── build_sequences.py
│   ├── compare_runs.py
│   ├── generate_sample_data.py
│   ├── run_project_check.py
│   └── sanity_check_no_leakage.py
├── src/
│   └── da_akt_project/
│       ├── data.py
│       ├── diagnosis.py
│       ├── model.py
│       ├── train.py
│       └── utils.py
├── reports/
│   ├── 高级软件工程课程项目报告.md
│   ├── 高级软件工程课程项目报告.docx
│   └── project_report.md
└── outputs/
    ├── test_run/
    ├── test_plain_akt/
    └── ablation_comparison.md
```

---

## 4. 环境安装

建议使用 Python 3.8+。

```bash
pip install -r requirements.txt
```

依赖包括：PyTorch、NumPy、Pandas、Scikit-learn、PyYAML、tqdm。

---

## 5. 数据格式

原始数据需要整理为 CSV，至少包含以下字段：

```csv
user_id,question_id,concept_id,correct,timestamp
```

| 字段 | 含义 |
|---|---|
| user_id | 学生编号 |
| question_id | 题目编号 |
| concept_id | 知识点编号 |
| correct | 答题结果，正确为 1，错误为 0 |
| timestamp | 作答时间或序列顺序，用于排序 |

项目自带 `data/sample_interactions.csv`，用于快速验证完整工程流程。正式实验时，可将 ASSIST2009 等数据整理成相同格式并保存为 `data/assist2009.csv`。

---

## 6. 快速运行

运行 DA-AKT 快速测试：

```bash
python -m src.da_akt_project.train --config configs/test.yaml
```

或：

```bash
bash run.sh
```

运行后输出在：

```text
outputs/test_run/
```

---

## 7. 防标签泄露检查

知识追踪中最重要的可靠性问题之一是：预测第 `t` 题时，模型不能看到当前答案 `r_t`。本项目已经实现严格历史交互协议：

```text
query t 只能访问 key < t
```

运行检查：

```bash
python scripts/sanity_check_no_leakage.py
```

期望输出：

```text
PASS: strict previous-interaction protocol has no current-label leakage.
```

---

## 8. 消融实验

运行 DA-AKT 与 Plain-AKT 快速对比：

```bash
bash run_ablation.sh
```

输出：

```text
outputs/ablation_comparison.md
outputs/ablation_comparison.csv
```

说明：压缩包内置的快速结果来自样例数据，主要用于证明工程流程可运行。正式报告可替换为真实数据集运行后的结果。

---

## 9. 使用 ASSIST2009 真实数据

将真实数据整理为：

```text
data/assist2009.csv
```

字段仍为：

```csv
user_id,question_id,concept_id,correct,timestamp
```

运行 DA-AKT：

```bash
python -m src.da_akt_project.train --config configs/assist2009_da_akt.yaml
```

运行普通 AKT 消融版本：

```bash
python -m src.da_akt_project.train --config configs/assist2009_plain_akt_ablation.yaml
```

---

## 10. 输出文件说明

训练结束后会生成：

| 文件 | 说明 |
|---|---|
| best_model.pt | 验证集 AUC 最优模型 |
| used_config.json | 本次运行实际配置 |
| metrics.json | 测试集指标和项目统计信息 |
| history.csv | 每轮训练/验证 AUC、ACC、Loss |
| test_predictions.csv | 测试集标签和预测概率 |
| diagnosis_predictions.csv | 学生、题目、知识点、难度、预测明细 |
| concept_mastery.csv | 知识点掌握度分析 |
| student_diagnosis.csv | 学生学习状态与薄弱知识点诊断 |
| experiment_report.md | 自动生成实验报告 |

---

## 11. 模型公式说明

题目难度估计：

```text
difficulty(q) = 1 - correct_rate_train(q)
```

难度门控融合：

```text
gate = sigmoid(W[base_embedding; difficulty_embedding])
enhanced_embedding = base_embedding + gate * difficulty_embedding
```

预测协议：

```text
P(r_t = 1) = DA-AKT(q_t, c_t, difficulty_t, history_{<t})
```

其中 `history_{<t}` 表示第 `t` 个位置之前的历史答题交互，不包含当前答案 `r_t`。

---

## 12. 课程报告说明

可直接使用的报告文件位于：

```text
reports/高级软件工程课程项目报告.md
reports/高级软件工程课程项目报告.docx
```

上传 GitHub 后，将报告第一部分中的 GitHub 链接替换为自己的仓库地址即可。

---

## 13. 一键自检与可部署性验证

提交前建议运行：

```bash
python scripts/run_project_check.py
```

该脚本会自动执行：

1. `python -m compileall -q src scripts`：检查 Python 语法；
2. `python scripts/sanity_check_no_leakage.py`：检查当前位置答案不泄露；
3. `configs/test.yaml`：运行 DA-AKT 快速训练；
4. `configs/test_plain_akt_ablation.yaml`：运行 Plain-AKT 消融训练；
5. `scripts/compare_runs.py`：生成消融对比报告；
6. 检查 `metrics.json`、`history.csv`、`student_diagnosis.csv`、`concept_mastery.csv` 等关键输出文件是否生成。

通过后会输出：

```text
PASS: project check completed successfully.
```

---

## 14. 数据集部署注意事项

正式替换数据集时，必须满足：

1. CSV 至少包含 `user_id,question_id,concept_id,correct,timestamp` 五列；
2. `correct` 只能为 `0` 或 `1`；
3. 同一学生的记录应能通过 `timestamp` 排序；
4. 每个学生至少保留 `min_seq_len` 条有效交互，否则该学生不会形成训练序列；
5. 项目默认按学生划分训练/验证/测试集，避免同一学生同时出现在不同数据划分中。

程序已经补充了数据合法性检查：缺失字段会直接报错；缺失必要字段的行会被删除并给出警告；非法 `correct` 值会被过滤；时间戳会优先按数值排序，其次按日期排序，避免出现字符串排序中 `10` 排在 `2` 前面的错误。

