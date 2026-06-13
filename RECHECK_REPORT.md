# DA-AKT 项目二次自检报告

## 自检时间
2026-06-13

## 自检结论
本次对压缩包中的 DA-AKT 高级软件工程课程项目进行了二次完整检查。检查内容覆盖目录结构、代码语法、依赖部署、数据格式、Demo 训练、消融实验、标签泄露检查、输出文件完整性和说明文档一致性。当前版本可以正常运行并满足课程项目提交要求。

## 1. 目录结构检查
项目目录按照代码、配置、数据、脚本、报告、输出结果进行分层管理：

- `src/da_akt_project/`：核心模型、训练、数据处理和诊断代码；
- `scripts/`：数据构建、防泄露检查、消融对比、一键自检脚本；
- `configs/`：快速测试配置和 ASSIST2009 正式配置；
- `data/`：样例数据和预处理结果；
- `outputs/`：Demo 训练结果、诊断结果和消融报告；
- `reports/`：课程报告 Markdown 与 Word 版本；
- `README.md`、`CHECK_REPORT.md`、`PROJECT_AUDIT_REPORT.md`、`GITHUB_UPLOAD_GUIDE.md`：说明、检查和上传文档。

检查结果：结构清晰，代码、数据、说明和实验输出分类合理。

## 2. 代码部署与语法检查
已执行以下命令：

```bash
python -m compileall -q src scripts
python scripts/sanity_check_no_leakage.py
python scripts/run_project_check.py
bash run.sh
bash run_ablation.sh
```

检查结果：

- Python 语法编译通过；
- 模型训练入口可以正常运行；
- `run.sh` 和 `run_ablation.sh` 可以从项目根目录直接执行；
- 一键自检脚本执行成功；
- 没有发现缺失导入、语法错误或运行入口错误。

## 3. 数据集格式检查
内置样例数据 `data/sample_interactions.csv` 检查结果：

| 项目 | 结果 |
|---|---:|
| 数据行数 | 6135 |
| 字段数 | 5 |
| 用户数 | 80 |
| 题目数 | 180 |
| 知识点数 | 30 |
| 缺失值 | 0 |
| correct 合法取值 | 0、1 |

要求字段为：

```csv
user_id,question_id,concept_id,correct,timestamp
```

检查结果：样例数据字段完全匹配代码读取要求，无缺失值，标签合法。

预处理后划分统计：

| split | 序列数 | 有效预测样本数 |
|---|---:|---:|
| train | 112 | 4139 |
| valid | 16 | 603 |
| test | 32 | 1227 |

## 4. Demo 与消融实验检查
DA-AKT 快速 Demo 结果：

| 指标 | 数值 |
|---|---:|
| Best Valid AUC | 0.5996 |
| Test AUC | 0.6335 |
| Test ACC | 0.5917 |
| Test Loss | 0.6765 |
| 参数量 | 103806 |

Plain-AKT 消融结果：

| 指标 | 数值 |
|---|---:|
| Best Valid AUC | 0.5484 |
| Test AUC | 0.5475 |
| Test ACC | 0.5281 |
| Test Loss | 0.7115 |
| 参数量 | 94846 |

检查结果：训练、验证、测试、最优模型保存、诊断输出和消融对比均可正常生成。

## 5. 标签泄露检查
已运行：

```bash
python scripts/sanity_check_no_leakage.py
```

输出结果：

```text
target_position=4, abs_logit_diff_after_flipping_current_label=0.0000000000
PASS: strict previous-interaction protocol has no current-label leakage.
```

检查结论：预测第 t 个位置时，模型不会看到当前位置答案 `r_t`。

## 6. 输出文件完整性检查
以下关键文件均已生成：

- `outputs/test_run/best_model.pt`
- `outputs/test_run/metrics.json`
- `outputs/test_run/history.csv`
- `outputs/test_run/test_predictions.csv`
- `outputs/test_run/diagnosis_predictions.csv`
- `outputs/test_run/concept_mastery.csv`
- `outputs/test_run/student_diagnosis.csv`
- `outputs/test_run/experiment_report.md`
- `outputs/test_plain_akt/metrics.json`
- `outputs/ablation_comparison.md`
- `outputs/ablation_comparison.csv`

## 7. 仍需注意的问题与建议

| 问题点 | 严重程度 | 说明 | 修改/处理方案 |
|---|---|---|---|
| 内置数据是样例数据 | 低 | Demo 结果主要用于证明项目可运行，不代表真实 ASSIST2009 最终效果 | 正式报告可替换 `data/assist2009.csv` 后运行正式配置 |
| requirements 未固定精确版本 | 低 | 不影响当前运行，但不同环境可能导致轻微差异 | 可在提交前补充测试环境说明，或固定最低版本 |
| README 中 GitHub 链接需要用户自己替换 | 低 | 上传仓库后才有真实链接 | 按 `GITHUB_UPLOAD_GUIDE.md` 上传后替换报告链接 |
| 正式大数据训练时间较长 | 中 | ASSIST2009 完整训练比 Demo 慢 | 可先用 `configs/test.yaml` 验证流程，再运行正式配置 |

## 最终结论
本项目已经通过二次自检，当前版本目录结构规范、代码可以部署运行、样例数据格式匹配、使用说明完整、Demo 和消融实验均已跑通。项目可以作为高级软件工程课程作业提交；提交前只需上传 GitHub 并将报告中的仓库链接替换为真实地址。
