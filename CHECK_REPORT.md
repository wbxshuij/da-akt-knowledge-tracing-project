# DA-AKT 项目自检、优化与 Demo 验证报告

## 0. 检查结论

本项目已完成重新检查、代码优化与快速 Demo 验证。当前版本能够正常部署运行，代码、数据、配置、说明、实验输出和报告材料齐全。项目定位为：**基于 AKT 改造的难度感知知识追踪与学习诊断系统**，不是 pyKT 的多模型复制。

本次实测命令：

```bash
python -m compileall -q src scripts
python scripts/sanity_check_no_leakage.py
python -m src.da_akt_project.train --config configs/test.yaml
python -m src.da_akt_project.train --config configs/test_plain_akt_ablation.yaml
python scripts/compare_runs.py --run_a outputs/test_run --name_a DA-AKT --run_b outputs/test_plain_akt --name_b Plain-AKT --output outputs/ablation_comparison.md
python scripts/run_project_check.py
```

检查结果：全部通过。

---

## 1. 目录结构是否规范

### 检查结果

当前目录结构合理，代码、数据、配置、说明、报告和实验输出已分离：

```text
DA_AKT_Advanced_SE_Project/
├── README.md                         # 项目总说明
├── requirements.txt                   # Python 依赖
├── run.sh                             # 快速运行 DA-AKT
├── run_ablation.sh                    # 快速运行消融实验
├── CHECK_REPORT.md                    # 原始检查报告
├── PROJECT_AUDIT_REPORT.md            # 本次自检优化报告
├── GITHUB_UPLOAD_GUIDE.md             # GitHub 上传说明
├── configs/                           # 实验配置
├── data/                              # 样例数据与自动生成的 processed 数据
├── scripts/                           # 数据生成、检查、对比脚本
├── src/da_akt_project/                # 核心源码
├── reports/                           # 课程报告材料
└── outputs/                           # Demo 输出结果
```

### 已发现问题与修改方案

| 问题点 | 影响 | 修改方案 | 状态 |
|---|---|---|---|
| `run.sh` 和 `run_ablation.sh` 依赖用户必须在项目根目录执行 | 从其他目录调用脚本时可能找不到配置文件 | 在脚本开头加入 `cd "$(dirname "$0")"` | 已修复 |
| 原检查脚本只验证 DA-AKT 快速训练，不完整 | 不能一次性验证消融实验和输出文件 | 扩展 `scripts/run_project_check.py`，加入语法检查、DA-AKT、Plain-AKT、对比报告和关键文件检查 | 已修复 |
| README 中缺少一键自检说明 | 老师或用户复现时步骤不够集中 | 新增“一键自检与可部署性验证”说明 | 已修复 |

---

## 2. 代码能否正常部署

### 语法与依赖检查

已执行：

```bash
python -m compileall -q src scripts
```

结果：通过，未发现 Python 语法错误。

依赖文件 `requirements.txt` 包含：

```text
torch>=2.0.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
PyYAML>=6.0
tqdm>=4.65.0
```

本次实测环境：Python 3.13.5，CPU 运行通过。

### Demo 训练检查

已执行：

```bash
python -m src.da_akt_project.train --config configs/test.yaml
```

输出目录：

```text
outputs/test_run/
```

生成文件完整：

```text
best_model.pt
metrics.json
history.csv
test_predictions.csv
diagnosis_predictions.csv
concept_mastery.csv
student_diagnosis.csv
experiment_report.md
used_config.json
```

Demo 结果：

| 指标 | 数值 |
|---|---:|
| Best Valid AUC | 0.5996 |
| Test AUC | 0.6335 |
| Test ACC | 0.5917 |
| Test Loss | 0.6765 |
| 训练序列数 | 112 |
| 验证序列数 | 16 |
| 测试序列数 | 32 |
| 可训练参数量 | 103,806 |

### 消融实验检查

已执行：

```bash
bash run_ablation.sh
```

消融结果：

| 模型 | use_difficulty | Test AUC | Test ACC | 参数量 |
|---|---:|---:|---:|---:|
| DA-AKT | True | 0.6335 | 0.5917 | 103,806 |
| Plain-AKT | False | 0.5475 | 0.5281 | 94,846 |

说明：该结果来自内置样例数据，主要用于证明工程流程可运行；正式报告可替换为真实数据集实验结果。

### 防标签泄露检查

已执行：

```bash
python scripts/sanity_check_no_leakage.py
```

输出：

```text
target_position=4, abs_logit_diff_after_flipping_current_label=0.0000000000
PASS: strict previous-interaction protocol has no current-label leakage.
```

结论：预测位置 `t` 时不会使用当前位置答案 `r_t`。

### 已发现问题与修改方案

| 问题点 | 影响 | 修改方案 | 状态 |
|---|---|---|---|
| 没有显式检查 `d_model` 是否能被 `num_attn_heads` 整除 | 参数配置错误时会在模型内部报错，不够直观 | 在训练入口加入配置检查并给出明确错误信息 | 已修复 |
| 验证集/测试集可能因数据过小或 `min_seq_len` 过大变为空 | 指标无意义但用户不一定发现 | 训练入口增加空 split 警告 | 已修复 |
| `scripts/run_project_check.py` 检查覆盖不完整 | 无法一次性确认项目整体可运行 | 扩展为完整自检脚本 | 已修复 |

---

## 3. 数据集格式是否匹配代码读取要求

### 检查结果

项目自带样例数据：

```text
data/sample_interactions.csv
```

字段：

```csv
user_id,question_id,concept_id,correct,timestamp
```

样例数据规模：6135 条交互。

自动预处理后生成：

```text
data/processed_test/train.npz
data/processed_test/valid.npz
data/processed_test/test.npz
data/processed_test/id_maps.json
data/processed_test/question_difficulty_stats.csv
data/processed_test/user_split.csv
data/processed_test/split_summary.csv
```

切分统计：

| split | n_sequences | n_target_interactions |
|---|---:|---:|
| train | 112 | 4139 |
| valid | 16 | 603 |
| test | 32 | 1227 |

### 已发现问题与修改方案

| 问题点 | 影响 | 修改方案 | 状态 |
|---|---|---|---|
| 原来直接 `astype(int)` 处理 `correct`，遇到异常值报错不够清楚 | 数据中存在空值或非法标签时难定位问题 | 改为 `pd.to_numeric(errors="coerce")`，非法标签过滤并给出警告 | 已修复 |
| 原来按 `timestamp` 原字段排序，若时间戳为字符串数字，可能出现 `10` 排在 `2` 前面 | 学生序列顺序可能错误，影响知识追踪建模 | 新增 `_timestamp_sort_key`，优先数值排序，其次日期排序，最后字符串排序 | 已修复 |
| 原来缺少原始数据路径不存在的友好提示 | 用户替换数据集时容易困惑 | 增加 `FileNotFoundError`，提示需要放置 CSV 或修改配置 | 已修复 |
| 原来未输出 split 级别数据统计 | 不方便判断训练/验证/测试是否合理 | 新增 `split_summary.csv` | 已修复 |

---

## 4. 使用说明是否完整

### 检查结果

`README.md` 已覆盖：

1. 项目定位；
2. 与 pyKT / 原 AKT 的区别；
3. 目录结构；
4. 环境安装；
5. 数据格式；
6. 快速运行命令；
7. 防标签泄露检查；
8. 消融实验；
9. ASSIST2009 真实数据部署方式；
10. 输出文件说明；
11. 模型公式说明；
12. 课程报告位置；
13. 一键自检与可部署性验证；
14. 数据集部署注意事项。

`GITHUB_UPLOAD_GUIDE.md` 已包含 GitHub 上传步骤和作业检查清单。

### 已发现问题与修改方案

| 问题点 | 影响 | 修改方案 | 状态 |
|---|---|---|---|
| README 原本缺少统一自检命令 | 用户需要手动执行多个命令 | 加入 `python scripts/run_project_check.py` | 已修复 |
| README 原本对数据异常处理说明不够详细 | 替换真实数据时容易出错 | 补充字段、标签、时间戳、最短序列要求 | 已修复 |
| 原本没有明确说明 Demo 结果来自样例数据 | 容易被误解为真实 ASSIST2009 结果 | 在 README 和消融报告中标注样例数据结果仅用于流程验证 | 已修复 |

---

## 5. 后续优化建议

当前项目已经满足课程作业提交要求。若要继续提升质量，建议如下：

1. **真实数据实验**：将 ASSIST2009 或其他公开数据集整理为项目要求的 CSV 格式，替换 `data/assist2009.csv` 后运行正式配置。
2. **增加图表输出**：根据 `history.csv` 自动画训练曲线图，放入报告中更直观。
3. **增加命令行参数覆盖配置**：当前主要通过 YAML 修改参数，后续可支持命令行覆盖 `epochs`、`lr`、`batch_size` 等。
4. **加入单元测试目录**：可以新增 `tests/`，对数据校验、mask、防泄露、诊断输出分别写测试。
5. **补充 Web 可视化界面**：后续可以用 Streamlit 展示学生诊断、知识点掌握度和预测结果。
6. **真实数据下多次随机种子实验**：正式科研对比建议至少运行 3 个随机种子，报告均值和标准差。

## 6. 最终结论

当前版本已经通过项目结构、语法依赖、数据格式、Demo 训练、消融实验、防泄露检查和说明文档检查。可作为高级软件工程课程项目提交。提交前只需要完成 GitHub 上传，并把报告中的 GitHub 链接替换为个人仓库地址。
