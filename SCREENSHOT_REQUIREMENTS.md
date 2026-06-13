# 作业截图要求与截图清单

你上传 GitHub 后，建议至少截图 5 张。截图时浏览器地址栏最好露出来，这样老师能看到仓库链接。

## 截图 1：GitHub 仓库首页

目的：证明项目已经上传到 GitHub。

截图内容需要包含：

- 浏览器地址栏中的仓库链接；
- 仓库名 `da-akt-knowledge-tracing-project`；
- 根目录文件列表；
- `README.md` 预览区域。

建议命名：

```text
截图1_GitHub仓库首页.png
```

---

## 截图 2：代码目录截图

进入仓库中的 `src/da_akt_project/` 目录截图。

截图内容需要包含：

- `model.py`；
- `data.py`；
- `train.py`；
- `diagnosis.py`；
- GitHub 地址栏。

这张图证明仓库里有完整代码。

建议命名：

```text
截图2_代码目录.png
```

---

## 截图 3：数据集目录截图

进入仓库中的 `data/` 目录截图。

截图内容需要包含：

- `sample_interactions.csv`；
- `processed_test/`；
- `processed_test_plain/`。

这张图证明仓库中包含数据集和处理后的样例数据。

建议命名：

```text
截图3_数据集目录.png
```

---

## 截图 4：README 使用说明截图

在仓库首页向下滚动到 README，或打开 `README.md` 文件截图。

截图内容需要包含：

- 环境安装命令；
- 快速运行命令；
- 数据格式说明；
- 实验结果输出说明。

这张图证明仓库里有用法 README 文件。

建议命名：

```text
截图4_README使用说明.png
```

---

## 截图 5：实验结果目录截图

进入仓库中的 `outputs/` 或 `outputs/test_run/` 目录截图。

截图内容需要包含：

- `metrics.json`；
- `history.csv`；
- `test_predictions.csv`；
- `student_diagnosis.csv`；
- `concept_mastery.csv`。

这张图证明项目有实验结果。

建议命名：

```text
截图5_实验结果目录.png
```

---

## 可选截图 6：本地运行成功截图

在终端运行：

```bash
python scripts/run_project_check.py
```

等出现全部通过后截图。

这张图可以放在报告“实验结果”部分，证明项目可以正常部署和运行。

建议命名：

```text
截图6_本地运行成功.png
```

---

## 报告里如何放截图

如果老师要求 Word 报告，可以把截图放在：

1. 第一部分 GitHub 链接后：放“仓库首页截图”；
2. 第二部分项目功能介绍后：放“代码目录截图”；
3. 第三部分实验结果后：放“实验结果目录截图”或“终端运行成功截图”；
4. 附录部分：放 README、数据集目录截图。
