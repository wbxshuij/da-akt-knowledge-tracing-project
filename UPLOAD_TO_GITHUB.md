# GitHub 上传完整步骤（交作业版）

> 注意：不要只把 `.zip` 压缩包上传到 GitHub。作业要求仓库里能直接看到代码、数据集和 README，所以需要把解压后的项目文件上传到仓库根目录。

## 一、上传前准备

你需要准备：

1. GitHub 账号；
2. 本项目解压后的文件夹；
3. Git 工具，推荐安装 Git for Windows；
4. 仓库名建议使用：`da-akt-knowledge-tracing-project`；
5. 仓库建议设置为 `Public`，方便老师直接访问。

本项目已经包含作业要求的内容：

| 作业要求 | 本项目对应文件 |
|---|---|
| 代码 | `src/`、`scripts/` |
| 数据集 | `data/sample_interactions.csv` |
| 用法 README | `README.md` |
| 实验结果 | `outputs/test_run/`、`outputs/test_plain_akt/`、`outputs/ablation_comparison.md` |
| 项目报告 | `reports/高级软件工程课程项目报告.docx`、`reports/高级软件工程课程项目报告.md` |

---

## 二、方法 A：命令行上传（推荐）

### 第 1 步：在 GitHub 新建仓库

1. 打开 GitHub；
2. 右上角点击 `+`；
3. 选择 `New repository`；
4. Repository name 填：`da-akt-knowledge-tracing-project`；
5. 选择 `Public`；
6. 不要勾选初始化 README、.gitignore、license，因为本项目已经有这些文件；
7. 点击 `Create repository`。

### 第 2 步：打开终端进入项目目录

Windows 可以在项目文件夹空白处右键，选择 `Open Git Bash here`。

或者使用命令：

```bash
cd DA_AKT_Advanced_SE_Project
```

### 第 3 步：执行上传命令

把下面命令中的 `你的GitHub用户名` 改成你的真实 GitHub 用户名。

```bash
git init
git add .
git commit -m "Initial commit: DA-AKT advanced software engineering project"
git branch -M main
git remote add origin https://github.com/你的GitHub用户名/da-akt-knowledge-tracing-project.git
git push -u origin main
```

如果 GitHub 要求登录，按提示登录即可。

### 第 4 步：复制仓库链接

上传成功后，仓库地址一般是：

```text
https://github.com/你的GitHub用户名/da-akt-knowledge-tracing-project
```

把这个链接复制到课程报告第一部分。

---

## 三、方法 B：网页上传（不用命令行）

网页上传适合不会使用 Git 命令的情况。

1. 先在 GitHub 新建一个空仓库；
2. 进入仓库首页；
3. 点击 `Add file`；
4. 选择 `Upload files`；
5. 打开本项目解压后的文件夹；
6. 全选里面的文件和文件夹，拖到 GitHub 上传区域；
7. 等待上传完成；
8. 底部 Commit message 填：`Initial commit: DA-AKT course project`；
9. 点击 `Commit changes`。

注意：网页上传时不要上传压缩包本身，而是上传解压后的项目内容。

---

## 四、上传后必须检查

上传完成后，仓库首页必须能看到：

```text
README.md
requirements.txt
configs/
data/
scripts/
src/
reports/
outputs/
```

如果只能看到一个 `.zip` 文件，说明上传错了，需要重新上传解压后的文件。

---

## 五、报告里需要填写的 GitHub 链接

在 `reports/高级软件工程课程项目报告.docx` 或 `.md` 中，把：

```text
上传本项目后填写自己的仓库地址
```

替换为你的真实仓库地址，例如：

```text
https://github.com/你的GitHub用户名/da-akt-knowledge-tracing-project
```

---

## 六、老师检查时的运行命令

老师或你自己可以按下面命令复现：

```bash
pip install -r requirements.txt
python scripts/sanity_check_no_leakage.py
python -m src.da_akt_project.train --config configs/test.yaml
python -m src.da_akt_project.train --config configs/test_plain_akt_ablation.yaml
bash run_ablation.sh
```

或者一键自检：

```bash
python scripts/run_project_check.py
```
