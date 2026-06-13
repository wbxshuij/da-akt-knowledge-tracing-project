# GitHub 上传说明

由于 GitHub 需要使用你的账号登录，压缩包无法直接替你上传到仓库。你可以按下面步骤完成最后一步。

## 1. 新建仓库

在 GitHub 新建仓库，例如：

```text
da-akt-knowledge-tracing-project
```

建议设置为 Public，方便老师访问。

## 2. 本地上传

进入项目根目录后执行：

```bash
git init
git add .
git commit -m "Initial commit: DA-AKT course project"
git branch -M main
git remote add origin https://github.com/你的用户名/da-akt-knowledge-tracing-project.git
git push -u origin main
```

## 3. 报告中填写链接

上传完成后，把课程报告第一部分中的链接改成：

```text
https://github.com/你的用户名/da-akt-knowledge-tracing-project
```

## 4. 检查仓库是否满足作业要求

仓库需要包含：

- 代码：`src/`、`scripts/`
- 数据集：`data/sample_interactions.csv`
- 使用说明：`README.md`
- 配置文件：`configs/`
- 实验结果：`outputs/test_run/`、`outputs/ablation_comparison.md`
- 项目报告：`reports/高级软件工程课程项目报告.md` 或 `.docx`
