#!/usr/bin/env bash
set -e

# 使用前请修改这两个变量
GITHUB_USERNAME="你的GitHub用户名"
REPO_NAME="da-akt-knowledge-tracing-project"

cd "$(dirname "$0")"

git init
git add .
git commit -m "Initial commit: DA-AKT advanced software engineering project"
git branch -M main
git remote add origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
git push -u origin main
