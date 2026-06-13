@echo off
REM 使用前请修改下面两个变量
set GITHUB_USERNAME=你的GitHub用户名
set REPO_NAME=da-akt-knowledge-tracing-project

git init
git add .
git commit -m "Initial commit: DA-AKT advanced software engineering project"
git branch -M main
git remote add origin https://github.com/%GITHUB_USERNAME%/%REPO_NAME%.git
git push -u origin main
pause
