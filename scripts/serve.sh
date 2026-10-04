#!/usr/bin/env bash
# 一键启动聊天页：bash scripts/serve.sh  （Ctrl-C 停止）
set -euo pipefail
cd "$(dirname "$0")/.."

echo "🛍️  正在启动，浏览器将自动打开 http://localhost:8000"
(sleep 2 && open http://localhost:8000) &

exec uv run uvicorn app.main:app --port 8000
