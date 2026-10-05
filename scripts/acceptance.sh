#!/usr/bin/env bash
# 验收脚本：SSE 流式 / 上下文续接 / 结构化抽取 / Function Calling
# 用法: 先起服务  uv run uvicorn app.main:app --port 8000
#       另开终端  bash scripts/acceptance.sh
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"

echo "=== 1) SSE 流式回复（逐 token delta 事件）==="
RESP=$(curl -sN -X POST "$BASE/api/chat/stream" -H 'Content-Type: application/json' \
  -d '{"user_id": "acceptance-user", "message": "你好，我想退一台空气炸锅，还没拆封"}')
echo "$RESP"
CID=$(echo "$RESP" | grep -m1 '^data:' | sed 's/^data://' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["conversation_id"])')
echo
echo "conversation_id=$CID"

echo
echo "=== 2) 第二轮验证上下文续接（回复应提到「空气炸锅」）==="
curl -sN -X POST "$BASE/api/chat/stream" -H 'Content-Type: application/json' \
  -d "{\"user_id\": \"acceptance-user\", \"conversation_id\": $CID, \"message\": \"我刚才想退的是什么商品？\"}"

echo
echo "=== 3) 结构化抽取（售后描述 → 固定字段 JSON）==="
curl -s -X POST "$BASE/api/extract" -H 'Content-Type: application/json' \
  -d '{"text": "我的订单A20241230001的空气炸锅还没收到，我要退款，希望全额退回"}'

echo
echo "=== 4) Function Calling（应出现 tool 状态帧后收敛回答）==="
TOOL_STREAM=$(curl -sN -X POST "$BASE/api/chat/stream" -H 'Content-Type: application/json' \
  -d '{"user_id": "acceptance-user", "message": "订单 1001 的物流到哪了"}')
echo "$TOOL_STREAM" | grep -E '^event:' | sort | uniq -c
if echo "$TOOL_STREAM" | grep -q '^event: tool' && echo "$TOOL_STREAM" | grep -q '^event: done'; then
  echo "✓ tool 帧出现且正常收敛"
else
  echo "✗ 未看到 tool 帧或未收敛" && exit 1
fi

echo
echo "=== 验收完成 ==="
