#!/usr/bin/env bash
# ch08 演示。下面每条单独开一个终端；不要在同一条 shell 里顺序执行，前两条会占住进程。
# MCP Server 没起来时应用仍能启动，工具在对话时现拉。

uv run python mcp_servers/logistics_server.py
uv run python mcp_servers/aftersales_server.py
# other terminal
uv run uvicorn app.main:app --reload

# 人为超时：只拖慢名单里的工具。读工具会重试，写工具不重试。
# TOOL_DELAY_TOOLS=query_order TOOL_DELAY_SECONDS=5 uv run uvicorn app.main:app
# TOOL_DELAY_TOOLS=create_ticket TOOL_DELAY_SECONDS=5 uv run uvicorn app.main:app
