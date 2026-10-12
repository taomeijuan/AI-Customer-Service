#!/usr/bin/env bash
# ch08 工具审计表（volume 已存在，需手动应用）
set -euo pipefail
cd "$(dirname "$0")/.."

for db in ecom_cs ecom_cs_test; do
  echo "→ $db"
  docker exec -i ecom-cs-mysql mysql --force -uroot -proot123 --default-character-set=utf8mb4 "$db" < db/init/10-ch08.sql
done
echo "✓ ch08 tool_audit_logs 就绪"
