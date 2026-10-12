#!/usr/bin/env bash
# ch08 工具审计表（volume 已存在，需手动应用）
# 表已在则跳过该库；不存在才灌 10-ch08.sql。语句失败时不忽略错误。
set -euo pipefail
cd "$(dirname "$0")/.."

for db in ecom_cs ecom_cs_test; do
  echo "→ $db"
  exists=$(docker exec ecom-cs-mysql mysql -N -uroot -proot123 --default-character-set=utf8mb4 -e \
    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='${db}' AND table_name='tool_audit_logs'")
  if [[ "$exists" == "1" ]]; then
    echo "  skip: tool_audit_logs already exists"
    continue
  fi
  docker exec -i ecom-cs-mysql mysql -uroot -proot123 --default-character-set=utf8mb4 "$db" < db/init/10-ch08.sql
done
echo "✓ ch08 tool_audit_logs 就绪"
