#!/usr/bin/env bash
# ch06 退款单表手动初始化（volume 已存在，docker-entrypoint 不会重跑 init）
set -euo pipefail
cd "$(dirname "$0")/.."

for db in ecom_cs ecom_cs_test; do
  echo "→ $db"
  docker exec -i ecom-cs-mysql mysql -uroot -proot123 --default-character-set=utf8mb4 "$db" < db/init/06-ch06.sql
done
echo "✓ ch06 refund_orders 就绪"
