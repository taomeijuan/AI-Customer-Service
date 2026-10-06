#!/usr/bin/env bash
# ch04 两表手动初始化（可重跑：IF NOT EXISTS）
set -euo pipefail
cd "$(dirname "$0")/.."

for db in ecom_cs ecom_cs_test; do
  echo "→ $db"
  docker exec -i ecom-cs-mysql mysql -uroot -proot123 --default-character-set=utf8mb4 "$db" < db/init/05-ch04.sql
done
echo "✓ ch04 两表就绪"
