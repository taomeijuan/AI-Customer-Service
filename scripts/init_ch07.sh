#!/usr/bin/env bash
# ch07 上下文管理两表变更（volume 已存在，需手动应用；两个 SQL 都要跑）
set -euo pipefail
cd "$(dirname "$0")/.."

for db in ecom_cs ecom_cs_test; do
  echo "→ $db"
  docker exec -i ecom-cs-mysql mysql -uroot -proot123 --default-character-set=utf8mb4 "$db" < db/init/07-ch07.sql
  docker exec -i ecom-cs-mysql mysql -uroot -proot123 --default-character-set=utf8mb4 "$db" < db/init/07-ch07-layers.sql
done
echo "✓ ch07 conversations 三列 + conversation_summaries 就绪"
