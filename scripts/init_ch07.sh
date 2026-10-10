#!/usr/bin/env bash
# ch07 上下文管理两表变更（volume 已存在，需手动应用；两个 SQL 都要跑）
set -euo pipefail
cd "$(dirname "$0")/.."

for db in ecom_cs ecom_cs_test; do
  echo "→ $db"
  docker exec -i ecom-cs-mysql mysql --force -uroot -proot123 --default-character-set=utf8mb4 "$db" < db/init/07-ch07.sql
  docker exec -i ecom-cs-mysql mysql --force -uroot -proot123 --default-character-set=utf8mb4 "$db" < db/init/07-ch07-layers.sql
  docker exec -i ecom-cs-mysql mysql --force -uroot -proot123 --default-character-set=utf8mb4 "$db" < db/init/08-ch07-citations.sql
  docker exec -i ecom-cs-mysql mysql --force -uroot -proot123 --default-character-set=utf8mb4 "$db" < db/init/09-ch07-options.sql
done
echo "✓ ch07 上下文表结构与 messages.citations 就绪"
