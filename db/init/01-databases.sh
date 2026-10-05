#!/usr/bin/env bash
# 建业务库与测试伴生库；docker-entrypoint 初始化阶段以 root 执行
set -e
mysql -uroot -proot123 --default-character-set=utf8mb4 << 'EOF'
CREATE DATABASE IF NOT EXISTS ecom_cs DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE DATABASE IF NOT EXISTS ecom_cs_test DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;
GRANT ALL PRIVILEGES ON ecom_cs_test.* TO 'ecom'@'%';
FLUSH PRIVILEGES;
EOF
