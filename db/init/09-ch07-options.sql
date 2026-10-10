-- ch07 补充：messages 表加 options 快照列（幂等，可重复执行）
-- 场景：侧栏回载要"所见即所存"——当轮的按钮组（转人工/建工单/申请退款+订单摘要）随行持久化
SET NAMES utf8mb4;
SET @c := (SELECT COUNT(*) FROM information_schema.COLUMNS
           WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'messages' AND COLUMN_NAME = 'options');
SET @s := IF(@c = 0,
  "ALTER TABLE messages ADD COLUMN options JSON NULL COMMENT 'ch07 按钮组快照{options:[...],order:{...}}' AFTER citations",
  "SELECT 'options column exists' AS note");
PREPARE st FROM @s; EXECUTE st; DEALLOCATE PREPARE st;
