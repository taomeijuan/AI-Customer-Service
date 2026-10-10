-- ch07 补充：messages 表加引用快照列（幂等 ALTER，可重复执行）
-- 场景：侧栏回载历史时还原 markdown 与可点击引用；旧实现只存正文，[n] 变无主死文本
SET NAMES utf8mb4;
SET @c := (SELECT COUNT(*) FROM information_schema.COLUMNS
           WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'messages' AND COLUMN_NAME = 'citations');
SET @s := IF(@c = 0,
  "ALTER TABLE messages ADD COLUMN citations JSON NULL COMMENT '该条回复的引用快照[{n,chunk_id,section_path,question,answer}]' AFTER tool_call_id",
  "SELECT 'citations column exists' AS note");
PREPARE st FROM @s; EXECUTE st; DEALLOCATE PREPARE st;
