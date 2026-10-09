-- ch06 退款单（用户授权自行设计，2026-10-09）
-- 语义：退款子流程终点产物；原因走固定类目，不追问用户
CREATE TABLE IF NOT EXISTS `refund_orders` (
  `refund_no` varchar(32) NOT NULL COMMENT '退款单号,如 R20261009001',
  `conversation_id` bigint unsigned NOT NULL COMMENT '关联会话,可倒查当时聊了什么',
  `order_no` varchar(32) NOT NULL COMMENT '平台订单号,如 1001',
  `reason_category` enum('七天无理由','质量问题','少件','与描述不符','其他') NOT NULL COMMENT '退款原因固定类目',
  `amount` decimal(10,2) NOT NULL COMMENT '退款金额',
  `status` enum('待审核','已同意','已拒绝','已退款') NOT NULL DEFAULT '待审核' COMMENT '处理状态',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`refund_no`),
  KEY `idx_conversation_id` (`conversation_id`),
  KEY `idx_order_no` (`order_no`),
  CONSTRAINT `fk_refunds_conversation` FOREIGN KEY (`conversation_id`) REFERENCES `conversations` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='退款单';

-- ch06 意图置信度入池：source 枚举扩宽（意图低置信成为新数据源）
ALTER TABLE `low_confidence_questions`
  MODIFY COLUMN `source` enum('retrieval_low_conf','self_check','user_feedback','intent_low_conf') NOT NULL COMMENT '低置信来源';
