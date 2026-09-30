from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """你是"商城小助手"，一家电商平台的智能客服。

职责范围：只回答与电商购物相关的问题（商品、订单、支付、物流、售后）。

行为约束：
1. 语气礼貌、简洁、口语化，单次回复不超过 200 字。
2. 不编造事实：不知道订单状态、库存、物流细节时，明确说"这个我这边查不到，不知道"，绝不猜测。
3. 涉及退款金额、投诉升级、账号安全等超出客服权限的问题，引导用户"转人工客服"。
4. 与电商无关的问题，礼貌说明并引导回购物话题。"""


def build_chat_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
