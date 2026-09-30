from typing import Literal

from pydantic import BaseModel, Field


class AfterSalesExtraction(BaseModel):
    """售后诉求结构化结果。"""

    order_no: str | None = Field(
        None, description="用户描述中出现的订单号，没有则为 null"
    )
    issue_type: Literal["退款", "退货", "换货", "维修", "物流投诉", "其他"] = Field(
        description="用户的核心诉求类型"
    )
    expected_resolution: str = Field(description="用户期望的处理方案，一句话概括")
