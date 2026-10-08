"""ch05.6 确定性假数据：同一订单号/商品名跨调用、跨进程答案一致。

回归背景：ch02 起工具每调随机（random.choice/uniform），同一订单号前后
答案漂移（状态/商品/金额/承运商全变），用户质疑「模型是不是在编」。
实为工具吐随机假数据、模型忠实转述。修复后同一实体永远同一套数据。
"""

from app.tools.ecommerce import query_logistics, query_order, query_product


async def test_same_order_always_same_data():
    a = await query_order.ainvoke({"order_no": "1001"})
    b = await query_order.ainvoke({"order_no": "1001"})
    assert a == b  # 状态/商品/金额/创建时间全一致


async def test_same_order_always_same_logistics():
    a = await query_logistics.ainvoke({"order_no": "1001"})
    b = await query_logistics.ainvoke({"order_no": "1001"})
    assert a == b  # 轨迹时刻也一致（锚点不随调用时间漂移）


async def test_same_product_always_same_quote():
    a = await query_product.ainvoke({"product_name": "空气炸锅"})
    b = await query_product.ainvoke({"product_name": "空气炸锅"})
    assert a == b


async def test_different_orders_have_diverse_data():
    rows = [await query_order.ainvoke({"order_no": str(i)}) for i in range(1, 9)]
    signatures = {(r["status"], r["product"], r["amount"]) for r in rows}
    assert len(signatures) > 1  # 8 个不同订单号不可能全部撞同一套签名（1/5^7 概率）