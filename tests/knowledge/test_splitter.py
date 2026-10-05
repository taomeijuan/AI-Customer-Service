from app.knowledge.splitter import split_markdown

MD = """# 售后政策

## 退款
### 退款时限
退款审核通过后1-3个工作日原路退回。超过7天未到账请联系客服。

### 退款方式
原路退回。

## 运费说明
普通订单满99元包邮，否则收取8元运费。偏远地区另计。
"""


def test_title_hierarchy_to_chunks():
    chunks = split_markdown(MD, content_type="policy", max_chars=800)
    refund = [c for c in chunks if c.questions == "退款时限"]
    assert len(refund) == 1
    assert refund[0].category == "售后政策>退款"
    assert "1-3个工作日" in refund[0].answer
    assert refund[0].section_path == "售后政策/退款/退款时限"
    assert refund[0].content_type == "policy"


def test_long_section_recursive_with_sentence_aligned_overlap():
    body = "这是一句话。" * 100  # 约 600 字
    md = f"# 政策\n## 长条款\n{body}"
    chunks = split_markdown(md, content_type="policy", max_chars=300)
    assert len(chunks) >= 2
    assert all(len(c.answer) <= 300 + 50 for c in chunks)
    # 重叠起点对齐句号：除首块外，每块开头必须是完整句首（「这」开头）而非半截
    for c in chunks[1:]:
        assert c.answer.startswith("这是一句话。") or c.answer[0] == "这"


def test_overlap_uses_nearest_sentence():
    # 尾部窗口内有多个句号：重叠起点必须取「最近」的句号（评审 Minor#5）
    text = "甲" * 10 + "。" + "乙" * 10 + "。" + "丙" * 260
    chunks = split_markdown(f"# 政策\n## 节\n{text}", content_type="policy", max_chars=200)
    assert len(chunks) >= 2
    assert chunks[1].answer.startswith("丙")  # 最近句号之后才是第二块开头


def test_table_rows_carry_header():
    md = """# 手册
## 退换货对照表
| 情形 | 时限 | 说明 |
| --- | --- | --- |
| 质量问题 | 30天 | 免费退换 |
| 不喜欢 | 7天 | 自担运费 |
| 临期商品 | 不支持 | 特殊商品 |
"""
    chunks = split_markdown(md, content_type="manual", table_rows_per_chunk=2)
    table_chunks = [c for c in chunks if "对照表" in (c.questions or "")]
    assert len(table_chunks) == 2  # 3 行数据按 2 行/块 → 2 块
    for c in table_chunks:
        assert c.answer.startswith("| 情形 | 时限 | 说明 |")  # 每块复制表头


def test_two_tables_in_one_section_each_get_own_header():
    """评审 Major 6：同节多张表，第二张表的表头不得被当作数据行。"""
    md = """# 手册
## 对照表
| A | B |
| --- | --- |
| a1 | b1 |

| C | D |
| --- | --- |
| c1 | d1 |
"""
    chunks = split_markdown(md, content_type="manual")
    assert len(chunks) == 2
    assert chunks[0].answer.startswith("| A | B |") and "a1" in chunks[0].answer
    assert chunks[1].answer.startswith("| C | D |") and "c1" in chunks[1].answer
    assert "C | D" not in chunks[0].answer  # 第一块不得混入第二张表内容


def test_text_before_table_keeps_order():
    """评审 Minor#6：节内「说明文字 + 表格」按行序产出，说明先行。"""
    md = """# 手册
## 退换货对照表
下表列出常见情形：
| 情形 | 时限 |
| --- | --- |
| 质量问题 | 30天 |
"""
    chunks = split_markdown(md, content_type="manual")
    assert chunks[0].answer.startswith("下表列出")
    assert chunks[1].answer.startswith("| 情形 | 时限 |")


def test_slightly_over_limit_paragraph_no_recursion():
    """评审 Major 5：段落拼接仅超限 1-2 字时不得无限递归。"""
    body = "字" * 400 + "\n\n" + "字" * 399
    chunks = split_markdown(f"# 政策\n## 长节\n{body}", content_type="policy", max_chars=800)
    assert len(chunks) >= 1  # 能正常返回即无 RecursionError


def test_key_clause_flag():
    md = "# 政策\n## 特殊商品\n临期商品不支持7天无理由退货。\n## 普通退货\n按流程办理。"
    chunks = split_markdown(md, content_type="policy")
    by_q = {c.questions: c for c in chunks}
    assert by_q["特殊商品"].is_key_clause == 1
    assert by_q["普通退货"].is_key_clause == 0


def test_faq_real_questions():
    md = """# 商品FAQ
## 空气炸锅可以用洗碗机洗吗
内胆可以，主机不可以。
"""
    chunks = split_markdown(md, content_type="faq")
    assert chunks[0].questions == "空气炸锅可以用洗碗机洗吗"
    assert chunks[0].category == "商品FAQ"
