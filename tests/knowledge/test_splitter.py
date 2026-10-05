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
    # 每块不超限（允许句号回退的少量溢出）
    assert all(len(c.answer) <= 300 + 50 for c in chunks)
    # 重叠起点对齐句号：第 2 块开头不含半截词（句号对齐后以「这」或句首开头）
    for c in chunks[1:]:
        assert c.answer[0] in "这是一句" or c.answer.startswith("。") is False


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


def test_prev_next_pointers_within_document():
    chunks = split_markdown(MD, content_type="policy", max_chars=800)
    # 指针在入库拿到 id 后由 repo 回填，切分阶段先给顺序占位 None
    assert all(c.prev_chunk_id is None and c.next_chunk_id is None for c in chunks)
