from langchain.messages import AIMessage, HumanMessage

from app.memory.trimmer import count_tokens, trim_history

long = HumanMessage("字" * 500)  # 具体 token 数不假设（BPE 会压缩重复字符）
short = HumanMessage("hi")


def test_within_budget_keeps_all():
    msgs = [short, short]
    assert trim_history(msgs, budget_tokens=1000) == msgs


def test_over_budget_drops_oldest():
    msgs = [long, long, short]
    # 预算恰好只装得下最新两条 → 第一条 long 必被裁
    budget = count_tokens(short) + count_tokens(msgs[1])
    kept = trim_history(msgs, budget_tokens=budget)
    assert kept == [msgs[1], msgs[2]]


def test_keeps_last_round_even_over_budget():
    # 预算只装得下最后一条 AI 回复 → 也必须整轮保留（H+A），不能留孤儿回复
    big_h = HumanMessage("字" * 900)
    a = AIMessage("好的")
    kept = trim_history([big_h, a], budget_tokens=count_tokens(a))
    assert kept == [big_h, a]


def test_always_keeps_last_round_single_message():
    big = HumanMessage("字" * 900)
    assert trim_history([big], budget_tokens=1) == [big]  # 仅一条历史时原样保留
