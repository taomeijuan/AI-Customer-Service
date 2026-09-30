from langchain.messages import HumanMessage

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


def test_only_newest_fits():
    msgs = [long, long, short]
    # 预算连「short + 一条 long」都装不下 → 只留最新一条
    budget = count_tokens(short) + 1
    kept = trim_history(msgs, budget_tokens=budget)
    assert kept == [msgs[2]]


def test_always_keeps_last_even_if_over():
    big = HumanMessage("字" * 900)
    kept = trim_history([big], budget_tokens=100)
    assert kept == [big]  # min_keep：至少保留最近 1 条
