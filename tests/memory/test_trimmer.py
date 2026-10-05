from langchain.messages import AIMessage, HumanMessage, ToolMessage

from app.memory.trimmer import count_tokens, trim_history, trim_history_groups

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


# ── trim_history_groups：工具往返原子裁剪（ch02 评审 Major 1）──


def _tool_round():
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "query_faq", "args": {"keyword": "退货"}, "id": "c1"}],
        ),
        ToolMessage(content="[]", tool_call_id="c1", name="query_faq"),
    ]


def test_groups_cut_round_atomically():
    h1, (a1, t1), h2 = long, _tool_round(), short
    msgs = [h1, a1, t1, h2]
    # 预算只装得下最后一轮 → 老轮整轮裁，无孤儿 tool 开头
    kept = trim_history_groups(msgs, budget_tokens=count_tokens(h2))
    assert kept == [h2]


def test_groups_tool_round_never_split_by_mid_boundary():
    h = HumanMessage("字" * 100)
    a, t = _tool_round()
    # 预算只够 tool 一条 → 整轮保留优先，绝不裁成 [t]
    assert trim_history_groups([h, a, t], budget_tokens=count_tokens(t)) == [h, a, t]


def test_groups_keep_last_round_even_if_huge():
    big_h = HumanMessage("字" * 900)
    a, t = _tool_round()
    assert trim_history_groups([big_h, a, t], budget_tokens=10) == [big_h, a, t]
