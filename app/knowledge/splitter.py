import re
from dataclasses import dataclass, field

KEY_CLAUSE_KEYWORDS = ("7天", "30天", "仅此一次", "最终解释权", "不支持")
MAX_CHARS_DEFAULT = 800
OVERLAP_CHARS_DEFAULT = 80
SENTENCE_END = "。！？"


@dataclass
class Chunk:
    """一条知识的三个向量化格 + 四类元数据。"""

    category: str
    questions: str
    answer: str
    section_path: str = ""
    content_type: str = ""
    is_key_clause: int = 0
    prev_chunk_id: int | None = None
    next_chunk_id: int | None = None


@dataclass
class _Section:
    """标题栈切出的一个 section：路径 + 标题 + 正文行（含表格行）。"""

    path: list[str]
    title: str
    lines: list[str] = field(default_factory=list)


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _is_separator_row(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:\-|]+\|", line.strip()))


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """超长正文递归切：先段落（空行），再句号；块间重叠回退到最近句号。"""
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1:
        blocks: list[str] = []
        cur = ""
        for p in paragraphs:
            candidate = f"{cur}\n{p}".strip() if cur else p
            if len(candidate) > max_chars and cur:
                blocks.append(cur)
                cur = p
            else:
                cur = candidate
        if cur:
            blocks.append(cur)
        if len(blocks) == 1:  # 单段落仍超长 → 退到句子级
            blocks = _split_long_text(text, max_chars, overlap_chars)
        return [b for blk in blocks for b in _split_long_text(blk, max_chars, overlap_chars)] if any(
            len(b) > max_chars for b in blocks
        ) else blocks

    # 句子级切分
    sentences = [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]
    blocks = []
    cur = ""
    for s in sentences:
        candidate = cur + s
        if len(candidate) > max_chars and cur:
            blocks.append(cur)
            # 重叠：取上一块尾部 overlap_chars，并回退到最近句号之后
            tail = cur[-overlap_chars:]
            cut = min((tail.rfind(p) for p in SENTENCE_END), default=-1)
            overlap = tail[cut + 1 :] if cut != -1 else ""
            cur = overlap + s
        else:
            cur = candidate
    if cur:
        blocks.append(cur)
    return blocks


def _emit_section(sec: _Section, max_chars: int, overlap_chars: int, table_rows: int) -> list[Chunk]:
    """一个 section → chunk 列表：普通文本块 / 表格块（带表头）。"""
    chunks: list[Chunk] = []
    category = ">".join(sec.path)
    section_path = "/".join([*sec.path, sec.title])  # 全路径含本节标题，溯源用
    base = {
        "category": category,
        "questions": sec.title,
        "section_path": section_path,
    }

    # 表格行（连续 | 行）单独成块组；其余文本走普通切分
    table_lines: list[str] = []
    text_lines: list[str] = []
    for line in sec.lines:
        (table_lines if _is_table_line(line) else text_lines).append(line)

    if table_lines:
        header = next((ln for ln in table_lines if not _is_separator_row(ln)), table_lines[0])
        data_rows = [ln for ln in table_lines if ln is not header and not _is_separator_row(ln)]
        for i in range(0, len(data_rows), table_rows):
            body = "\n".join([header] + data_rows[i : i + table_rows])
            chunks.append(Chunk(answer=body, **base))

    text = "\n".join(text_lines).strip()
    if text:
        for piece in _split_long_text(text, max_chars, overlap_chars):
            chunks.append(Chunk(answer=piece, **base))

    for c in chunks:
        c.is_key_clause = int(any(kw in c.answer for kw in KEY_CLAUSE_KEYWORDS))
    return chunks


def split_markdown(
    md: str,
    content_type: str,
    max_chars: int = MAX_CHARS_DEFAULT,
    overlap_chars: int = OVERLAP_CHARS_DEFAULT,
    table_rows_per_chunk: int = 3,
) -> list[Chunk]:
    """结构感知切分：按标题层级切 section，产出 Chunk 列表（prev/next 由入库侧回填）。"""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []  # (层级, 标题)

    for line in md.splitlines():
        stripped = line.strip()
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            stack = [(lv, t) for lv, t in stack if lv < level]
            stack.append((level, title))
            sections.append(
                _Section(path=[t for _, t in stack[:-1]] or [title], title=title)
            )
        elif sections:
            sections[-1].lines.append(stripped)

    chunks: list[Chunk] = []
    for sec in sections:
        for c in _emit_section(sec, max_chars, overlap_chars, table_rows_per_chunk):
            c.content_type = content_type
            chunks.append(c)
    return chunks
