import re
from dataclasses import dataclass, field

KEY_CLAUSE_KEYWORDS = ("7天", "30天", "仅此一次", "最终解释权", "不支持")
MAX_CHARS_DEFAULT = 800
OVERLAP_CHARS_DEFAULT = 80
SENTENCE_END = "。！？"


@dataclass
class Chunk:
    """一条知识的三个向量化格 + 元数据（prev/next 指针由入库侧回填）。"""

    category: str
    questions: str
    answer: str
    section_path: str = ""
    content_type: str = ""
    is_key_clause: int = 0


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


def _tail_overlap(block: str, overlap_chars: int) -> str:
    """取上一块尾部做重叠，起点回退到「最近」的句号（不是最早的）。"""
    tail = block[-overlap_chars:]
    cut = max((tail.rfind(p) for p in SENTENCE_END), default=-1)
    return tail[cut + 1 :] if cut != -1 else ""


def _split_by_sentences(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    sentences = [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]
    blocks: list[str] = []
    cur = ""
    for s in sentences:
        candidate = cur + s
        if len(candidate) > max_chars and cur:
            blocks.append(cur)
            cur = _tail_overlap(cur, overlap_chars) + s
        else:
            cur = candidate
    if cur:
        blocks.append(cur)
    return blocks


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """超长正文切分：先段落（空行），块间句号对齐重叠；单块仍超长退到句子级。"""
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
        out: list[str] = []
        prev: str | None = None
        for b in blocks:
            if prev is not None:
                b = _tail_overlap(prev, overlap_chars) + b  # 段落块间也带重叠
            out.append(b)
            prev = b
        result: list[str] = []
        for b in out:
            if len(b) > max_chars:
                result.extend(_split_by_sentences(b, max_chars, overlap_chars))
            else:
                result.append(b)
        return result

    return _split_by_sentences(text, max_chars, overlap_chars)


def _emit_section(
    sec: _Section, max_chars: int, overlap_chars: int, table_rows: int
) -> list[Chunk]:
    """一个 section → chunk 列表。按原行序产出：表格连续行成块组（每块复制表头），表格间文本先行。"""
    chunks: list[Chunk] = []
    base = {
        "category": ">".join(sec.path),
        "questions": sec.title,
        "section_path": "/".join([*sec.path, sec.title]),
    }

    def emit_text(lines: list[str]) -> None:
        text = "\n".join(lines).strip()
        if not text:
            return
        for piece in _split_long_text(text, max_chars, overlap_chars):
            chunks.append(Chunk(answer=piece, **base))

    def flush_table(table_lines: list[str]) -> None:
        header = next((ln for ln in table_lines if not _is_separator_row(ln)), table_lines[0])
        data_rows = [ln for ln in table_lines if ln is not header and not _is_separator_row(ln)]
        for i in range(0, len(data_rows), table_rows):
            chunks.append(
                Chunk(answer="\n".join([header] + data_rows[i : i + table_rows]), **base)
            )

    text_buf: list[str] = []
    table_buf: list[str] = []
    for line in sec.lines:
        if _is_table_line(line):
            if text_buf:  # 表格前累积的说明文字先行成块
                emit_text(text_buf)
                text_buf = []
            table_buf.append(line)
        else:
            if table_buf:  # 表格连续行结束 → 成组产出（每张表各自识别表头）
                flush_table(table_buf)
                table_buf = []
            text_buf.append(line)
    if table_buf:
        flush_table(table_buf)
    emit_text(text_buf)

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
