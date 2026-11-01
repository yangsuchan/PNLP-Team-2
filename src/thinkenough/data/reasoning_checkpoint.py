"""Reasoning checkpoints for long-CoT outputs.

요약: [2단계 Checkpoint] Qwen3 출력에서 <think> 추론 부분을 꺼내 문장/줄/번호 단위로 자르고, 각 지점까지의 추론(prefix)을 checkpoint로 만든다.

Pipeline for one generation record:
  1. parse_generation   split `<think>...</think>` reasoning from the final answer,
                        tolerating malformed outputs and recording why (parse_status)
  2. segmenter          return segment spans over the reasoning text; segments are
                        contiguous and cover it exactly, so joining them restores it
  3. build_checkpoints  one checkpoint per segment end (+ index 0 = no reasoning),
                        each with its cumulative reasoning prefix and char offsets
  4. token alignment    only when the ORIGINAL generated token ids are provided:
                        map checkpoint boundaries (end of prefix content, trailing
                        whitespace excluded) to generated-token indices.
                        Without them every token position stays None (null).

Reference answers are never an input here.

IMPORTANT: token alignment is verified only on synthetic token ids (the test
text re-tokenized). Whether it matches real Qwen3 generations, and therefore
the hidden-state positions, must be re-checked on actual Qwen3 outputs.
"""

import re
from dataclasses import dataclass, field

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# ---------------------------------------------------------------- parsing


@dataclass
class ParsedGeneration:
    reasoning_text: str          # stripped text between the think tags ("" if none)
    reasoning_char_offset: int   # where reasoning_text starts inside generated_text
    final_answer: str | None     # text after </think> (stripped), None if absent
    parse_status: str
    warnings: list = field(default_factory=list)


def parse_generation(generated_text, *, think_tag_in_prompt=False):
    """Split a raw generation into reasoning and final answer.

    parse_status values:
      ok                     <think> R </think> A
      missing_open_tag       R </think> A        (template may put <think> in the prompt)
      unclosed_think         <think> R           (no </think>, e.g. cut by max_new_tokens)
      no_think_tags          A                   (no reasoning at all)
      empty_reasoning        <think> </think> A
      empty_output           ""
    A second think block only adds the warning "multiple_think_blocks";
    the first block is used.
    """
    text = generated_text or ""
    warnings = []
    if not text.strip():
        return ParsedGeneration("", 0, None, "empty_output")

    open_idx = text.find(THINK_OPEN)
    close_idx = text.find(THINK_CLOSE, open_idx + len(THINK_OPEN) if open_idx >= 0 else 0)

    if open_idx >= 0 and close_idx >= 0:
        inner_start, inner_end, status = open_idx + len(THINK_OPEN), close_idx, "ok"
    elif open_idx < 0 and close_idx >= 0:
        inner_start, inner_end, status = 0, close_idx, "missing_open_tag"
    elif open_idx >= 0:
        inner_start, inner_end, status = open_idx + len(THINK_OPEN), len(text), "unclosed_think"
    elif think_tag_in_prompt:
        inner_start, inner_end, status = 0, len(text), "unclosed_think"
    else:
        return ParsedGeneration("", 0, text.strip(), "no_think_tags")

    if status != "unclosed_think" and THINK_OPEN in text[close_idx:]:
        warnings.append("multiple_think_blocks")

    inner = text[inner_start:inner_end]
    lead = len(inner) - len(inner.lstrip())
    reasoning = inner.strip()
    answer = None if status == "unclosed_think" else text[close_idx + len(THINK_CLOSE):].strip()
    if not reasoning:
        status = "empty_reasoning" if status == "ok" else status
        warnings.append("no_reasoning_text")
    return ParsedGeneration(reasoning, inner_start + lead, answer, status, warnings)


# ---------------------------------------------------------------- segmenters
# A segmenter maps reasoning text -> list of (start, end) char spans that are
# contiguous and cover the text exactly. Trailing whitespace stays with the
# preceding segment. Register new strategies in SEGMENTERS.

# only abbreviations usually followed by a capital/digit; lowercase continuations are handled separately
_ABBREVIATIONS = re.compile(r"\b(?:e\.g|i\.e|vs|approx|cf|fig|dr|mr|mrs|ms)\.$", re.IGNORECASE)
_LIST_MARKER_BEFORE = re.compile(r"(?:^|\n)[ \t]*\d+\.$")
_STEP_MARKER = re.compile(r"^[ \t]*(?:\d+[.)]|step\s*\d+\s*[:.)]|[-*•])[ \t]+", re.IGNORECASE | re.MULTILINE)


def _spans_from_cuts(text, cuts):
    """Turn cut positions into contiguous spans; whitespace-only pieces merge left."""
    spans, start = [], 0
    for cut in sorted(set(c for c in cuts if 0 < c < len(text))) + [len(text)]:
        if text[start:cut].strip():
            spans.append([start, cut])
            start = cut
        elif spans:
            spans[-1][1] = cut
            start = cut
    if spans and spans[-1][1] < len(text):
        spans[-1][1] = len(text)
    return [tuple(s) for s in spans]


def _after_whitespace(text, i):
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def segment_sentences(text):
    """Sentence boundaries: [.!?] (+ closing quotes/brackets) followed by whitespace, or a newline.

    Not a boundary: decimals (3.5), abbreviations (e.g., approx.), list markers
    ("1." at line start), and a period whose next word starts lowercase.
    """
    cuts = []
    for m in re.finditer(r"[.!?]+[\"')\]]*(?=\s)|\n", text):
        end = m.end()
        if m.group() != "\n" and m.group()[0] == ".":
            before = text[: m.start() + 1]
            if _ABBREVIATIONS.search(before) or _LIST_MARKER_BEFORE.search(before):
                continue
            nxt = _after_whitespace(text, end)
            if nxt < len(text) and text[nxt].islower() and "\n" not in text[end:nxt]:
                continue
        cuts.append(_after_whitespace(text, end))
    return _spans_from_cuts(text, cuts)


def segment_lines(text):
    """One segment per non-empty line."""
    return _spans_from_cuts(text, [_after_whitespace(text, m.end()) for m in re.finditer(r"\n", text)])


def segment_numbered_steps(text):
    """Cut before lines starting with a step marker (1. / 2) / Step 3: / - / * / •).

    Text before the first marker is its own segment. If there are no markers
    the whole text is one segment.
    """
    return _spans_from_cuts(text, [m.start() for m in _STEP_MARKER.finditer(text)])


SEGMENTERS = {
    "sentence": segment_sentences,
    "line": segment_lines,
    "numbered_step": segment_numbered_steps,
    # extension points: "discourse" (cut before "Wait,", "Hmm," ...), min/max segment tokens
}
SEGMENTER_VERSION = "v1"


# ---------------------------------------------------------------- token alignment


def _bytes_to_unicode():
    """GPT-2 byte-level BPE table (Qwen tokenizers use the same scheme)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class QwenTokenAligner:
    """Char offsets of each ORIGINAL generated token inside the decoded text.

    Works on the byte level, so a multi-byte character split over tokens is
    handled. Returns None when its decoded text differs from tokenizer.decode.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.byte_decoder = {v: k for k, v in _bytes_to_unicode().items()}
        self.added = {i: t.content for i, t in tokenizer.added_tokens_decoder.items()}

    def token_bytes(self, token_id):
        if token_id in self.added:
            return self.added[token_id].encode("utf-8")
        token = self.tokenizer.convert_ids_to_tokens(token_id)
        return bytes(self.byte_decoder[c] for c in token)

    def offsets(self, token_ids):
        byte_spans, buf = [], bytearray()
        for tid in token_ids:
            b = self.token_bytes(tid)
            byte_spans.append((len(buf), len(buf) + len(b)))
            buf.extend(b)
        try:
            text = buf.decode("utf-8")
        except UnicodeDecodeError:
            return None
        if text != self.tokenizer.decode(token_ids, skip_special_tokens=False):
            return None
        byte_to_char = []
        for i, ch in enumerate(text):
            byte_to_char.extend([i] * len(ch.encode("utf-8")))
        byte_to_char.append(len(text))
        return text, [(byte_to_char[s], byte_to_char[e]) for s, e in byte_spans]


def align_boundary(offsets, char_pos):
    """Exclusive token end so that tokens[:end] cover text[:char_pos].

    Returns (end, exact) where exact is False if char_pos falls inside a token
    (that token is then included).
    """
    for k, (s, e) in enumerate(offsets):
        if e >= char_pos and s < char_pos:
            return k + 1, e == char_pos
        if s >= char_pos:
            return k, True
    return len(offsets), True


# ---------------------------------------------------------------- checkpoints


def checkpoint_id(run_id, question_id, segmenter, index):
    """Unique key that links a checkpoint to its forced answers and labels."""
    from urllib.parse import quote
    identity = f"s:{quote(question_id, safe='')}" if isinstance(question_id, str) else str(question_id)
    return f"{run_id}/{identity}/{segmenter}/{index}"


def build_checkpoints(record, segmenter="sentence", tokenizer=None, aligner=None, keep_prefix_text=True):
    """Checkpoints for one generation record (reference answers are never read).

    record: dict in the format of docs/qwen3_output_spec.md (question_id, split,
            generated_text, optional generated_token_ids / prompt_token_count ...).
    tokenizer: used only for re-tokenized token counts when original ids are absent.
    aligner: QwenTokenAligner; used only when record["generated_token_ids"] is given.
    """
    parsed = parse_generation(
        record.get("generated_text"),
        think_tag_in_prompt=record.get("generation_config", {}).get("think_tag_in_prompt", False),
    )
    text = parsed.reasoning_text
    spans = SEGMENTERS[segmenter](text) if text else []

    token_ids = record.get("generated_token_ids")
    alignment, offsets = "no_token_ids", None
    if token_ids:
        if aligner is None:
            alignment = "no_aligner"
        else:
            result = aligner.offsets(token_ids)
            if result is None:
                alignment = "decode_failed"
            elif result[0] != record.get("generated_text"):
                alignment = "text_mismatch"
            else:
                offsets = result[1]
                alignment = "aligned"

    ends = [0] + [e for _, e in spans]
    starts = [0] + [s for s, _ in spans]
    prefixes = [text[:e] for e in ends]
    if offsets is None and tokenizer is not None:
        counts = [len(ids) for ids in tokenizer(prefixes, add_special_tokens=False)["input_ids"]] if prefixes else []
        count_source = "retokenized"
    else:
        counts, count_source = [None] * len(ends), ("generated_ids" if offsets else None)

    base = record.get("generated_text") or ""
    tok_start = None
    if offsets is not None:
        tok_start, start_exact = align_boundary(offsets, parsed.reasoning_char_offset)
        if not start_exact:
            alignment = "start_inside_token"
    prompt_len = record.get("prompt_token_count")
    run_id = record.get("run_id") or "norun"

    checkpoints = []
    for k, (seg_start, end) in enumerate(zip(starts, ends)):
        tok_end, status = None, alignment
        # align on the end of the prefix CONTENT: trailing whitespace belongs to the
        # next token in Qwen's tokenization (" So"), which must not be included
        content_end = parsed.reasoning_char_offset + len(prefixes[k].rstrip())
        if offsets is not None:
            tok_end, exact = align_boundary(offsets, content_end)
            tok_end = max(tok_end, tok_start)
            if alignment != "aligned":
                status = alignment
            elif exact:
                status = "exact"
            elif not base[content_end:offsets[tok_end - 1][1]].strip():
                status = "exact_up_to_whitespace"   # e.g. '.\n\n' is one token
            else:
                status = "boundary_inside_token"    # last token also holds next-segment text
            counts[k] = tok_end - tok_start
        checkpoints.append({
            "checkpoint_id": checkpoint_id(run_id, record.get("question_id"), segmenter, k),
            "run_id": run_id,
            "question_id": record.get("question_id"),
            "split": record.get("split"),
            "checkpoint_index": k,
            "num_checkpoints": len(ends),
            "is_final": k == len(ends) - 1,
            "checkpoint_type": "no_reasoning" if k == 0 else segmenter,
            "segment_text": text[seg_start:end] if k else "",
            "reasoning_prefix": prefixes[k] if keep_prefix_text else None,
            "char_start": 0,
            "char_end": end,
            "segment_char_start": seg_start if k else 0,
            "prefix_token_count": counts[k],
            "token_count_source": count_source,
            "reasoning_token_start": tok_start,
            "reasoning_token_end": tok_end,
            # position of the last token of the prefix in the full (prompt + generation) sequence
            "hidden_state_position": (prompt_len + tok_end - 1) if (tok_end is not None and prompt_len is not None) else None,
            "alignment_status": status,
            "parse_status": parsed.parse_status,
            "parse_warnings": parsed.warnings,
            "reasoning_char_offset": parsed.reasoning_char_offset,
            "alignment_char_end": content_end,  # absolute char used for token alignment
            "segmenter": f"{segmenter}/{SEGMENTER_VERSION}",
            "source": {
                "model": record.get("model"),
                "tokenizer": record.get("tokenizer"),
                "prompt_version": record.get("prompt_version"),
                "synthetic": record.get("synthetic", False),
            },
        })
    assert base[parsed.reasoning_char_offset: parsed.reasoning_char_offset + len(text)] == text
    return checkpoints, parsed
