"""Extract the short final answer from a Qwen3 output (reasoning is never scored).

요약: [3·4단계 공용] 모델 출력에서 최종 짧은 답만 뽑는다. 라벨 만들기와 평가가 같은 규칙을 쓴다.

Shared by checkpoint labeling (forced answers) and evaluation (final answers),
so both use exactly the same rule:

  1. if "</think>" occurs, keep only the text after the LAST "</think>";
     if "<think>" occurs without a closing tag, there is no answer (think_unclosed)
  2. drop chat special tokens (<|im_end|>, <|endoftext|>, ...)
  3. keep the first non-empty line
  4. drop a leading label ("Answer:", "A:", "Final answer:", "The answer is")
  5. drop markdown emphasis / surrounding quotes and trailing . ! ;
Scoring-level normalization (lowercase, articles, digits -> words, ...) is NOT
done here; it is done by sqa3d_scoring.clean_answer.

Status values: ok, empty, think_unclosed, no_output.
"""

import re

_SPECIAL = re.compile(r"<\|[a-z_]+\|>")
_LABEL = re.compile(r"^(?:final\s+answer|answer|the\s+answer\s+is)\s*[:：\-]?\s+", re.IGNORECASE)
_LABEL_ONLY = re.compile(r"^(?:final\s+answer|answer|a)\s*[:：]\s*", re.IGNORECASE)
EXTRACTION_VERSION = "v1"


def extract_final_answer(text):
    """Return (answer, status). The answer is "" unless status == "ok"."""
    if text is None:
        return "", "no_output"
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return "", "think_unclosed"
    text = _SPECIAL.sub("", text)
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    line = _LABEL_ONLY.sub("", line)
    line = _LABEL.sub("", line) if _LABEL.match(line) and len(line.split()) > 1 else line
    for _ in range(2):  # "**two**." -> "two"
        line = re.sub(r"[.!;。]+$", "", line.strip()).strip()
        line = line.strip("*_`").strip().strip("\"'“”").strip()
    return (line, "ok") if line else ("", "empty")
