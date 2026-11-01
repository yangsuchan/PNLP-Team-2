"""Answer-free prompt construction for SQA3D.

요약: [1단계 데이터] SQA3D 문항의 input(캡션, 상황, 질문)만으로 모델 프롬프트를 만든다. 정답은 들어가지 않는다. 템플릿은 임시.

`build_prompt` takes a whole processed record but reads ONLY `record["input"]`,
so reference answers, metadata (incl. ground-truth position/rotation) and
source info can never reach the model. scripts/validate_sqa3d.py checks this.

The template is PROVISIONAL: it mirrors the official LLM evaluation prompt
(captions joined with ". ", situation and question joined with a space) so
token counts are comparable to the official GPT-3 baseline. Replace it once the
Qwen3 / long-CoT prompt is decided, then recompute token counts.
"""

PROMPT_TEMPLATE_ID = "sqa3d_official_gpt_v0"
PROMPT_TEMPLATE = """
    Context: there is a book on the desk. A laptop with a green cover is to the left of the book.
    Q: I'm working by the desk. What is on the desk beside the book?
    A: laptop
    Context: {}
    Q: {}
    A:
"""

INPUT_KEYS = frozenset({"scene_id", "situation", "question", "scene_captions"})


def captions_text(scene_captions):
    return ". ".join(c["caption"] for c in scene_captions)


def build_prompt(record):
    inp = record["input"]
    return PROMPT_TEMPLATE.format(
        captions_text(inp["scene_captions"]),
        " ".join([inp["situation"], inp["question"]]),
    )
