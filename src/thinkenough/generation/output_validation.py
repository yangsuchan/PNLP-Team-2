"""Validate the draft Qwen3 handoff contract without loading model weights."""
from thinkenough.identity import valid_question_id

import json
import math
from collections import Counter
from pathlib import Path

OPEN, CLOSE = 151667, 151668
STRINGS = ('run_id', 'model', 'tokenizer', 'quantization', 'prompt_version', 'prompt_text', 'generated_text', 'finish_reason', 'split')
REQUIRED = STRINGS + ('question_id', 'enable_thinking', 'prompt_token_ids', 'generated_token_ids', 'prompt_token_count', 'generated_token_count', 'reasoning_token_count', 'think_close_token_index', 'final_answer_text')
FORBIDDEN = {'reference_answers', 'answers', 'answer', 'ground_truth', 'gold_answer', 'gold_answers'}

def integer(v):
    return type(v) is int and v >= 0

def token_list(v):
    return isinstance(v, list) and all(integer(x) for x in v)

def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('non-finite JSON number: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)

def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        for line, text in enumerate(handle, 1):
            try:
                yield line, strict_json(text), None
            except (ValueError, RecursionError) as exc:
                yield line, None, str(exc)

def forbidden_paths(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix + '.' + key if prefix else key
            if key.lower() in FORBIDDEN:
                yield path
            yield from forbidden_paths(child, path)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            if isinstance(child, (dict, list)):
                yield from forbidden_paths(child, f'{prefix}[{i}]')

class GenerationValidator:
    """One instance per JSONL. decode(ids) must preserve special tokens."""
    def __init__(self, references=None, run_config=None, decode=None, tokenizer_id=None):
        self.references = references
        self.run_config = run_config or {}
        self.decode = decode
        self.tokenizer_id = tokenizer_id
        self.seen = set()
        self.runs = {}
        self.issues = []
        self.records = 0
        self.valid_objects = 0

    def issue(self, line, code, message, severity='error'):
        self.issues.append(dict(line=line, severity=severity, code=code, message=message))

    def validate(self, r, line=1):
        self.records += 1
        def error(code, message):
            self.issue(line, code, message)
        if not isinstance(r, dict):
            error('record_type', 'Each line must be a JSON object.')
            return
        self.valid_objects += 1
        for key in REQUIRED:
            if key not in r:
                error('missing_field', key)
        for key in STRINGS:
            if key in r and (not isinstance(r[key], str) or (key not in ('generated_text',) and not r[key].strip())):
                error('field_type', key + ' must be a non-empty string.')
        if 'question_id' in r and not valid_question_id(r['question_id']):
            error('field_type', 'question_id must be a non-negative integer or non-empty string without slash.')
        for key in ('prompt_token_count', 'generated_token_count'):
            if key in r and not integer(r[key]):
                error('field_type', key + ' must be a non-negative integer (not bool).')
        for key in ('reasoning_token_count', 'think_close_token_index'):
            if key in r and r[key] is not None and not integer(r[key]):
                error('field_type', key + ' must be a non-negative integer or null.')
        if 'enable_thinking' in r and type(r['enable_thinking']) is not bool:
            error('field_type', 'enable_thinking must be bool.')
        if r.get('final_answer_text') is not None and not isinstance(r['final_answer_text'], str):
            error('field_type', 'final_answer_text must be string or null.')
        if r.get('split') not in ('train', 'validation', 'test'):
            error('split', 'split must be train/validation/test.')
        if r.get('finish_reason') not in ('stop', 'length'):
            error('finish_reason', 'finish_reason must be stop/length.')
        for path in forbidden_paths(r):
            error('answer_field', 'Forbidden answer field: ' + path)
        for key in ('model', 'tokenizer'):
            value = r.get(key)
            if isinstance(value, str) and ('@' not in value or not value.rsplit('@', 1)[-1] or '<' in value):
                error('revision', key + ' must include a concrete @revision.')
        config = r.get('generation_config', self.run_config.get('generation_config', self.run_config))
        if not isinstance(config, dict) or not config:
            error('generation_config', 'Non-empty generation_config or --run-config is required.')
            config = {}
        if 'think_tag_in_prompt' in config and type(config['think_tag_in_prompt']) is not bool:
            error('generation_config', 'think_tag_in_prompt must be bool.')
        if 'max_new_tokens' in config and (not integer(config['max_new_tokens']) or config['max_new_tokens'] == 0):
            error('generation_config', 'max_new_tokens must be a positive integer.')
        if 'eos_token_ids' in config and (not token_list(config['eos_token_ids']) or not config['eos_token_ids']):
            error('generation_config', 'eos_token_ids must be a non-empty integer list.')
        qid, run, split = r.get('question_id'), r.get('run_id'), r.get('split')
        if valid_question_id(qid) and isinstance(run, str) and isinstance(split, str):
            identity = (run, split, qid)
            if identity in self.seen:
                error('duplicate_record', 'Duplicate (run_id, split, question_id).')
            self.seen.add(identity)
            if self.references is not None:
                if qid not in self.references:
                    error('unknown_question', 'question_id not found in reference dataset.')
                elif self.references[qid] != split:
                    error('reference_split', 'split differs from reference dataset.')
        if isinstance(run, str):
            signature = {k: r.get(k) for k in ('model', 'tokenizer', 'quantization', 'prompt_version', 'enable_thinking')}
            signature['generation_config'] = config
            if run in self.runs and self.runs[run] != signature:
                error('run_mismatch', 'Model/tokenizer/settings changed inside one run_id.')
            self.runs.setdefault(run, signature)
        ids = {}
        for prefix in ('prompt', 'generated'):
            value = r.get(prefix + '_token_ids')
            if not token_list(value):
                error('token_ids', prefix + '_token_ids must be a list of non-negative integers.')
                continue
            ids[prefix] = value
            if len(value) != r.get(prefix + '_token_count'):
                error('token_count', prefix + '_token_count != len(token_ids).')
            if not value:
                error('empty_tokens', prefix + '_token_ids is empty.')
        if 'generated' in ids:
            generated = ids['generated']
            opens = [i for i, t in enumerate(generated) if t == OPEN]
            closes = [i for i, t in enumerate(generated) if t == CLOSE]
            close = closes[0] if closes else None
            if r.get('think_close_token_index') != close:
                error('think_close_index', 'think_close_token_index does not match original token IDs.')
            if len(opens) > 1 or len(closes) > 1 or (opens and close is not None and opens[0] > close):
                error('think_structure', 'Repeated or out-of-order think tags.')
            start = opens[0] + 1 if opens else 0
            if not opens and r.get('enable_thinking') is True:
                if config.get('think_tag_in_prompt') is not True:
                    error('think_open_missing', 'Missing generated <think>; set think_tag_in_prompt=true when appropriate.')
                if OPEN not in ids.get('prompt', []):
                    error('think_prompt_missing', 'Prompt token IDs do not contain <think>.')
            if config.get('think_tag_in_prompt') is True and (opens or OPEN not in ids.get('prompt', [])):
                error('think_prompt_config', 'think_tag_in_prompt conflicts with token IDs.')
            expected_count = close - start if close is not None else None
            if r.get('reasoning_token_count') != expected_count:
                error('reasoning_count', 'reasoning_token_count does not match think boundaries.')
            if close is None:
                if r.get('final_answer_text') is not None:
                    error('final_answer', 'final_answer_text must be null without </think> (draft contract).')
                if r.get('enable_thinking') is True:
                    self.issue(line, 'unclosed_think', 'No closing think tag; exclude from complete-CoT evaluation.', 'warning')
            text = r.get('generated_text')
            if isinstance(text, str):
                if text.count('<think>') != len(opens) or text.count('</think>') != len(closes):
                    error('think_text', 'Text think tags disagree with token IDs.')
                if close is not None and '</think>' in text and r.get('final_answer_text') != text.split('</think>', 1)[1]:
                    error('final_answer', 'final_answer_text must preserve all text after </think>.')
            limit = config.get('max_new_tokens')
            if integer(limit) and len(generated) > limit:
                error('token_limit', 'Generated length exceeds max_new_tokens.')
            if r.get('finish_reason') == 'length':
                self.issue(line, 'truncated', 'Generation hit a length limit.', 'warning')
                if integer(limit) and len(generated) != limit:
                    error('length_finish', 'finish_reason=length but generated length differs from max_new_tokens.')
            eos = config.get('eos_token_ids')
            if token_list(eos) and eos and r.get('finish_reason') == 'stop' and generated and generated[-1] not in eos:
                error('eos', 'finish_reason=stop but last token is not configured EOS.')
        if self.decode is not None:
            if self.tokenizer_id != r.get('tokenizer'):
                error('tokenizer_identity', 'Selected tokenizer must match record tokenizer@revision.')
            else:
                for prefix, tokens in ids.items():
                    try:
                        if self.decode(tokens) != r.get(prefix + '_text'):
                            error('decode_mismatch', prefix + '_text != decode(original IDs).')
                    except Exception as exc:
                        error('decode_failed', prefix + ': ' + type(exc).__name__)
        timing = r.get('timing')
        if timing is not None:
            if not isinstance(timing, dict):
                error('timing', 'timing must be an object.')
            else:
                for key, value in timing.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        error('timing', key + ' must be finite and non-negative.')
        if r.get('hidden_state_ref') is not None and not isinstance(r['hidden_state_ref'], dict):
            error('hidden_state_ref', 'hidden_state_ref must be an object or null.')

    def report(self):
        counts = Counter(i['severity'] for i in self.issues)
        return dict(status='fail' if counts['error'] else 'pass', records=self.records,
                    errors=counts['error'], warnings=counts['warning'],
                    checks={'tokenizer_decode': 'enabled' if self.decode else 'skipped: no tokenizer supplied',
                            'reference_membership': 'enabled' if self.references is not None else 'skipped: no references supplied',
                            'prompt_leakage': 'forbidden JSON keys only; semantic leakage not verified',
                            'hidden_states': 'reference type only; files/tensors not verified'}, issues=self.issues)

def load_references(paths):
    result = {}
    for path in paths:
        for line, row, problem in read_jsonl(path):
            if problem or not isinstance(row, dict) or not valid_question_id(row.get('question_id')) or row.get('split') not in ('train', 'validation', 'test'):
                raise ValueError(f'Invalid reference record: {path}:{line}')
            qid = row['question_id']
            if qid in result:
                raise ValueError(f'Duplicate reference question_id: {qid}')
            result[qid] = row['split']
    if not result:
        raise ValueError('Reference dataset is empty.')
    return result
