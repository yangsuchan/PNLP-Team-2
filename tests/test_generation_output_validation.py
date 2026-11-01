"""Synthetic contract tests; no model/network required."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from thinkenough.generation.output_validation import GenerationValidator, load_references, strict_json

@pytest.fixture
def record():
    return dict(question_id=1, split='train', run_id='synthetic_contract', model='Qwen/Qwen3-8B@test_revision', tokenizer='Qwen/Qwen3-8B@test_revision', quantization='bf16', prompt_version='sqa3d_official_gpt_v0', enable_thinking=True, prompt_text='prompt', prompt_token_ids=[10], prompt_token_count=1, generated_token_ids=[151667, 20, 151668, 30, 151645], generated_token_count=5, generated_text='<think>reason</think>yes<|im_end|>', reasoning_token_count=1, think_close_token_index=2, final_answer_text='yes<|im_end|>', finish_reason='stop', generation_config={'max_new_tokens':10, 'eos_token_ids':[151645], 'think_tag_in_prompt':False})

def codes(v):
    return {i['code'] for i in v.issues if i['severity'] == 'error'}

def test_valid(record):
    v = GenerationValidator({1:'train'})
    v.validate(record)
    assert v.report()['errors'] == 0
    assert 'skipped' in v.report()['checks']['tokenizer_decode']

@pytest.mark.parametrize('key,value,code', [
    ('question_id', True, 'field_type'), ('question_id', [], 'field_type'),
    ('prompt_token_count', False, 'field_type'), ('generated_token_count', 99, 'token_count'),
    ('generated_token_ids', [True], 'token_ids'), ('generated_token_ids', [-1], 'token_ids'),
    ('generated_token_ids', None, 'token_ids'), ('generated_token_ids', [], 'empty_tokens'),
    ('think_close_token_index', 3, 'think_close_index'), ('reasoning_token_count', 2, 'reasoning_count'),
    ('final_answer_text', 'yes', 'final_answer'), ('generated_text', 'yes', 'think_text'),
    ('model', 'Qwen/Qwen3-8B', 'revision'), ('tokenizer', 'Qwen/Qwen3-8B@<hash>', 'revision'),
    ('split', 'val', 'split'), ('finish_reason', 'oops', 'finish_reason'),
    ('generation_config', [], 'generation_config'), ('enable_thinking', 1, 'field_type'),
    ('hidden_state_ref', [], 'hidden_state_ref'), ('timing', {'decode_ms':float('nan')}, 'timing'),
    ('metadata', {'answers':['yes']}, 'answer_field'), ('reference_answers', ['yes'], 'answer_field'),
    ('generation_config', {'max_new_tokens':3}, 'token_limit'),
    ('generation_config', {'eos_token_ids':[1]}, 'eos'),
])
def test_bad(record, key, value, code):
    record[key] = value
    v = GenerationValidator()
    v.validate(record)
    assert code in codes(v)

def test_missing_and_bad_objects(record):
    del record['final_answer_text']
    v = GenerationValidator()
    v.validate(record)
    v.validate([])
    assert {'missing_field', 'record_type'} <= codes(v)

def test_run_and_duplicates(record):
    v = GenerationValidator()
    v.validate(record)
    v.validate(record)
    other = copy.deepcopy(record)
    other['question_id'] = 2
    other['generation_config']['max_new_tokens'] = 20
    v.validate(other)
    assert {'duplicate_record', 'run_mismatch'} <= codes(v)

def test_reference_checks(record):
    v = GenerationValidator({1:'test'})
    v.validate(record)
    assert 'reference_split' in codes(v)
    v = GenerationValidator({})
    v.validate(record)
    assert 'unknown_question' in codes(v)

def test_prompt_open(record):
    record.update(prompt_token_ids=[10,151667], prompt_token_count=2, generated_token_ids=[20,151668,30,151645], generated_token_count=4, think_close_token_index=1, generated_text='reason</think>yes<|im_end|>')
    record['generation_config']['think_tag_in_prompt'] = True
    v = GenerationValidator()
    v.validate(record)
    assert not codes(v)
    record['generation_config']['think_tag_in_prompt'] = False
    v = GenerationValidator()
    v.validate(record)
    assert 'think_open_missing' in codes(v)

def test_truncated(record):
    record.update(generated_token_ids=[151667,20], generated_token_count=2, generated_text='<think>reason', think_close_token_index=None, reasoning_token_count=None, final_answer_text=None, finish_reason='length')
    record['generation_config']['max_new_tokens'] = 2
    v = GenerationValidator()
    v.validate(record)
    assert not codes(v)
    assert v.report()['warnings'] == 2

def test_decoder(record):
    mapping = {tuple(record['prompt_token_ids']):record['prompt_text'],tuple(record['generated_token_ids']):record['generated_text']}
    v = GenerationValidator(decode=lambda ids:mapping[tuple(ids)], tokenizer_id=record['tokenizer'])
    v.validate(record)
    assert not codes(v)
    record['generated_text'] += ' '
    v.validate(record)
    assert 'decode_mismatch' in codes(v)

def test_decoder_identity_and_failure(record):
    v = GenerationValidator(decode=lambda ids:'', tokenizer_id='wrong')
    v.validate(record)
    assert 'tokenizer_identity' in codes(v)
    def broken(ids):
        raise ValueError('broken')
    v = GenerationValidator(decode=broken, tokenizer_id=record['tokenizer'])
    v.validate(record)
    assert 'decode_failed' in codes(v)

def test_external_config(record):
    config = record.pop('generation_config')
    v = GenerationValidator(run_config={'generation_config':config})
    v.validate(record)
    assert not codes(v)

@pytest.mark.parametrize('text', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', ''])
def test_strict_json(text):
    with pytest.raises(ValueError):
        strict_json(text)

def test_reference_loader(tmp_path):
    p = tmp_path/'ref.jsonl'
    p.write_text('{"question_id":1,"split":"train"}\n', encoding='utf-8')
    assert load_references([p]) == {1:'train'}
    with pytest.raises(ValueError):
        load_references([p,p])

def test_cli(record, tmp_path):
    p, report = tmp_path/'in.jsonl', tmp_path/'report.json'
    p.write_text(json.dumps(record)+'\n{broken\n[]\n', encoding='utf-8')
    cmd = [sys.executable,str(ROOT/'scripts/validate_generation_output.py'),'--input',str(p),'--report',str(report)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 1
    data = json.loads(report.read_text())
    assert data['records'] == 3
    assert {'invalid_json','record_type'} <= {i['code'] for i in data['issues']}
    p.write_text(json.dumps(record)+'\n', encoding='utf-8')
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    p.write_text('', encoding='utf-8')
    assert subprocess.run(cmd, capture_output=True).returncode == 1
    p.unlink()
    assert subprocess.run(cmd, capture_output=True).returncode == 2

def test_cli_warning_strict(record, tmp_path):
    record.update(generated_token_ids=[151667,20], generated_token_count=2, generated_text='<think>reason', think_close_token_index=None, reasoning_token_count=None, final_answer_text=None, finish_reason='length')
    record['generation_config']['max_new_tokens'] = 2
    p = tmp_path/'in.jsonl'
    p.write_text(json.dumps(record)+'\n', encoding='utf-8')
    cmd = [sys.executable,str(ROOT/'scripts/validate_generation_output.py'),'--input',str(p)]
    assert subprocess.run(cmd,capture_output=True).returncode == 0
    result = subprocess.run(cmd+['--strict'],capture_output=True,text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)['status'] == 'fail'

def test_cli_no_overwrite(record, tmp_path):
    p = tmp_path/'in.jsonl'
    original = json.dumps(record)+'\n'
    p.write_text(original, encoding='utf-8')
    result = subprocess.run([sys.executable,str(ROOT/'scripts/validate_generation_output.py'),'--input',str(p),'--report',str(p)],capture_output=True)
    assert result.returncode == 2
    assert p.read_text() == original
