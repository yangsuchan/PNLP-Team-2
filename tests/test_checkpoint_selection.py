"""Selection and CLI integration tests using synthetic checkpoints only."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from thinkenough.data.checkpoint_selection import select_checkpoints, verify_selection, normalize_config
from thinkenough.data.reasoning_checkpoint import build_checkpoints


def make_rows(n=10):
    record = {'question_id':1, 'split':'train', 'run_id':'test', 'generated_text':'<think>'+''.join(f'Sentence {i}. ' for i in range(n))+'</think>yes'}
    rows,_ = build_checkpoints(record)
    assert len(rows) == n+1
    return rows


def indices(rows):
    return [c['checkpoint_index'] for c in rows]


def token_rows():
    rows = make_rows(5)
    for row,count in zip(rows,[0,20,55,75,130,140]):
        row.update(token_count_source='generated_ids',alignment_status='exact',prefix_token_count=count,reasoning_token_start=1,reasoning_token_end=1+count,hidden_state_position=100+count)
    return rows


def test_default_is_unchanged_and_independent():
    rows = make_rows()
    selected,report = select_checkpoints(rows)
    assert selected == rows
    selected[1]['source']['model'] = 'mutation'
    assert rows[1]['source']['model'] is None
    assert report['selected_checkpoints'] == 11


@pytest.mark.parametrize('config,expected', [
    ({'strategy':'every_n','interval':3},[0,3,6,9,10]),
    ({'strategy':'every_n','interval':50},[0,10]),
    ({'strategy':'uniform','max_checkpoints':4},[0,3,6,10]),
    ({'strategy':'uniform','max_checkpoints':2},[0,10]),
    ({'strategy':'uniform','max_checkpoints':20},list(range(11))),
])
def test_selection_keeps_original_fields(config,expected):
    rows = make_rows()
    selected,report = select_checkpoints(rows,config)
    assert indices(selected) == expected
    for row in selected:
        assert row == rows[row['checkpoint_index']]
        assert row['num_checkpoints'] == 11
    assert report['forced_answer_calls_saved'] == 11-len(expected)
    assert verify_selection(rows,selected,config) == report


@pytest.mark.parametrize('config', [
    {'strategy':'all'}, {'strategy':'every_n','interval':3}, {'strategy':'uniform','max_checkpoints':2}
])
def test_empty_reasoning_single_endpoint(config):
    rows = make_rows(0)
    selected,_ = select_checkpoints(rows,config)
    assert indices(selected) == [0]
    assert selected[0]['is_final'] is True


def test_token_interval():
    rows = token_rows()
    selected,_ = select_checkpoints(rows,{'strategy':'token_interval','interval':50})
    assert indices(selected) == [0,2,4,5]
    assert selected[-1] == rows[-1]  # final kept even though fewer than 50 tokens since last


@pytest.mark.parametrize('field,value', [
    ('token_count_source','retokenized'), ('alignment_status','boundary_inside_token'),
    ('prefix_token_count',None), ('reasoning_token_end',999), ('reasoning_token_start',True)
])
def test_token_interval_rejects_unsafe_positions(field,value):
    rows = token_rows()
    rows[2][field] = value
    with pytest.raises(ValueError):
        select_checkpoints(rows,{'strategy':'token_interval','interval':50})


@pytest.mark.parametrize('config', [
    {'strategy':'unknown'}, {'strategy':'every_n'}, {'strategy':'uniform','max_checkpoints':1},
    {'strategy':'every_n','interval':True}, {'strategy':'token_interval','interval':0},
    {'strategy':'all','interval':3}, {'strategy':'uniform','max_checkpoints':3,'interval':2},
    {'strategy':'all','include_final':False}, {'stratgey':'uniform'}
])
def test_bad_config(config):
    with pytest.raises(ValueError):
        normalize_config(config)


def test_reject_partial_duplicate_and_mixed_input():
    rows = make_rows()
    for bad in ([],rows[1:],rows+[rows[0]],rows[:3]+rows[4:]):
        with pytest.raises(ValueError):
            select_checkpoints(bad)
    rows[2]['segmenter'] = 'line/v1'
    with pytest.raises(ValueError):
        select_checkpoints(rows)


def test_group_runs_and_splits():
    a = make_rows(4)
    b = copy.deepcopy(a)
    for c in b:
        c['run_id'] = 'another'
        c['checkpoint_id'] = c['checkpoint_id'].replace('test/','another/')
    selected,report = select_checkpoints(a+b,{'strategy':'uniform','max_checkpoints':2})
    assert len(report['groups']) == 2
    assert len(selected) == 4


def test_verify_detects_changed_field_and_missing_row():
    rows = make_rows()
    config = {'strategy':'every_n','interval':3}
    selected,_ = select_checkpoints(rows,config)
    with pytest.raises(ValueError):
        verify_selection(rows,selected[:-1],config)
    selected[1]['hidden_state_position'] = 999
    with pytest.raises(ValueError):
        verify_selection(rows,selected,config)


def run_cli(*args):
    return subprocess.run([sys.executable,str(ROOT/'scripts/select_checkpoints.py'),*map(str,args)],capture_output=True,text=True,encoding='utf-8')


def test_cli_build_select_verify_and_reject_corruption(tmp_path):
    full,selected,config = tmp_path/'full.jsonl',tmp_path/'selected.jsonl',tmp_path/'config.json'
    built = subprocess.run([sys.executable,str(ROOT/'scripts/build_checkpoints.py'),'--no-tokenizer','--output',str(full)],capture_output=True,text=True)
    assert built.returncode == 0,built.stderr
    config.write_text(json.dumps({'checkpoint_selection':{'strategy':'uniform','max_checkpoints':3}}),encoding='utf-8')
    original = full.read_bytes()
    result = run_cli('--input',full,'--output',selected,'--config',config)
    assert result.returncode == 0,result.stderr
    report = json.loads(result.stdout)
    assert report['original_checkpoints'] > report['selected_checkpoints']
    assert Path(str(selected)+'.selection.json').exists()
    assert full.read_bytes() == original
    verified = run_cli('--input',full,'--verify',selected,'--config',config)
    assert verified.returncode == 0,verified.stderr
    text = selected.read_text(encoding='utf-8')
    selected.write_text(text.replace('"num_checkpoints":','"num_checkpoints":999,"old_count":',1),encoding='utf-8')
    assert run_cli('--input',full,'--verify',selected,'--config',config).returncode == 1


def test_cli_no_output_on_failure_and_no_overwrite(tmp_path):
    full,out = tmp_path/'full.jsonl',tmp_path/'out.jsonl'
    full.write_text(''.join(json.dumps(c)+'\n' for c in make_rows()),encoding='utf-8')
    original = full.read_bytes()
    assert run_cli('--input',full,'--output',full).returncode == 1
    assert full.read_bytes() == original
    out.write_text('KEEP',encoding='utf-8')
    result = run_cli('--input',full,'--output',out,'--strategy','token_interval','--interval',50)
    assert result.returncode == 1
    assert out.read_text() == 'KEEP'


def test_cli_flags_and_config_conflict(tmp_path):
    full,out,cfg = tmp_path/'full.jsonl',tmp_path/'out.jsonl',tmp_path/'config.json'
    full.write_text(''.join(json.dumps(c)+'\n' for c in make_rows()),encoding='utf-8')
    cfg.write_text('{"strategy":"all"}',encoding='utf-8')
    assert run_cli('--input',full,'--output',out,'--config',cfg,'--strategy','all').returncode == 1
    result = run_cli('--input',full,'--output',out,'--strategy','every_n','--interval',3)
    assert result.returncode == 0,result.stderr
    assert indices([json.loads(s) for s in out.read_text().splitlines()]) == [0,3,6,9,10]
