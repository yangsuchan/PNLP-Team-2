"""Deterministic selection of existing checkpoints; never renumber or retokenize."""
from thinkenough.identity import valid_question_id

from collections import OrderedDict
from copy import deepcopy

STRATEGIES = ('all', 'every_n', 'uniform', 'token_interval')
DEFAULT_CONFIG = {'strategy': 'all'}


def normalize_config(config=None):
    config = dict(config or {})
    unknown = set(config) - {'strategy', 'interval', 'max_checkpoints', 'include_initial', 'include_final'}
    if unknown:
        raise ValueError(f'Unknown selection settings: {sorted(unknown)}')
    for key in ('include_initial', 'include_final'):
        if key in config and config.pop(key) is not True:
            raise ValueError(f'{key} must be true: initial and final checkpoints are required.')
    strategy = config.get('strategy', 'all')
    if strategy not in STRATEGIES:
        raise ValueError(f'Unknown selection strategy: {strategy}')
    allowed = {'strategy'} | ({'interval'} if strategy in ('every_n', 'token_interval') else {'max_checkpoints'} if strategy == 'uniform' else set())
    if set(config) - allowed:
        raise ValueError(f'Options do not apply to {strategy}: {sorted(set(config) - allowed)}')
    result = {'strategy': strategy}
    if strategy != 'all':
        key = 'max_checkpoints' if strategy == 'uniform' else 'interval'
        value = config.get(key)
        minimum = 2 if key == 'max_checkpoints' else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}.')
        result[key] = value
    return result


def group_full_checkpoints(records):
    """Reject already sampled/incomplete inputs to prevent silent resampling."""
    groups, ids = OrderedDict(), set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('Every checkpoint must be an object.')
        for key in ('run_id', 'question_id', 'split', 'checkpoint_id', 'checkpoint_index', 'num_checkpoints', 'is_final'):
            if key not in row:
                raise ValueError(f'Missing checkpoint field: {key}')
        if not isinstance(row['run_id'], str) or not row['run_id'] or not valid_question_id(row['question_id']) or row['split'] not in ('train','validation','test'):
            raise ValueError('Invalid run_id/question_id/split.')
        cid = row['checkpoint_id']
        if not isinstance(cid, str) or cid in ids:
            raise ValueError(f'Invalid or duplicate checkpoint_id: {cid}')
        ids.add(cid)
        group = (row['run_id'], row['split'], row['question_id'])
        groups.setdefault(group, []).append(row)
    if not groups:
        raise ValueError('Checkpoint input is empty.')
    for key, rows in groups.items():
        if any(type(c['checkpoint_index']) is not int for c in rows):
            raise ValueError(f'Non-integer checkpoint_index: {key}')
        rows.sort(key=lambda c: c['checkpoint_index'])
        if [c['checkpoint_index'] for c in rows] != list(range(len(rows))):
            raise ValueError(f'Input must contain all contiguous checkpoints, starting at 0: {key}')
        if any(type(c['num_checkpoints']) is not int or c['num_checkpoints'] != len(rows) for c in rows):
            raise ValueError(f'Incomplete checkpoint group: {key}')
        if any(type(c['is_final']) is not bool or c['is_final'] != (i == len(rows)-1) for i,c in enumerate(rows)):
            raise ValueError(f'Invalid final checkpoint flags: {key}')
        if rows[0].get('checkpoint_type') != 'no_reasoning':
            raise ValueError(f'Checkpoint 0 must be no_reasoning: {key}')
        if len({c.get('segmenter') for c in rows}) != 1:
            raise ValueError(f'Mixed segmenters: {key}')
    return groups


def choose(rows, config):
    last = len(rows)-1
    strategy = config['strategy']
    if strategy == 'all':
        return rows
    if strategy == 'every_n':
        indices = set(range(0, len(rows), config['interval']))
    elif strategy == 'uniform':
        count = min(config['max_checkpoints'], len(rows))
        # Integer arithmetic: deterministic evenly spaced original checkpoint indices.
        indices = {i * last // (count-1) for i in range(count)} if count > 1 else {0}
    else:
        counts = []
        for c in rows:
            start, end, count = c.get('reasoning_token_start'), c.get('reasoning_token_end'), c.get('prefix_token_count')
            if (c.get('token_count_source') != 'generated_ids'
                or c.get('alignment_status') not in ('exact','exact_up_to_whitespace')
                or any(type(v) is not int or v < 0 for v in (start,end,count))
                or end-start != count):
                raise ValueError('token_interval requires exact original generated-token alignment for every checkpoint; retokenized/missing positions are not accepted.')
            counts.append(count)
        if counts[0] != 0 or counts != sorted(counts) or len({c['reasoning_token_start'] for c in rows}) != 1:
            raise ValueError('Invalid or non-monotonic original token positions.')
        indices, previous = {0}, 0
        for i, count in enumerate(counts[1:],1):
            if count-previous >= config['interval']:
                indices.add(i)
                previous = count
    indices.update((0,last))
    return [rows[i] for i in sorted(indices)]


def select_checkpoints(records, config=None):
    config = normalize_config(config)
    groups = group_full_checkpoints(records)
    selected, summaries = [], []
    for (run,split,qid), rows in groups.items():
        chosen = choose(rows,config)
        selected.extend(deepcopy(chosen))
        summaries.append({'run_id':run, 'split':split, 'question_id':qid,
                          'original_count':len(rows), 'selected_count':len(chosen),
                          'selected_indices':[c['checkpoint_index'] for c in chosen]})
    before = sum(len(rows) for rows in groups.values())
    return selected, {'selection_version':'v1', 'config':config, 'groups':summaries,
                      'original_checkpoints':before, 'selected_checkpoints':len(selected),
                      'forced_answer_calls_saved':before-len(selected),
                      'note':'Counts assume one forced answer per selected checkpoint; not measured GPU time. Original checkpoint fields are preserved.'}


def verify_selection(source, selected, config):
    expected, report = select_checkpoints(source,config)
    if selected != expected:
        raise ValueError('Selected records differ from the deterministic selection (membership, order, or original field values).')
    return report
