"""Select from a complete checkpoint JSONL, or verify a saved selection. No GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from thinkenough.data.checkpoint_selection import STRATEGIES, select_checkpoints, verify_selection, normalize_config


def load_json(text):
    def pairs(items):
        result = {}
        for k,v in items:
            if k in result:
                raise ValueError(f'Duplicate JSON key: {k}')
            result[k] = v
        return result
    def invalid(value):
        raise ValueError(f'Invalid JSON constant: {value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def read_rows(path):
    rows = []
    with path.open(encoding='utf-8-sig') as f:
        for line,text in enumerate(f,1):
            try:
                rows.append(load_json(text))
            except ValueError as exc:
                raise ValueError(f'{path}:{line}: {exc}') from exc
    return rows


def atomic_write(path, content):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='\n',dir=path.parent,delete=False) as f:
            tmp = f.name
            f.write(content)
        os.replace(tmp,path)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path,help='Complete original checkpoint JSONL')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output',type=Path,help='New selected JSONL; never use the input path')
    mode.add_argument('--verify',type=Path,help='Existing selected JSONL to verify; does not rewrite it')
    p.add_argument('--config',type=Path,help='Selection JSON object or {checkpoint_selection: {...}}')
    p.add_argument('--strategy',choices=STRATEGIES)
    p.add_argument('--interval',type=int)
    p.add_argument('--max-checkpoints',type=int)
    p.add_argument('--report',type=Path,help='Default: OUTPUT.selection.json (create mode only)')
    args = p.parse_args(argv)
    try:
        if args.config and any(v is not None for v in (args.strategy,args.interval,args.max_checkpoints)):
            raise ValueError('Use --config OR strategy flags, not both.')
        if args.config:
            raw = load_json(args.config.read_text(encoding='utf-8-sig'))
            if not isinstance(raw,dict):
                raise ValueError('Config must be a JSON object.')
            if 'checkpoint_selection' in raw:
                if set(raw) != {'checkpoint_selection'}:
                    raise ValueError('Selection config wrapper accepts only checkpoint_selection; set segmenter during build_checkpoints.')
                raw = raw['checkpoint_selection']
            if not isinstance(raw,dict):
                raise ValueError('checkpoint_selection must be an object.')
            config = normalize_config(raw)
        else:
            config = {'strategy':args.strategy or 'all'}
            if args.interval is not None:
                config['interval'] = args.interval
            if args.max_checkpoints is not None:
                config['max_checkpoints'] = args.max_checkpoints
            config = normalize_config(config)
        report_path = args.report or (Path(str(args.output)+'.selection.json') if args.output else None)
        inputs = [args.input] + ([args.config] if args.config else []) + ([args.verify] if args.verify else [])
        outputs = [v for v in (args.output,report_path) if v is not None]
        paths = [v.resolve() for v in inputs+outputs]
        if len(paths) != len(set(paths)):
            raise ValueError('Input, output, config, verify, and report paths must be distinct.')
        # Read and validate everything before writing any output.
        source = read_rows(args.input)
        if args.verify:
            report = verify_selection(source,read_rows(args.verify),config)
        else:
            selected,report = select_checkpoints(source,config)
        report.update(status='pass',mode='verify' if args.verify else 'select',
                      source=str(args.input),source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
        if args.output:
            text = ''.join(json.dumps(c,ensure_ascii=False,allow_nan=False)+'\n' for c in selected)
            report['output_sha256'] = hashlib.sha256(text.encode('utf-8')).hexdigest()
            atomic_write(args.output,text)
        output = json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        if report_path:
            atomic_write(report_path,output)
        print(output,end='')
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'Checkpoint selection failed: {exc}',file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
