"""CLI for original Qwen3 JSONL validation; no model/GPU required."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from thinkenough.generation.output_validation import GenerationValidator, load_references, read_jsonl, strict_json

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--references', type=Path, nargs='+')
    p.add_argument('--run-config', type=Path)
    p.add_argument('--tokenizer', help='Exact tokenizer ID@revision; optional CPU decode verification')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--report', type=Path)
    p.add_argument('--strict', action='store_true', help='Treat warnings as a failing exit code')
    args = p.parse_args(argv)
    if args.report and args.report.resolve() in {x.resolve() for x in [args.input, *(args.references or []), *([args.run_config] if args.run_config else [])]}:
        p.error('--report must not overwrite an input file')
    try:
        config = strict_json(args.run_config.read_text(encoding='utf-8-sig')) if args.run_config else None
        if config is not None and not isinstance(config, dict):
            raise ValueError('run config must be an object')
        refs = load_references(args.references) if args.references else None
        decode = None
        if args.tokenizer:
            if '@' not in args.tokenizer:
                raise ValueError('--tokenizer requires ID@revision')
            from transformers import AutoTokenizer
            name, revision = args.tokenizer.rsplit('@', 1)
            tok = AutoTokenizer.from_pretrained(name, revision=revision, local_files_only=args.local_files_only, trust_remote_code=False)
            if tok.convert_tokens_to_ids('<think>') != 151667 or tok.convert_tokens_to_ids('</think>') != 151668:
                raise ValueError('Tokenizer does not use expected Qwen3 think token IDs')
            decode = lambda ids: tok.decode(ids, skip_special_tokens=False)
        validator = GenerationValidator(refs, config, decode, args.tokenizer)
        for line, row, problem in read_jsonl(args.input):
            if problem:
                validator.records += 1
                validator.issue(line, 'invalid_json', problem)
            else:
                validator.validate(row, line)
        if validator.records == 0:
            validator.issue(0, 'empty_file', 'Input JSONL is empty.')
        report = validator.report()
        report['strict'] = args.strict
        failed = bool(report['errors'] or (args.strict and report['warnings']))
        report['status'] = 'fail' if failed else 'pass'
        output = json.dumps(report, ensure_ascii=False, indent=2)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(output + '\n', encoding='utf-8')
        print(output)
        return 1 if failed else 0
    except (OSError, ValueError, ImportError) as exc:
        print(f'Validation setup failed: {exc}', file=sys.stderr)
        return 2

if __name__ == '__main__':
    sys.exit(main())
