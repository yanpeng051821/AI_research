"""Bind assistant-authored check judgments and aggregate an unchanged B0/S1 pair."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics

from aggregate_sft_scores import outcome, validate_review
from load_exported_pretrain import read_json, sha256_file, write_json


def check_pair_inputs(cases, baseline, candidate):
    if not len(cases) == len(baseline) == len(candidate) == 36:
        raise ValueError('incomplete generation coverage')
    if len({r['id'] for r in cases}) != 36:
        raise ValueError('duplicate probe IDs')
    for case, b0, s1 in zip(cases, baseline, candidate):
        if case['id'] != b0['id'] or case['id'] != s1['id']:
            raise ValueError('case order/identity mismatch')
        if b0['input_ids'] != s1['input_ids'] or b0['rendered_prompt'] != s1['rendered_prompt']:
            raise ValueError('paired prompts differ')
        if any(row['category'] != case['category'] or row['tier'] != case['tier'] for row in (b0, s1)):
            raise ValueError('category/tier mismatch')
        for row in (b0, s1):
            tokens = row['generated_ids']
            if not tokens or len(tokens) != row['generated_tokens'] or len(tokens) > 256:
                raise ValueError('invalid generation length')
            expected = 'eos' if tokens[-1] == 2 else 'length_limit'
            if row['stop_reason'] != expected or (expected == 'length_limit' and len(tokens) != 256):
                raise ValueError('termination record mismatch')


def duplicate_fourgram_fraction(ids):
    grams = [tuple(ids[i:i+4]) for i in range(max(0, len(ids)-3))]
    return 0.0 if not grams else (len(grams) - len(set(grams))) / len(grams)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--judgments', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_dir.resolve()
    recipe = root / 'evaluation_recipe'
    b0dir, s1dir = root/'results/b0_v1', root/'results/s1_v1'
    if args.output_dir.exists():
        raise FileExistsError('preserve prior analysis; use a new version')
    inputs = [recipe/'evaluation_freeze.json', recipe/'evaluation_config.json', recipe/'probes.jsonl',
              recipe/'scoring_v2_patch.json', b0dir/'generations.jsonl', b0dir/'validation.json',
              b0dir/'verification.json', b0dir/'scoring_v2/summary.json',
              s1dir/'generations.jsonl', s1dir/'validation.json', s1dir/'result.json',
              s1dir/'run_manifest.json', s1dir/'evaluation_member.json', args.judgments, Path(__file__)]
    identities = {str(p.resolve()): sha256_file(p) for p in inputs}
    freeze = read_json(recipe/'evaluation_freeze.json')
    for path, spec in freeze['files'].items():
        if sha256_file(Path(path)) != spec['sha256']:
            raise ValueError('frozen evaluation changed')
    b0summary = read_json(b0dir/'scoring_v2/summary.json')
    for path, digest in b0summary['inputs_sha256'].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('B0 review source changed')
    b0verify = read_json(b0dir/'verification.json')
    for name in ('generations.jsonl', 'validation.json', 'result.json', 'run_manifest.json'):
        if sha256_file(b0dir/name) != b0verify['files'][name]['sha256']:
            raise ValueError('B0 run source changed')
    result = read_json(s1dir/'result.json')
    member = read_json(s1dir/'evaluation_member.json')
    if (result['status'] != 'evaluation_completed_scoring_pending'
            or read_json(s1dir/'run_manifest.json')['status'] != 'completed'
            or member['status'] != 'completed' or member['scoring_version'] != 'scoring_v2'):
        raise ValueError('S1 evaluation is incomplete')
    for mapping in (read_json(s1dir/'run_manifest.json')['inputs_and_runtime_sha256'], member['inputs_sha256']):
        if any(sha256_file(Path(p)) != h for p, h in mapping.items()):
            raise ValueError('S1 evaluation source changed')
    case_raw = (recipe/'probes.jsonl').read_bytes().splitlines(keepends=True)
    b0_raw = (b0dir/'generations.jsonl').read_bytes().splitlines(keepends=True)
    s1_raw = (s1dir/'generations.jsonl').read_bytes().splitlines(keepends=True)
    cases, b0, s1 = [[json.loads(r) for r in lines] for lines in (case_raw, b0_raw, s1_raw)]
    check_pair_inputs(cases, b0, s1)
    notes = read_json(args.judgments)
    if len(notes) != 36 or len({r['id'] for r in notes}) != 36:
        raise ValueError('incomplete assistant judgment coverage')
    patch = read_json(recipe/'scoring_v2_patch.json')
    if (patch['base_evaluation_freeze_sha256'] != sha256_file(recipe/'evaluation_freeze.json')
            or member['inputs_sha256'][str((recipe/'scoring_v2_patch.json').resolve())] != sha256_file(recipe/'scoring_v2_patch.json')):
        raise ValueError('V2 patch mismatch')
    reviews, decisions = [], []
    baseline_decisions = {r['id']: r['task_outcome'] for r in b0summary['decisions']}
    tiers, categories = {}, {}
    for index, (case, generation, note) in enumerate(zip(cases, s1, notes)):
        if note['id'] != case['id']:
            raise ValueError('assistant judgment order mismatch')
        rubric = deepcopy(case)
        if case['id'] == patch['case_id']:
            rubric['checks'].append(patch['append_check'])
        if len(note['checks']) != len(rubric['checks']):
            raise ValueError(f'missing check judgments for {case["id"]}')
        quotes = note['evidence_quotes']
        if not quotes or any(not q or q not in generation['raw_text'] for q in quotes):
            raise ValueError(f'invalid evidence for {case["id"]}')
        checks = [{'index': j, 'criterion': criterion, 'status': judgment['status'],
                   'reason': judgment['reason'], 'evidence_quotes': quotes}
                  for j, (criterion, judgment) in enumerate(zip(rubric['checks'], note['checks']), 1)]
        label = outcome([c['status'] for c in checks])
        review = {'id': case['id'], 'category': case['category'], 'tier': case['tier'],
                  'case_line_sha256': hashlib.sha256(case_raw[index]).hexdigest(),
                  'generation_line_sha256': hashlib.sha256(s1_raw[index]).hexdigest(),
                  'checks': checks, 'task_outcome': label,
                  'quality_observations': note.get('quality_observations', []),
                  'reviewer': 'assistant_full_output_initial_review', 'independently_reviewed': False,
                  'cases_file_sha256': sha256_file(recipe/'probes.jsonl'),
                  'generations_file_sha256': sha256_file(s1dir/'generations.jsonl'),
                  'evaluation_freeze_sha256': sha256_file(recipe/'evaluation_freeze.json'),
                  'scoring_patch_sha256': sha256_file(recipe/'scoring_v2_patch.json')}
        validate_review(review, rubric, generation, case_raw[index], s1_raw[index])
        reviews.append(review)
        old = baseline_decisions[case['id']]
        pair = ('uncertain' if 'uncertain' in (old, label) else
                'both_pass' if old == label == 'pass' else
                'B0_only_pass' if old == 'pass' else 'S1_only_pass' if label == 'pass' else 'both_nonpass')
        decisions.append({'id': case['id'], 'tier': case['tier'], 'category': case['category'],
                          'B0': old, 'S1': label, 'paired_outcome': pair,
                          'B0_stop': b0[index]['stop_reason'], 'S1_stop': generation['stop_reason']})
        tiers.setdefault(case['tier'], Counter())[label] += 1
        categories.setdefault(case['category'], Counter())[label] += 1
    nll0, nll1 = read_json(b0dir/'validation.json'), read_json(s1dir/'validation.json')
    if any(v['valid_tokens'] != 833052 or v['records'] != 2048 or v['batch_size'] != 4 for v in (nll0, nll1)):
        raise ValueError('NLL scope mismatch')
    def stats(counts):
        n = sum(counts.values())
        return {'records': n, **{s: counts[s] for s in ('pass','fail','uncertain')},
                'pass_rate_lower_bound': counts['pass']/n,
                'pass_rate_upper_bound': (counts['pass']+counts['uncertain'])/n}
    def behavior(rows):
        return {'stop_reasons': dict(Counter(r['stop_reason'] for r in rows)),
                'generated_tokens': sum(r['generated_tokens'] for r in rows),
                'mean_generated_tokens': statistics.mean(r['generated_tokens'] for r in rows),
                'median_generated_tokens': statistics.median(r['generated_tokens'] for r in rows),
                'mean_duplicate_fourgram_fraction': statistics.mean(duplicate_fourgram_fraction(r['generated_ids']) for r in rows),
                'note': 'Repeated token 4-gram fraction is a descriptive diagnostic, not a new pass/fail rule.'}
    summary = {'status': 'paired_assistant_initial_review_complete_not_independently_reviewed',
               'scoring_version': 'scoring_v2', 'by_tier_S1': {k: stats(v) for k,v in tiers.items()},
               'by_category_S1': {k: stats(v) for k,v in categories.items()},
               'paired_by_tier': {tier: dict(Counter(r['paired_outcome'] for r in decisions if r['tier'] == tier))
                                  for tier in tiers},
               'nll': {'B0_fp32': nll0['nll'], 'S1_fp32': nll1['nll'],
                       'absolute_delta': nll1['nll']-nll0['nll'], 'relative_delta': nll1['nll']/nll0['nll']-1,
                       'valid_targets_each': 833052},
               'behavior': {'B0': behavior(b0), 'S1': behavior(s1)},
               'decisions': decisions, 'exact_paired_input_ids_and_prompts': True,
               'independently_reviewed': False, 'user_reviewed': False,
               'semantic_judgments_automatically_verified': False, 'inputs_sha256': identities}
    if any(sha256_file(Path(p)) != h for p,h in identities.items()):
        raise ValueError('analysis inputs changed during aggregation')
    args.output_dir.mkdir(parents=True)
    with (args.output_dir/'reviews.assistant.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for row in reviews:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    write_json(args.output_dir/'summary.json', summary)
    write_json(s1dir/'verification.json', {
        'status': 'generation_nll_and_pair_inputs_verified', 'generation_records': 36,
        'evaluation_freeze_sha256': sha256_file(recipe/'evaluation_freeze.json'),
        'scoring_patch_sha256': sha256_file(recipe/'scoring_v2_patch.json'),
        'exact_input_ids_and_prompt_match': True, 'model_state_unchanged_gate_passed': True,
        'files': {name: {'sha256': sha256_file(s1dir/name), 'bytes': (s1dir/name).stat().st_size}
                  for name in ('generations.jsonl','validation.json','result.json','run_manifest.json','evaluation_member.json')},
        'semantic_judgments_automatically_verified': False,
    })
    print(json.dumps({k:v for k,v in summary.items() if k not in ('decisions','inputs_sha256')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
