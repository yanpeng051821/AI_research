from copy import deepcopy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from finalize_sft_run import audit_updates, finite_tree
from evaluate_sft_pair_member import validate_member
from analyze_sft_pair import check_pair_inputs, duplicate_fourgram_fraction


def sample_updates():
    core = {'planned_updates': 2, 'effective_batch_records': 4, 'train_records': 5,
            'candidate': {'training': {'micro_batch_size': 2}}}
    rows = [{'event': 'update', 'global_step': 1, 'committed_samples': 4, 'micro_batches': 2,
             'valid_tokens': 7, 'loss': 1.0},
            {'event': 'update', 'global_step': 2, 'committed_samples': 5, 'micro_batches': 1,
             'valid_tokens': 2, 'loss': 0.8}]
    return core, rows


def test_audit_consumes_tail_without_repeating_records():
    core, rows = sample_updates()
    assert len(audit_updates(rows, core, 9)) == 2


@pytest.mark.parametrize('field,value', [('global_step', 1), ('committed_samples', 8),
                                        ('micro_batches', 2), ('loss', float('nan'))])
def test_audit_rejects_broken_tail_or_nonfinite_metrics(field, value):
    core, rows = sample_updates()
    rows[-1][field] = value
    with pytest.raises(ValueError):
        audit_updates(rows, core, 9)


def test_finite_tree_checks_optimizer_tensors():
    assert finite_tree({'m': torch.tensor([1.0]), 'step': torch.tensor(2)})
    assert not finite_tree({'v': [torch.tensor([float('inf')])]} )


def test_s1_source_and_architecture_are_independently_bound():
    config = {'model': {'hidden_size': 8}, 'trained_sequence_length': 768}
    source = {'stage': 'supervised_fine_tuning', 'contract_sha256': 'a',
              'global_step': 2, 'checkpoint_sha256': 'b'}
    manifest = {'source': source, 'model_state_sha256': 'c'}
    audit = {'status': 'passed', 'checkpoint_sha256': 'b', 'model_state_sha256': 'c'}
    freeze = {'contract_sha256': 'a', 'expected_updates': 2}
    validate_member(manifest, config, deepcopy(config), audit, freeze)
    wrong = deepcopy(config)
    wrong['model']['hidden_size'] = 16
    with pytest.raises(ValueError, match='configuration'):
        validate_member(manifest, wrong, config, audit, freeze)
    manifest['source']['stage'] = 'engineering_gate'
    with pytest.raises(ValueError, match='verified final'):
        validate_member(manifest, config, config, audit, freeze)


def test_pair_requires_identical_actual_prompts_not_just_case_ids():
    cases = [{'id': str(i), 'category': 'x', 'tier': 'core'} for i in range(36)]
    rows = [{**case, 'input_ids': [1, 3], 'rendered_prompt': 'fixed',
             'generated_ids': [4, 2], 'generated_tokens': 2, 'stop_reason': 'eos'} for case in cases]
    other = deepcopy(rows)
    check_pair_inputs(cases, rows, other)
    other[0]['input_ids'] = [1, 5]
    with pytest.raises(ValueError, match='prompts'):
        check_pair_inputs(cases, rows, other)


def test_pair_rejects_incomplete_length_limit_and_describes_repetition():
    cases = [{'id': str(i), 'category': 'x', 'tier': 'core'} for i in range(36)]
    rows = [{**case, 'input_ids': [1], 'rendered_prompt': 'fixed',
             'generated_ids': [4], 'generated_tokens': 1, 'stop_reason': 'length_limit'} for case in cases]
    with pytest.raises(ValueError, match='termination'):
        check_pair_inputs(cases, rows, deepcopy(rows))
    assert duplicate_fourgram_fraction([1, 2]) == 0.0
    assert duplicate_fourgram_fraction([1]*6) == pytest.approx(2/3)
