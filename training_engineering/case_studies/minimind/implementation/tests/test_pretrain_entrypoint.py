import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/pretrain_smoke'


def run_cli(output, *extra):
    return subprocess.run(
        [sys.executable, str(ROOT/'scripts/train_pretrain.py'),
         '--train-data', str(FIXTURES/'train.jsonl'),
         '--validation-data', str(FIXTURES/'validation.jsonl'),
         '--output-dir', str(output), *map(str, extra)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )


def test_cli_success_completed_resume_and_overwrite_guard(tmp_path):
    first, resumed = tmp_path/'first', tmp_path/'resumed'
    result = run_cli(first)
    assert result.returncode == 0, result.stderr
    report = json.loads((first/'result.json').read_text(encoding='utf-8'))
    assert report['global_step'] == report['updates_this_run'] == 2
    assert report['committed_samples'] == 5
    assert report['validation']['valid_tokens'] > 0
    assert not (first/'error.txt').exists()
    saved_bytes = (first/'checkpoint.pt').read_bytes()
    result = run_cli(resumed, '--resume', first/'checkpoint.pt')
    assert result.returncode == 0, result.stderr
    restored = json.loads((resumed/'result.json').read_text(encoding='utf-8'))
    assert restored['updates_this_run'] == 0
    assert restored['global_step'] == 2
    assert restored['validation'] == report['validation']
    a = torch.load(first/'checkpoint.pt', map_location='cpu', weights_only=True)
    b = torch.load(resumed/'checkpoint.pt', map_location='cpu', weights_only=True)
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
    result = run_cli(first)
    assert result.returncode != 0
    assert 'new or empty output directory' in result.stderr
    assert (first/'checkpoint.pt').read_bytes() == saved_bytes
    assert json.loads((first/'result.json').read_text(encoding='utf-8')) == report


def test_cli_records_missing_input_and_exits_nonzero(tmp_path):
    output = tmp_path/'failed'
    result = run_cli(output, '--train-data', tmp_path/'missing.jsonl')
    assert result.returncode != 0
    error = (output/'error.txt').read_text(encoding='utf-8')
    assert 'Traceback' in error and 'missing.jsonl' in error
    assert not (output/'result.json').exists()
    assert not (output/'checkpoint.pt').exists()


def test_validation_failure_keeps_completed_checkpoint(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('pretrain_entry_test', ROOT/'scripts/train_pretrain.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    output = tmp_path/'validation-failed'
    monkeypatch.setattr(sys, 'argv', ['train_pretrain.py',
        '--train-data', str(FIXTURES/'train.jsonl'),
        '--validation-data', str(FIXTURES/'validation.jsonl'),
        '--output-dir', str(output)])

    def fail_validation(*args):
        raise RuntimeError('simulated validation failure')

    monkeypatch.setattr(entry, 'evaluate_nll', fail_validation)
    with pytest.raises(RuntimeError, match='simulated validation failure'):
        entry.main()
    saved = torch.load(output/'checkpoint.pt', map_location='cpu', weights_only=True)
    assert saved['global_step'] == 2
    assert saved['sampler_state']['committed_position'] == 5
    assert 'simulated validation failure' in (output/'error.txt').read_text(encoding='utf-8')
    assert not (output/'result.json').exists()
