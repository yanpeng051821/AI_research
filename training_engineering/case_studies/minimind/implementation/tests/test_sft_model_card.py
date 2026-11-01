import importlib.util
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT.parent / 'experiments/sft_mini_v1'
CARDS = EXPERIMENT / 'release/model_card_v1'
spec = importlib.util.spec_from_file_location('sft_card_chat', CARDS / 'chat.py')
chat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_bilingual_metadata_and_relative_links_are_consistent():
    metadata = []
    for language in ('README.md', 'README_zh.md'):
        text = (CARDS/language).read_text(encoding='utf-8')
        header = yaml.safe_load(text.split('---', 2)[1])
        assert header['base_model'] == 'yanpeng051821/minimind-dense-64m-mini-pretrain'
        assert header['base_model_relation'] == 'finetune'
        assert header['pipeline_tag'] == 'text-generation'
        assert header['language'] == ['zh']
        assert 'license' not in header
        assert 'library_name' not in header
        assert '(chat.py)' in text
        metadata.append(header)
    assert metadata[0] == metadata[1]
    assert '(README_zh.md)' in (CARDS/'README.md').read_text(encoding='utf-8')
    assert '(README.md)' in (CARDS/'README_zh.md').read_text(encoding='utf-8')


def test_cards_quote_actual_metrics_and_model_identities():
    summary = read_json(EXPERIMENT/'results/paired_analysis_v1/summary.json')
    exported = read_json(ROOT/'exports/sft_mini_one_epoch_v1_fp32_v1/export_manifest.json')
    for name in ('README.md', 'README_zh.md'):
        text = (CARDS/name).read_text(encoding='utf-8')
        assert f'{summary["nll"]["B0_fp32"]:.10f}' in text
        assert f'{summary["nll"]["S1_fp32"]:.10f}' in text
        assert f'{exported["parameters"]:,}' in text
        assert exported['model_state_sha256'] in text
        for key in ('checkpoint_sha256', 'contract_sha256', 'initial_model_state_sha256', 'data_contract_sha256'):
            assert exported['source'][key] in text


def test_chat_template_matches_actual_frozen_style_identity_probe():
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'assets/minimind_tokenizer', local_files_only=True)
    first = json.loads((EXPERIMENT/'results/identity_diagnostic_v1/generations.jsonl').read_text(encoding='utf-8').splitlines()[0])
    rendered, ids = chat.encode_prompt(tokenizer, '你是谁？')
    assert rendered == first['rendered_prompt']
    assert ids == first['input_ids']
    assert rendered.endswith('<think>\n\n</think>\n\n')


@pytest.mark.parametrize('ids,budget', [([], 1), ([1], 0), ([1], -1), ([1]*513, 256)])
def test_chat_rejects_invalid_budget_without_silent_truncation(ids, budget):
    with pytest.raises(ValueError):
        chat.check_budget(ids, budget, 768)


def test_chat_accepts_exact_context_boundary():
    chat.check_budget([1]*512, 256, 768)
