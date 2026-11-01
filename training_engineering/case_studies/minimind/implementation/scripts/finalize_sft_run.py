"""Audit a completed frozen SFT run and export its unchanged native FP32 weights."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics

import torch
from transformers import AutoTokenizer

from export_pretrain_bundle import create_bundle, describe, verify_in_fresh_process
from freeze_sft_training import check
from load_exported_pretrain import read_json, sha256_file, state_digest, write_json
from minimind_lab.models.config import ModelConfig
from minimind_lab.models.model import MinimindForCausalLM
from minimind_lab.training.experiment import canonical_hash


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def audit_updates(rows, core, expected_targets):
    updates = [r for r in rows if r['event'] == 'update']
    require(len(updates) == core['planned_updates'], 'incomplete update coverage')
    previous, targets = 0, 0
    for index, row in enumerate(updates, 1):
        require(row['global_step'] == index, 'update sequence mismatch')
        consumed = min(core['effective_batch_records'], core['train_records'] - previous)
        require(row['committed_samples'] == previous + consumed, 'sample consumption mismatch')
        micro = core['candidate']['training']['micro_batch_size']
        require(row['micro_batches'] == math.ceil(consumed / micro), 'tail micro-batch mismatch')
        require(type(row['valid_tokens']) is int and row['valid_tokens'] > 0, 'invalid target count')
        require(finite_tree(row), 'non-finite update metrics')
        targets += row['valid_tokens']
        previous = row['committed_samples']
    require(previous == core['train_records'] and targets == expected_targets, 'epoch totals mismatch')
    return updates


def audit_checkpoint(value, receipt, contract, expected_targets):
    core = contract['identity']['core']
    step = core['planned_updates']
    require(value['global_step'] == receipt['global_step'] == step, 'final step mismatch')
    require(receipt['contract_sha256'] == contract['contract_sha256'], 'receipt contract mismatch')
    meta = value['metadata']
    require(meta['contract_sha256'] == contract['contract_sha256'] and meta['execution_kind'] == 'formal_sft',
            'not a checkpoint from this formal run')
    require(meta['initial_model_state_sha256'] == core['candidate']['initial_model_state_sha256'],
            'initial B0 mismatch')
    require(meta['valid_tokens_seen'] == expected_targets, 'checkpoint target total mismatch')
    require(meta['validation_records'] == core['validation_records'], 'validation scope mismatch')
    sampler = value['sampler_state']
    require(sampler['committed_position'] == receipt['committed_samples'] == core['train_records'],
            'final data cursor mismatch')
    require(canonical_hash(sampler['indices']) == core['sample_order_sha256'], 'sample order mismatch')
    require(value['scheduler']['last_epoch'] == step, 'scheduler step mismatch')
    states = value['optimizer']['state']
    require(bool(states) and all(int(s['step']) == step for s in states.values()), 'AdamW step mismatch')
    require(finite_tree(value['model']) and finite_tree(states), 'non-finite model/optimizer state')
    require(value['torch_rng_state'].dtype == torch.uint8 and value['cuda_rng_state_all'], 'missing RNG state')
    state = value['model']
    require(all(t.dtype == torch.float32 for t in state.values()), 'model is not native FP32')
    require(torch.equal(state['backbone.embedding.weight'], state['lm_head.weight']), 'conflicting tied weights')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--supervisor-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    args = parser.parse_args()
    require(not args.report_dir.exists() and not args.bundle_dir.exists(), 'use new output directories')
    args.report_dir.mkdir(parents=True)
    marker = args.report_dir / 'run_manifest.json'
    marker.write_text(json.dumps({'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat()}),
                      encoding='utf-8')
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        freeze_path = args.experiment_dir / 'training_recipe/training_freeze.json'
        contract, data = check(freeze_path)
        core = contract['identity']['core']
        run = read_json(args.run_dir / 'run_manifest.json')
        supervisor = read_json(args.supervisor_dir / 'process.json')
        require(run['status'] == run['stop_reason'] == 'completed', 'training has not completed')
        require(supervisor['status'] == 'exited_successfully' and supervisor['exit_code'] == 0, 'launcher not successful')
        require(run['contract_sha256'] == contract['contract_sha256'], 'run contract mismatch')
        expected = data['split_statistics']['train']['total_targets_after_shift']
        require(run['global_step'] == core['planned_updates'] and run['committed_samples'] == core['train_records']
                and run['valid_tokens_seen'] == expected, 'manifest totals mismatch')
        stderr = args.supervisor_dir / 'stderr.log'
        require(stderr.stat().st_size == 0, 'stderr requires review')
        metrics = args.run_dir / 'metrics.jsonl'
        rows = [json.loads(raw) for raw in metrics.read_text(encoding='utf-8').splitlines()]
        updates = audit_updates(rows, core, expected)
        validations = [r for r in rows if 'nll' in r]
        require([r['global_step'] for r in validations] == [0, *range(5000, core['planned_updates'], 5000), core['planned_updates']],
                'validation schedule mismatch')
        require(all(r['valid_tokens'] == 833052 and r['records'] == 2048 and r['precision'] == 'bf16'
                    and finite_tree(r) for r in validations), 'invalid monitoring validation')
        require(validations[-1]['nll'] == run['final_validation']['nll'], 'final validation mismatch')
        saved = [r for r in rows if r['event'] == 'checkpoint_saved']
        require([r['global_step'] for r in saved] == [*range(1000, core['planned_updates'], 1000), core['planned_updates']],
                'checkpoint schedule mismatch')
        for row in saved:
            path = Path(row['path'])
            receipt = read_json(Path(str(path) + '.receipt.json'))
            require(path.is_file() and receipt['global_step'] == row['global_step']
                    and receipt['contract_sha256'] == contract['contract_sha256'], 'checkpoint/receipt inventory mismatch')
        checkpoint_path = Path(run['checkpoint'])
        receipt = read_json(Path(str(checkpoint_path) + '.receipt.json'))
        checkpoint_hash = sha256_file(checkpoint_path)
        require(checkpoint_hash == receipt['checkpoint_sha256'], 'final checkpoint byte hash mismatch')
        value = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        audit_checkpoint(value, receipt, contract, expected)
        model_config = core['model']
        model = MinimindForCausalLM(ModelConfig(**model_config['model'])).float().eval()
        model.load_state_dict(value['model'], strict=True)
        final_digest = state_digest(model)
        adam_states = len(value['optimizer']['state'])
        del value
        start = datetime.fromisoformat(supervisor['started_at_utc'].replace('Z', '+00:00'))
        end = datetime.fromisoformat(supervisor['finished_at_utc'].replace('Z', '+00:00'))
        def weighted_loss(subset):
            return sum(r['loss'] * r['valid_tokens'] for r in subset) / sum(r['valid_tokens'] for r in subset)
        windows = [r['window_seconds'] for r in updates]
        norms = [r['grad_norm'] for r in updates]
        audit = {
            'status': 'passed', 'training_completed': True, 'contract_sha256': contract['contract_sha256'],
            'updates': len(updates), 'records': run['committed_samples'], 'valid_targets': expected,
            'micro_batches': sum(r['micro_batches'] for r in updates), 'last_update': updates[-1],
            'all_update_metrics_finite': True, 'final_parameters_and_optimizer_finite': True,
            'optimizer_state_entries': adam_states, 'scheduler_and_adam_steps_match': True,
            'sampler_order_and_committed_position_match': True,
            'checkpoint_sha256': checkpoint_hash, 'checkpoint_bytes': checkpoint_path.stat().st_size,
            'model_state_sha256': final_digest, 'checkpoint_count': len(saved),
            'checkpoint_inventory': 'All scheduled files/receipts present; only final checkpoint bytes fully hashed/loaded.',
            'checkpoint_total_bytes': sum(Path(r['path']).stat().st_size for r in saved),
            'started_at_utc': supervisor['started_at_utc'], 'finished_at_utc': supervisor['finished_at_utc'],
            'wall_seconds': (end-start).total_seconds(), 'active_run_seconds': run['active_run_seconds'],
            'update_window_seconds_sum': sum(windows), 'checkpoint_seconds_sum': sum(r['seconds'] for r in saved),
            'validation_seconds_sum': sum(r['seconds'] for r in validations),
            'window_median_seconds': statistics.median(windows), 'window_mean_seconds': statistics.mean(windows),
            'grad_norm_median_preclip': statistics.median(norms), 'grad_norm_max_preclip': max(norms),
            'updates_above_clip_threshold': sum(n > 1.0 for n in norms),
            'target_weighted_loss_first1000': weighted_loss(updates[:1000]),
            'target_weighted_loss_last1000': weighted_loss(updates[-1000:]),
            'peak_allocated_bytes': max(r['memory']['peak_allocated_bytes'] for r in updates),
            'reserved_bytes_min_max': [min(r['memory']['reserved_bytes'] for r in updates), max(r['memory']['reserved_bytes'] for r in updates)],
            'post_update_allocated_bytes_min_max': [min(r['memory']['allocated_bytes'] for r in updates), max(r['memory']['allocated_bytes'] for r in updates)],
            'bf16_monitoring_validation': validations, 'stderr_bytes': 0,
            'source_files': {str(p.resolve()): sha256_file(p) for p in (metrics, args.run_dir/'run_manifest.json',
                             args.run_dir/'contract.json', args.supervisor_dir/'process.json', stderr, freeze_path, Path(__file__))},
            'limitations': ['Not an audit of all historical checkpoint tensor states.',
                           'Monitoring BF16 NLL is not final FP32 paired evaluation.',
                           'Finite memory readings do not prove absence of every possible leak.'],
        }
        write_json(args.report_dir / 'training_audit.json', audit)
        print('Final training state and log coverage verified; exporting S1.', flush=True)
        tokenizer_dir = ROOT / 'assets/minimind_tokenizer'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        provenance = {'stage': 'supervised_fine_tuning', 'global_step': run['global_step'],
                      'checkpoint_file': checkpoint_path.name, 'checkpoint_sha256': checkpoint_hash,
                      'checkpoint_bytes': checkpoint_path.stat().st_size,
                      'contract_sha256': contract['contract_sha256'], 'freeze_sha256': sha256_file(freeze_path),
                      'initial_model_state_sha256': core['candidate']['initial_model_state_sha256'],
                      'data_contract_sha256': core['candidate']['data_contract_sha256']}
        bundle = args.bundle_dir.resolve()
        manifest = create_bundle(model, model_config, tokenizer_dir, core['code_sha256'], provenance, bundle)
        # Replace the generic pretraining description before sealing this new SFT bundle.
        (bundle / 'README.md').write_text(
            '# MiniMind mini SFT: native FP32 local inference bundle\n\n'
            'Fine-tuned from our own MiniMind mini pretrained model, not official upstream weights.\n'
            'Native minimind_lab runtime; not a Transformers AutoModel adapter.\n'
            'SFT on complete conversations up to 768 tokens; assistant-only causal supervision.\n'
            'Use the packaged tokenizer chat template with add_generation_prompt=True and '
            'open_thinking=False for the frozen assistant diagnostic. The raw --prompt CLI is '
            'a continuation/loading check, not a chat frontend.\n'
            'FP32 tied weights; no optimizer/recovery states. Local verification and paired '
            'evaluation are recorded in the experiment; public release/model card pending.\n'
            'Data truth, licensing, broader safety, tools and long-context ability are not certified.\n', encoding='utf-8')
        manifest['export_adapter_sha256'] = sha256_file(Path(__file__))
        manifest['files'] = [describe(p, bundle) for p in sorted(bundle.rglob('*'))
                             if p.is_file() and p.name != 'export_manifest.json']
        (bundle / 'export_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        verification = verify_in_fresh_process(bundle, model, tokenizer)
        require(state_digest(model) == final_digest and sha256_file(checkpoint_path) == checkpoint_hash, 'source mutated')
        write_json(args.report_dir / 'export_verification.json', {
            **verification, 'bundle_location': str(bundle), 'stage': 'sft', 'checkpoint_sha256': checkpoint_hash,
            'export_manifest_sha256': sha256_file(bundle/'export_manifest.json'),
            'model_safetensors_sha256': sha256_file(bundle/'model.safetensors'),
            'hf_upload': 'not_performed', 'source_checkpoint_unchanged': True})
        shutil.copyfile(bundle/'export_manifest.json', args.report_dir/'export_manifest.json')
        marker.write_text(json.dumps({'status': 'completed', 'audit': 'passed', 'export': 'verified',
                                     'bundle': str(bundle), 'completed_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2), encoding='utf-8')
        print(json.dumps({'status': 'completed', 'bundle': str(bundle), 'audit_report': str(args.report_dir),
                          'model_state_sha256': final_digest}), flush=True)
    except BaseException as exc:
        import traceback
        marker.write_text(json.dumps({'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()}, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
