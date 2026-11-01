"""Bind a verified final S1 bundle to the unchanged frozen B0 evaluation engine."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from evaluate_sft import run as evaluate
from load_exported_pretrain import read_json, sha256_file, verify_bundle, write_json


def validate_member(s1_manifest, s1_config, b0_config, audit, training_freeze):
    if s1_config != b0_config:
        raise ValueError('S1 architecture/tokenizer configuration differs from B0')
    source = s1_manifest['source']
    if (source.get('stage') != 'supervised_fine_tuning'
            or source.get('contract_sha256') != training_freeze['contract_sha256']
            or source.get('global_step') != training_freeze['expected_updates']
            or source.get('checkpoint_sha256') != audit['checkpoint_sha256']
            or s1_manifest['model_state_sha256'] != audit['model_state_sha256']
            or audit['status'] != 'passed'):
        raise ValueError('S1 is not the verified final state of the frozen training run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    recipe = args.experiment_dir.resolve() / 'evaluation_recipe'
    freeze_path = recipe / 'evaluation_freeze.json'
    freeze = read_json(freeze_path)
    if freeze['status'] != 'frozen':
        raise ValueError('evaluation not frozen')
    for path, spec in freeze['files'].items():
        if sha256_file(Path(path)) != spec['sha256']:
            raise ValueError(f'frozen evaluation input/runtime changed: {path}')
    for package, version in freeze['runtime_versions'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f'evaluation dependency changed: {package}')
    patch_path = recipe / 'scoring_v2_patch.json'
    patch = read_json(patch_path)
    if (patch['status'] != 'user_approved_before_sft' or patch['version'] != 'scoring_v2'
            or patch['base_evaluation_freeze_sha256'] != sha256_file(freeze_path)):
        raise ValueError('V2 patch identity mismatch')
    training_path = args.experiment_dir / 'training_recipe/training_freeze.json'
    training = read_json(training_path)
    if training['files']['scoring_patch']['sha256'] != sha256_file(patch_path):
        raise ValueError('scoring patch differs from training freeze')
    audit_path = args.audit_dir / 'training_audit.json'
    verification_path = args.audit_dir / 'export_verification.json'
    audit, verification = read_json(audit_path), read_json(verification_path)
    bundle = args.bundle.resolve()
    exported = verify_bundle(bundle)
    if (verification['status'] != 'passed' or not verification['parameter_state_exact']
            or verification['export_manifest_sha256'] != sha256_file(bundle / 'export_manifest.json')):
        raise ValueError('export verification does not match the bundle')
    baseline_bundle = Path(__file__).resolve().parents[1] / 'exports/pretrain_mini_one_epoch_v1_fp32_v2'
    validate_member(exported, read_json(bundle/'config.json'), read_json(baseline_bundle/'config.json'), audit, training)
    tracked = [freeze_path, patch_path, training_path, audit_path, verification_path,
               bundle/'export_manifest.json', Path(__file__)]
    identity = {str(p.resolve()): sha256_file(p) for p in tracked}
    evaluate(SimpleNamespace(config=recipe/'evaluation_config.json', cases=recipe/'probes.jsonl',
                             inspection=recipe/'inspection.frozen.json', bundle=bundle,
                             output_dir=args.output_dir.resolve(), device='cuda', run=args.run))
    if any(sha256_file(Path(p)) != digest for p, digest in identity.items()):
        raise ValueError('paired member inputs changed during evaluation')
    write_json(args.output_dir / 'evaluation_member.json', {
        'member': 'S1', 'status': 'completed' if args.run else 'checked_not_evaluated',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'scoring_version': 'scoring_v2',
        'initial_B0_model_state_sha256': freeze['baseline']['model_state_sha256'],
        'S1_model_state_sha256': exported['model_state_sha256'], 'inputs_sha256': identity,
        'evaluation_engine': 'unchanged evaluate_sft.run and generate_case',
        'only_experimental_model_variable': 'final S1 weights with verified source',
        'note': 'The reused engine calls its generic provenance field pretrain_source; its S1 stage is recorded here.',
    })


if __name__ == '__main__':
    main()
