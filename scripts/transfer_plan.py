"""Frozen matrix and artifact lineage for the shared transfer baselines."""
import hashlib
import json
from pathlib import Path
from ectt_common import digest, inside, load_contract, read
import compact_grid
import pretransfer_checks

LAUNCH = 'shared-transfer-baselines'
PREFLIGHT = 'shared-transfer-data'
PROBE = 'shared-transfer-probe'
SOURCE = 'shared-transfer-source'
TARGET = 'shared-transfer-target'
REPORT = 'shared-transfer-report'
STAGES = (PREFLIGHT, PROBE, SOURCE, TARGET, REPORT)
GPU_STAGES = (PROBE, SOURCE, TARGET)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def plan(code):
    contract, _ = load_contract(code)
    recipe = read(code/'config/shared_transfer_v1.json')
    files = ('transfer_plan.py', 'transfer_train.py', 'transfer_model.py', 'transfer_worker.py')
    ident = fingerprint({'pretransfer': pretransfer_checks.identity(code),
                         'recipe': digest(code/'config/shared_transfer_v1.json'),
                         'source': {name: digest(code/'scripts'/name) for name in files},
                         'helpers': {name: digest(code/'scripts'/name) for name in
                                     ('causal_protocol.py', 'acquire.py', 'pems08_infer.py')}})
    datasets = contract['datasets']
    source, target = [], []
    for family in recipe['families']:
        for seed in contract['seeds']:
            folds = {}
            for slug, profile in datasets.items():
                excluded = ('bike', 'taxi') if slug in ('bike', 'taxi') else (slug,)
                fold = '+'.join(excluded)
                if fold not in folds:
                    folds[fold] = len(source)
                    source.append({'family': family, 'seed': seed, 'fold': fold,
                                   'excluded': list(excluded), 'index': len(source),
                                   'sources': [name for name in datasets if name not in excluded]})
                target.append({'family': family, 'seed': seed, 'dataset': slug, 'index': len(target),
                               'source_index': folds[fold],
                               'budgets': list(profile['budgets'])})
    probe = [next(row for row in source if row['family'] == family and row['fold'] == 'pems08')
             for family in recipe['families']]
    if len(source) != 42 or len(target) != 48:
        raise ValueError('Unexpected held-out source or target matrix')
    return {'identity': ident, 'recipe': recipe, 'source': source, 'target': target, 'probe': probe}


def store_path(project, matrix):
    return inside(project, project/'outputs/ectt/shared_transfer'/matrix['identity'])


def foundation(project, code, full=False):
    evidence = pretransfer_checks.foundation(project, code, full=full)
    recipe = read(code/'config/shared_transfer_v1.json')
    job = inside(project, project/'results/slurm_jobs'/recipe['completed_recovery_job'])
    status = read(job/'job_status.json')
    result = read(job/'report/transfer_check_result.json')
    if (status.get('stage') != pretransfer_checks.STAGE or status.get('shell_exit_code') != 0
            or status.get('execution_state') != 'COMPLETED'
            or result.get('identity') != pretransfer_checks.identity(code)
            or result.get('qwen_full_size_fixed_support_optimizer_resume_verified') is not True
            or result.get('qwen_recovery_passed') is not True):
        raise ValueError('Required full-size fixed-support Qwen recovery is missing or incompatible')
    collection = inside(project, project/'results/slurm_jobs'/evidence['collector_job']/
                        'report/compact_grid_result.json')
    rows = read(collection)
    return evidence, rows
