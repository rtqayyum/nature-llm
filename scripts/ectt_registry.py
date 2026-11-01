"""All 35 experiment families and concrete run dimensions, without fake runs."""
import csv
import json
from pathlib import Path
from ectt_common import read, write, digest, load_contract

DEPENDENCIES = {
    'D01': ['F01'], 'D02': ['D01'], 'D03': ['D02'], 'S01': ['D03'],
    'S02': ['D03'], 'P01': ['D03'], 'P02': ['P01'],
    'B01': ['D03'], 'B02': ['B01', 'P02'], 'B03': ['B02', 'P02'],
    'B04': ['B03', 'S01', 'S02'], 'M01': ['B04'], 'M02': ['M01'],
    'M03': ['M01'], 'M04': ['M03'], 'A01': ['B04'], 'A02': ['M01'],
    'A03': ['M01'], 'A04': ['M01'], 'A05': ['M01'], 'A06': ['M03', 'S01'],
    'C01': ['D03'], 'C02': ['D03'], 'C03': ['D03'], 'G01': ['M01'],
    'G02': ['G01'], 'R01': ['M01'], 'R02': ['R01'], 'R03': ['M01'],
    'R04': ['R01', 'S01'], 'R05': ['M04'], 'O01': ['G01', 'S01'],
    'O02': ['M01', 'S01'], 'O03': ['M01']}


def source_domains(contract, target, mode_out=False):
    groups = contract['folds']['mode_out_secondary']
    excluded = {target}
    if target in ('bike', 'taxi'):
        excluded |= {'bike', 'taxi'}
    if mode_out:
        excluded |= set(next(group for group in groups.values() if target in group))
    return sorted(set(contract['datasets']) - excluded)


def build(code):
    code = Path(code)
    c, e = load_contract(code)
    record = read(code / e['master_record'])
    if digest(code / record['master_source']) != record['master_sha256']:
        raise ValueError('Master document source/record mismatch')
    executor = {f: s for s, row in e['stages'].items() for f in row['families']}
    families = []
    for row in record['experiments']:
        f = row['id']
        families.append({**row, 'depends_on': DEPENDENCIES.get(f, []),
            'implementation': 'HISTORICAL_EVIDENCE' if f == 'F01' else 'IMPLEMENTED' if f in executor else
                'INDEPENDENT_REVIEW_REQUIRED' if f == 'S01' else 'PLANNED_NOT_IMPLEMENTED',
            'stage': executor.get(f),
            'readiness': 'HISTORICAL_REPORT_TO_VERIFY' if f == 'F01' else 'WAITING_FOR_PREREQUISITES',
            'scope': 'conditional/optional families require their documented prerequisites; no invented missing metadata'})
    if len(families) != 35 or len({f['id'] for f in families}) != 35:
        raise ValueError('Expected the complete 35-family master registry')
    known = {f['id'] for f in families}
    for f in families:
        if not set(f['depends_on']) <= known:
            raise ValueError('Unknown family prerequisite')
    # One source fit per unique source set/objective/seed, shared across target budgets.
    source_fits, target_runs = {}, []
    for target, d in c['datasets'].items():
        sources = source_domains(c, target)
        fold = '-'.join(sources)
        for seed in c['seeds']:
            for objective in ('ordinary', 'verification'):
                sid = 'qwen-' + objective + '-s%d-src-' % seed + fold
                source_fits[sid] = {'run_id': sid, 'sources': sources, 'seed': seed,
                    'backbone': c['model']['backbone'], 'objective': objective,
                    'state': 'PLANNED_NOT_IMPLEMENTED', 'gpus': 1, 'cpus': 5,
                    'datasets_excluded_from_all_source_learning': sorted(set(c['datasets']) - set(sources))}
                for budget, b in d['budgets'].items():
                    rid = '%s-%s-s%d-q%s' % (target, objective, seed, budget)
                    target_runs.append({'run_id': rid, 'dataset': target, 'seed': seed,
                        'budget': budget, 'source_fit': sid if b['eligible_for_learned_adaptation'] else None,
                        'objective': objective, 'test_targets': d['test_targets'],
                        'support_targets': b['support_target_rows'], 'verification_targets': b['verification_target_rows'],
                        'state': 'PLANNED_NOT_IMPLEMENTED' if b['eligible_for_learned_adaptation'] else 'PERSISTENCE_FALLBACK_NO_EXTRA_LABELS',
                        'gpus': 1 if b['eligible_for_learned_adaptation'] else 0, 'cpus': 5,
                        'role': 'development' if target in c['development_targets'] else 'confirmatory',
                        'policies': ['local', 'unconditional', 'identical_cs', 'fixed_prefix', 'learned_gate'],
                        'prediction_reuse': 'Same saved L/T candidates across policy-only comparisons'})
    dimensions = {
        'core': {'datasets': list(c['datasets']), 'seeds': c['seeds'], 'budgets': c['partitions']['sparse_target_budgets']},
        'objective_controls': ['ordinary', 'verification', 'mean_only', 'envelope_only'],
        'backbone_controls': ['pretrained_qwen', 'random_qwen_same_trainability', 'compact_nonlanguage'],
        'external': c['external_comparators'], 'outages': c['secondary']['outages'],
        'statistics': c['statistics'], 'success_criteria': c['success_criteria'],
        'optional_seeds': c['secondary']['five_seed_extension'],
        'optional_rule': c['secondary']['extension_rule']}
    return {'schema': 'ECTT_EXPERIMENT_REGISTRY_V1', 'families': families, 'dimensions': dimensions,
            'source_fit_plan': list(source_fits.values()), 'target_run_plan': target_runs,
            'max_concurrent_jobs': 1, 'automatic_chaining': False,
            'important': 'This registry is a design, not completed execution. Only declared implemented stages can be submitted.'}


def save(code, directory):
    directory = Path(directory)
    matrix = build(code)
    write(directory / 'experiment_registry.json', matrix)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'experiment_families.csv').open('w', newline='') as stream:
        fields = ['id', 'status', 'description', 'implementation', 'stage', 'depends_on']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: ';'.join(v) if isinstance(v, list) else v for k, v in row.items() if k in fields}
                        for row in matrix['families'])
    return matrix
