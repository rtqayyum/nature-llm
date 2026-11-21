"""Package independent review inputs and evaluate the approved held-out inventory."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys
import tarfile
import tempfile

from ectt_common import completed, digest, finish, inside, load_contract, read, write
import heldout_candidates as candidates
from transfer_plan import fingerprint
from verification_pilot import job_report
from verification_pilot_score import candidate_arrays

STAGE = 'shared-transfer-ectt-heldout-score'
RULES = (*candidates.windows.RULES, candidates.TWO_THIRDS)
COLLECTION = '3100396'
PREFLIGHT = '3100394'
EXTRA = ('scripts/heldout_score.py', 'scripts/heldout_thresholds.py', 'tests/test_heldout_thresholds.py',
         'research/HELDOUT_S01_INTERNAL_REVIEW.md', 'scripts/transfer_parallel.py', 'scripts/slurm_job.py',
         'scripts/slurm_submit.py', 'scripts/ectt_env.py', 'config/ectt_gpu_requirements.txt',
         'tests/test_heldout_score.py', 'research/HELDOUT_S01_HANDOFF.md')


def review_inputs(project, code):
    plan = candidates.matrix(code)
    identity = plan['identity']
    collector = job_report(project, COLLECTION, candidates.REPORT)
    packet = job_report(project, PREFLIGHT, candidates.PROBE)/'S01_packet'
    result = read(collector/'transfer_result.json')
    prior = read(packet/'manifest.json')
    inventory_file = collector/'candidate_inventory.json'
    inventory = read(inventory_file)
    if (prior['identity'] != identity or prior['tasks'] != plan['tasks'] or prior['files'] != plan['reviewed_hashes']
            or prior['rules'] != list(RULES) or result['identity'] != identity
            or result['scientific_status'] != 'HELDOUT_CANDIDATES_READY' or result['failed'] != []
            or result['diagnostic_cases_checked'] != 54 or inventory['identity'] != identity
            or result['verification_labels_scored'] != 0 or result['test_labels_scored'] != 0
            or inventory['failures'] != []):
        raise ValueError('Expected complete unscored held-out collection 3100396')
    root = project/'outputs/ectt/heldout_gpt2'/identity/'groups'
    expected = {str(root/str(group['index'])/d): {**group, 'dataset': d}
                for group in plan['tasks'] for d in group['targets']}
    records = inventory['records']
    if len(records) != 54 or {r['case']: r['task'] for r in records} != expected:
        raise ValueError('Candidate inventory differs from the six-domain matrix')
    hashes = {**plan['reviewed_hashes'], **{name: digest(code/name) for name in EXTRA}}
    manifest = {'collection_job': COLLECTION, 'preflight_job': PREFLIGHT, 'candidate_identity': identity,
                'scorer_snapshot': code.name,
                'inventory_sha256': digest(inventory_file), 'preflight_manifest_sha256': digest(packet/'manifest.json'),
                'reviewed_hashes': hashes}
    manifest['review_identity'] = fingerprint(manifest)
    return manifest, records, inventory_file, packet/'manifest.json'


def approval_path(project, manifest):
    return project/'work/ectt_s01_reviews'/(manifest['review_identity']+'.json')


def build_packet(project, code):
    manifest, _, inventory, prior = review_inputs(project, code)
    work = project/'work'
    out = work/('S01_heldout_'+COLLECTION+'_'+manifest['review_identity'][:12]+'.tar.gz')
    with tempfile.TemporaryDirectory(dir=work) as temp:
        temp = Path(temp)
        write(temp/'review_manifest.json', manifest)
        write(temp/'disposition_template.json', {**manifest, 'reviewer': '', 'date': '', 'disposition': '',
              'findings': [], 'status': 'PENDING', 'independent_review_complete': False})
        archive = temp/'packet.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for name, sha in manifest['reviewed_hashes'].items():
                if digest(code/name) != sha:
                    raise ValueError('Code changed during review packaging: '+name)
                tar.add(code/name, arcname='S01/'+name, recursive=False)
            for name, path in [('candidate_inventory.json', inventory), ('preflight_manifest.json', prior),
                               ('review_manifest.json', temp/'review_manifest.json'),
                               ('disposition_template.json', temp/'disposition_template.json')]:
                tar.add(path, arcname='S01/'+name, recursive=False)
        archive.replace(out)
    print('Review archive', out)
    print('Reviewer disposition belongs at', approval_path(project, manifest))
    print('After approval, run the CPU scoring stage', STAGE, 'from snapshot', code.name)
    return out


def admission(project, manifest):
    location = approval_path(project, manifest)
    if not location.is_file():
        raise ValueError('Independent S01 review is pending. Expected signed-off disposition at '+str(location))
    payload = location.read_bytes()
    record = json.loads(payload)
    if (record.get('status') != 'APPROVED' or record.get('independent_review_complete') is not True
            or any(record.get(k) != v for k, v in manifest.items())
            or any(not isinstance(record.get(k), str) or not record[k].strip()
                   for k in ('reviewer', 'date', 'disposition'))):
        raise ValueError('Reviewer disposition does not approve this inventory and scoring code')
    return hashlib.sha256(payload).hexdigest()


def checked_case(record, identity, contract):
    folder = Path(record['case'])
    marker = folder/'COMPLETE.json'
    if digest(marker) != record['manifest_sha256']:
        raise ValueError('Frozen completion record changed: '+str(folder))
    saved = read(marker)
    if saved['identity'] != identity or saved['state'] != 'COMPLETED':
        raise ValueError('Unexpected frozen candidate identity')
    for name in ('frozen_candidates.json', 'verification_candidates.npz', 'test_candidates.npz'):
        if digest(folder/name) != saved['artifacts'][name]:
            raise ValueError('Frozen forecast input changed: '+str(folder/name))
    meta = read(folder/'frozen_candidates.json')
    task = record['task']
    profile = contract['datasets'][task['dataset']]
    ledger = profile['budgets']['0.05']
    if (meta['task'] != task or task['dataset'] in task['sources'] or meta['budget'] != '0.05'
            or meta['target_container_sha256'] != profile['container_sha256'] or meta['support_raw'] != ledger['support_raw']
            or meta['verification_targets'] != ledger['verification_target_rows'] or meta['test_targets'] != profile['test_targets']
            or not meta['support_raw'][1] <= meta['verification_targets'][0] < meta['verification_targets'][1] <= meta['test_targets'][0]
            or meta['unique_support_values'] != (meta['support_raw'][1]-meta['support_raw'][0])*profile['shape'][0]
            or not math.isfinite(meta['support_scale']) or meta['support_scale'] < 1):
        raise ValueError('Candidate bounds, scale or support label ledger differ')
    return meta


def decisions_for(meta, prefix, raw, contract):
    import numpy as np
    from heldout_thresholds import replay

    start, end = meta['verification_targets']
    origins = np.arange(start, end)
    settings = dict(rho=contract['verifier']['rho'], delta=contract['verifier']['delta_per_stream'],
                    eta=contract['training_design']['mechanism_coefficients']['epsilon'],
                    minimum=contract['partitions']['minimum_verification_origins'])
    decisions = {}
    for rule in RULES:
        next_row = start
        limit = candidates.two_thirds_rows(len(origins)) if rule == candidates.TWO_THIRDS else len(origins)

        def matured(origin):
            nonlocal next_row
            if origin != next_row or next_row >= start+limit:
                raise ValueError('Unexpected verification label requested')
            next_row += 1
            truth = np.asarray(raw[:, origin], dtype=np.float64)
            if truth.shape != prefix['local'].shape[1:] or not np.isfinite(truth).all() or np.any(truth < 0):
                raise ValueError('Invalid verification label')
            return truth

        decision = replay(rule, origins, prefix['local'], prefix['transfer'],
                          meta['support_scale'], matured, settings, limit)
        n = next_row-start
        if (n != decision['verification_origins_read'] or len(decision['trace']) != n
                or (not decision['accepted'] and n != limit)):
            raise ValueError('Invalid label ledger')
        decisions[rule] = decision
    return decisions


def test_rows(meta, forecasts, truth, decisions, contract):
    import numpy as np
    from shared_baselines import independent_scores

    if truth.shape != forecasts['local'].shape or not np.isfinite(truth).all() or np.any(truth < 0):
        raise ValueError('Invalid test cohort')
    metrics = {k: independent_scores(v, truth) for k, v in forecasts.items()}
    errors = {k: np.abs(v-truth).mean(axis=1) for k, v in forecasts.items()}
    available = meta['verification_targets'][1]-meta['verification_targets'][0]
    threshold = contract['training_design']['mechanism_coefficients']['epsilon']
    rows, columns = [], {'local': errors['local']}
    for rule in RULES:
        decision = decisions[rule]
        accept = decision['accepted']
        chosen = 'transfer' if accept else 'local'
        deployed = metrics[chosen]
        excess = (errors[chosen]-errors['local'])/meta['support_scale']
        positive = np.maximum(excess, 0.)
        n = decision['verification_origins_read']
        restricted = n if accept else available+1
        spent = n*truth.shape[1]
        delta = deployed['MAE']-metrics['local']['MAE']
        row = {k: meta['task'][k] for k in ('fold', 'dataset', 'family', 'seed', 'arm')}
        row.update(rule=rule, selected=chosen, accepted=accept, useful=accept and delta<0,
                   harmful=accept and delta>0, material_harm_fraction=float(np.mean(excess>threshold)),
                   materially_harmful_acceptance=accept and delta/meta['support_scale']>threshold,
                   positive_excess_mean=float(positive.mean()), positive_excess_p90=float(np.quantile(positive, .9)),
                   verification_rows=n, available_rows=available, verification_fraction=n/available,
                   restricted_first_acceptance=restricted, restricted_fraction=restricted/(available+1),
                   censored=not accept, support_values=meta['unique_support_values'], verification_values=spent,
                   total_target_values=spent+meta['unique_support_values'], test_origins=len(truth))
        for name, scores in [('local', metrics['local']), ('transfer', metrics['transfer']), ('deployed', deployed)]:
            row.update({name+'_'+key: value for key, value in scores.items()})
        rows.append(row)
        columns[rule] = errors[chosen]
    return rows, columns


def summarize(rows, errors, contract):
    import numpy as np
    from transfer_parallel import block_means

    domains = contract['confirmatory_targets']
    seeds = contract['seeds']
    arms = ('suffix_mae', 'prefix_mae', 'verification_margin')
    keys = [(arm, rule) for arm in arms for rule in RULES]
    lookup = {(r['dataset'], r['seed'], r['arm'], r['rule']): r for r in rows}
    if len(rows) != len(lookup) or len(rows) != len(domains)*len(seeds)*len(keys):
        raise ValueError('Incomplete policy comparisons')
    groups, panels = {}, {}
    for domain in domains:
        local = np.mean([errors[domain,s,arms[0]]['local'] for s in seeds], axis=0)
        columns = [local]
        groups[domain] = {}
        for arm, rule in keys:
            selected = [lookup[domain, s, arm, rule] for s in seeds]
            if any(not np.array_equal(errors[domain,s,arm]['local'], errors[domain,s,arms[0]]['local']) for s in seeds):
                raise ValueError('Local test comparator differs within a matched seed')
            columns.append(np.mean([errors[domain,s,arm][rule] for s in seeds], axis=0))
            group = {k: mean(r[k] for r in selected) for k in ('local_MAE', 'deployed_MAE', 'transfer_MAE',
                     'verification_rows', 'verification_fraction', 'restricted_fraction', 'total_target_values',
                     'material_harm_fraction', 'positive_excess_mean')}
            group.update({k: sum(int(r[k]) for r in selected) for k in
                          ('accepted', 'useful', 'harmful', 'materially_harmful_acceptance')})
            groups[domain][arm+'/'+rule] = group
        panels[domain] = np.column_stack(columns)
    stats = contract['statistics']
    blocks = {}
    for length in (stats['block_steps'], *stats['sensitivity_blocks']):
        sampled = []
        for idx, domain in enumerate(domains):
            # Use identical circular block starts for the paired NYC rows.
            fold = domains.index('bike') if domain in ('bike', 'taxi') else idx
            rng = np.random.default_rng([20261111, length, fold])
            draws = block_means(panels[domain], length, stats['paired_bootstrap_replicates'], rng)
            sampled.append(np.divide(draws[:,1:], draws[:,:1], out=np.full_like(draws[:,1:], np.nan), where=draws[:,:1]>0))
        blocks[length] = np.mean(sampled, axis=0)
    macro, contrasts = {}, {}
    for arm, rule in keys:
        label = arm+'/'+rule
        group = [groups[d][label] for d in domains]
        macro[label] = {k: mean(g[k] for g in group) for k in ('verification_fraction', 'restricted_fraction',
                         'material_harm_fraction', 'verification_rows')}
        macro[label].update(relative_MAE=mean(g['deployed_MAE']/g['local_MAE'] for g in group)
                            if all(g['local_MAE']>0 for g in group) else None,
                            **{k: sum(g[k] for g in group) for k in ('accepted', 'useful', 'harmful', 'materially_harmful_acceptance')})
    pairs = [((arms[-1], r), (a, r)) for r in RULES for a in arms[:-1]]
    pairs += [((a,candidates.TWO_THIRDS), (a,'fixed_budget_mean')) for a in arms]
    for treatment, control in pairs:
        a,b = ('/'.join(x) for x in (treatment, control))
        left,right = macro[a],macro[b]
        item = {k: left[k]-right[k] for k in ('verification_fraction', 'restricted_fraction', 'material_harm_fraction',
                                               'accepted', 'useful', 'harmful', 'materially_harmful_acceptance')}
        item['relative_MAE'] = None if left['relative_MAE'] is None or right['relative_MAE'] is None else left['relative_MAE']-right['relative_MAE']
        item['relative_MAE_intervals'] = {}
        for length, sample in blocks.items():
            effect = sample[:,keys.index(treatment)]-sample[:,keys.index(control)]
            item['relative_MAE_intervals'][str(length)] = np.quantile(effect,[.025,.975]).tolist() if np.isfinite(effect).all() else None
        contrasts[a+' vs '+b] = item
    return {'domains': groups, 'macro': macro, 'contrasts': contrasts,
            'limited_block_counts': {str(length): [d for d in domains if len(panels[d])<8*length] for length in blocks},
            'nyc_shared_source_fold': ['bike', 'taxi'], 'intervals': 'Descriptive 95% paired circular row-block intervals.'}


def write_summary(path, comparison, case_count):
    lines = ['# Held-out anchored GPT-2 scoring', '',
             f'{case_count} frozen candidates and {case_count*len(RULES)} decisions across six domains.',
             'Primary comparison uses the locked two-sided verifier and the prefix-MAE control.', '',
             '| Objective and rule | Macro deployed / local MAE | Mean verification rows | Accepted | Useful | Harmful |',
             '|---|---:|---:|---:|---:|---:|']
    for key, r in comparison['macro'].items():
        relative = 'undefined' if r['relative_MAE'] is None else format(r['relative_MAE'], '.4f')
        lines.append(f"| {key} | {relative} | {r['verification_rows']:.1f} | {r['accepted']} | {r['useful']} | {r['harmful']} |")
    name = 'verification_margin/locked_two_sided vs prefix_mae/locked_two_sided'
    effect = comparison['contrasts'][name]
    lines += ['', 'Margin minus prefix macro relative MAE under the primary rule = '+str(effect['relative_MAE']),
              'Descriptive 95% paired interval at 144 rows per block = '+str(effect['relative_MAE_intervals']['144']),
              'Per-case accuracy, harm and label counts are in metrics.csv. Domain summaries and paired differences are in comparison.json.',
              'Verification rows count labels consulted by the rule. Later observations can still enter forecast inputs.',
              'Bike and taxi share a source fold. Domain counts and intervals refer to this ordered-row benchmark.']
    path.write_text('\n'.join(lines)+'\n')


def evaluate(project, code, store, records, manifest, approved, contract):
    import numpy as np
    from transfer_worker import raw_arrays, raw_paths

    # Validate the archived forecast inputs before opening the first label stream.
    cases = []
    for record in records:
        meta = checked_case(record, manifest['candidate_identity'], contract)
        cases.append((meta, Path(record['case']), record['manifest_sha256']))
    files = raw_paths(project, code, contract)
    raw = raw_arrays(files, contract['confirmatory_targets'], contract)
    decisions = {}
    for meta, folder, key in cases:
        slug = meta['task']['dataset']
        prefix = candidate_arrays(folder/'verification_candidates.npz', meta['verification_targets'], raw[slug].shape[0])
        decisions[key] = decisions_for(meta, prefix, raw[slug], contract)
    path = store/'decisions.json'
    payload = {'review_identity': manifest['review_identity'], 'approval_sha256': approved, 'decisions': decisions}
    if path.exists():
        if read(path) != payload:
            raise ValueError('Prior verification decisions differ on retry')
    else:
        write(path, payload)
    decisions_sha = digest(path)
    decisions = read(path)['decisions']
    rows, errors = [], {}
    for meta, folder, key in cases:
        task = meta['task']
        slug = task['dataset']
        start, end = meta['test_targets']
        forecasts = candidate_arrays(folder/'test_candidates.npz', [start,end], raw[slug].shape[0])
        truth = np.asarray(raw[slug][:,start:end].T, dtype=np.float64)
        case_rows, columns = test_rows(meta, forecasts, truth, decisions[key], contract)
        rows.extend(case_rows)
        errors[slug,task['seed'],task['arm']] = columns
    if digest(path) != decisions_sha:
        raise ValueError('Archived decisions changed during scoring')
    with (store/'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparison = summarize(rows, errors, contract)
    write(store/'comparison.json', comparison)
    write_summary(store/'SUMMARY.md', comparison, len(records))
    return {'method_evaluated': True, 'scientific_status': 'HELDOUT_GPT2_SCORED',
            'S01_independent_review': 'APPROVED_FOR_PINNED_HELDOUT_SCORER',
            'review_identity': manifest['review_identity'], 'review_disposition_sha256': approved,
            'candidate_identity': manifest['candidate_identity'], 'candidate_inventory_sha256': manifest['inventory_sha256'],
            'decisions_sha256': decisions_sha, 'frozen_candidates_scored': len(records), 'policy_comparisons': len(rows),
            'new_optimizer_steps': 0, 'evaluation_store': str(store)}


def score(project, code, report):
    manifest, records, _, _ = review_inputs(project, code)
    approved = admission(project, manifest)
    evaluation = fingerprint({'review': manifest['review_identity'], 'approval': approved})
    store = inside(project, project/'outputs/ectt/heldout_gpt2_scores'/evaluation)
    store.mkdir(parents=True, exist_ok=True)
    with (store/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not completed(store, evaluation):
            contract, _ = load_contract(code)
            result = evaluate(project, code, store, records, manifest, approved, contract)
            write(store/'transfer_result.json', result)
            finish(store, evaluation)
        for name in ('SUMMARY.md', 'transfer_result.json', 'metrics.csv', 'comparison.json', 'decisions.json'):
            shutil.copyfile(store/name, report/name)
    return 0


def dispatch(project, code, report):
    from ectt_env import ensure
    with (project/'work/acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        python = ensure(project, code, report)
    return subprocess.call([str(python), '-u', str(code/'scripts/heldout_score.py'), '--project', str(project),
                            '--worker-report', str(report)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build-packet', action='store_true')
    mode.add_argument('--worker-report', type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    code = Path(__file__).resolve().parents[1]
    if args.build_packet:
        build_packet(project, code)
        return 0
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or os.environ.get('TRAFFIC_STAGE') != STAGE:
        raise ValueError('Run the held-out scorer with the Slurm submission stage')
    report = inside(project, args.worker_report)
    if report != project/'results/slurm_jobs'/job/'report':
        raise ValueError('Unexpected held-out scoring report path')
    try:
        return score(project, code, report)
    except Exception as exc:
        (report/'SUMMARY.md').write_text('# Held-out scoring failed\n\n'+str(exc)+'\n')
        raise


if __name__ == '__main__':
    sys.exit(main())
