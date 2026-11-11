"""Score the reviewed frozen development pilot and retain every paired outcome."""
import csv
import fcntl
import hashlib
import json
import math
import shutil
from statistics import mean

from ectt_common import completed, digest, finish, inside, read, write
from transfer_plan import fingerprint
import verification_pilot as pilot


def admission(project, code, plan=None):
    plan = plan if plan is not None else pilot.foundation(project, code)
    path = pilot.approval_path(project, plan)
    if not path.is_file():
        raise ValueError(f'Independent S01 review disposition is required at {path}')
    content = path.read_bytes()
    record = json.loads(content)
    if (record.get('status') != 'APPROVED' or record.get('independent_review_complete') is not True
            or record.get('pilot_identity') != plan['identity']
            or record.get('reviewed_hashes') != pilot.reviewed_hashes(code)
            or any(not isinstance(record.get(key), str) or not record[key].strip()
                   for key in ('reviewer', 'date', 'disposition'))):
        raise ValueError('Independent S01 disposition does not approve the exact pilot and scorer')
    report = pilot.job_report(project, pilot.jobs(project, plan)['report'], pilot.REPORT)
    result = read(report/'transfer_result.json')
    if (result.get('identity') != plan['identity'] or result.get('failed')
            or result.get('scientific_status') != 'ECTT_PILOT_CANDIDATES_READY'
            or result.get('diagnostic_cases_checked') != len(plan['tasks'])):
        raise ValueError('Complete frozen pilot candidate collection is required')
    return hashlib.sha256(content).hexdigest()


def candidate_arrays(path, bounds, nodes):
    import numpy as np
    with np.load(path, allow_pickle=False) as values:
        if set(values.files) != {'origins', 'local', 'transfer'}:
            raise ValueError('Frozen candidate shard has unexpected fields')
        if not np.array_equal(values['origins'], np.arange(*bounds)):
            raise ValueError('Frozen candidate origin ledger differs')
        arrays = {name: np.asarray(values[name], dtype=np.float64) for name in ('local', 'transfer')}
    for array in arrays.values():
        if array.shape != (bounds[1]-bounds[0], nodes) or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError('Invalid frozen candidate predictions')
    return arrays


def frozen_cases(project, plan, contract):
    import numpy as np
    root = pilot.store_path(project, plan)
    budget = plan['recipe']['budget']
    cases, comparators = [], {}
    for task in plan['tasks']:
        case = root/'cases'/str(task['index'])
        if not completed(case, plan['identity']):
            raise ValueError('Missing frozen candidate case')
        result = read(case/'result.json')
        manifest = read(case/'frozen_candidates.json')
        artifacts = read(case/'COMPLETE.json')['artifacts']
        residual = read(case/'target/latest.json')
        slug = task['dataset']
        profile = contract['datasets'][slug]
        ledger = profile['budgets'][budget]
        nodes = profile['shape'][0]
        verification, test = ledger['verification_target_rows'], profile['test_targets']
        support = ledger['support_raw']
        scale = manifest['support_scale']
        if (result['task'] != task or manifest['task'] != task or slug in task['sources']
                or result['scientific_status'] != 'ECTT_PILOT_FROZEN_CASE_COMPLETE'
                or result['source']['task'] != task
                or result['target']['optimizer_steps'] != contract['training_design']['target_updates']
                or artifacts['best.pt'] != manifest['source_checkpoint_sha256']
                or result['source']['best_sha256'] != manifest['source_checkpoint_sha256']
                or artifacts['target/'+residual['file']] != manifest['target_residual_sha256']
                or residual['sha256'] != manifest['target_residual_sha256']
                or manifest['budget'] != budget or manifest['support_raw'] != support
                or manifest['support_targets'] != ledger['support_target_rows']
                or manifest['unique_support_values'] != (support[1]-support[0])*nodes
                or manifest['target_container_sha256'] != profile['container_sha256']
                or manifest['verification_targets'] != verification or manifest['test_targets'] != test
                or manifest['verification_history_union'] != [verification[0]-12, verification[1]-1]
                or manifest['test_history_union'] != [test[0]-12, test[1]-1]
                or not support[1] <= verification[0] < verification[1] <= test[0]
                or manifest['verification_labels_used_for_scoring'] or manifest['test_labels_used_for_scoring']
                or not math.isfinite(scale) or scale < 1):
            raise ValueError('Frozen candidate lineage or access ledger differs')
        prefix = candidate_arrays(case/'verification_candidates.npz', verification, nodes)
        forecasts = candidate_arrays(case/'test_candidates.npz', test, nodes)
        pair = (slug, task['seed'])
        reference = (manifest['scaler'], scale, manifest['local_completion_sha256'])
        if pair not in comparators:
            comparators[pair] = reference, prefix['local'], forecasts['local']
        else:
            old, old_prefix, old_test = comparators[pair]
            if reference != old or not np.array_equal(prefix['local'], old_prefix) or not np.array_equal(forecasts['local'], old_test):
                raise ValueError('Matched arms have different local candidates or support scalers')
        cases.append({'task': task, 'manifest': manifest, 'result': result,
                      'prefix': prefix, 'forecasts': forecasts, 'completion_sha256': digest(case/'COMPLETE.json')})
    return cases


def decide(case, raw, contract):
    import numpy as np
    from frozen_verifier_replay import replay

    manifest = case['manifest']
    start, end = manifest['verification_targets']
    prefix = case['prefix']
    consumed = []

    def matured(origin):
        if origin != start+len(consumed) or origin >= end:
            raise ValueError('Verification reader requested a nonchronological label')
        consumed.append(origin)
        return raw[:, origin]

    result = replay(np.arange(start, end), prefix['local'], prefix['transfer'], manifest['support_scale'],
                    matured, rho=contract['verifier']['rho'], delta=contract['verifier']['delta_per_stream'],
                    eta=contract['training_design']['mechanism_coefficients']['epsilon'],
                    minimum=contract['partitions']['minimum_verification_origins'])
    result['read_origins'] = consumed
    result['frozen_case_sha256'] = case['completion_sha256']
    return result


def score_test(case, truth, decision, contract):
    import numpy as np
    from shared_baselines import independent_scores

    forecasts = case['forecasts']
    manifest = case['manifest']
    if truth.shape != forecasts['local'].shape or not np.isfinite(truth).all() or np.any(truth < 0):
        raise ValueError('Invalid test truth cohort')
    selected = forecasts[decision['selected_candidate']]
    local_error = np.abs(forecasts['local']-truth).mean(axis=1)
    transfer_error = np.abs(forecasts['transfer']-truth).mean(axis=1)
    deployed_error = np.abs(selected-truth).mean(axis=1)
    excess = (deployed_error-local_error)/manifest['support_scale']
    eta = contract['training_design']['mechanism_coefficients']['epsilon']
    local = independent_scores(forecasts['local'], truth)
    transfer = independent_scores(forecasts['transfer'], truth)
    deployed = independent_scores(selected, truth)
    row = {k: case['task'][k] for k in ('family', 'dataset', 'seed', 'arm')}
    row.update(local_MAE=local['MAE'], unconditional_transfer_MAE=transfer['MAE'],
               deployed_MAE=deployed['MAE'], deployed_RMSE=deployed['RMSE'],
               deployed_WMAPE_percent=deployed['WMAPE_percent'], accepted=decision['accepted'],
               useful_acceptance=decision['accepted'] and deployed['MAE'] < local['MAE'],
               harmful_acceptance=decision['accepted'] and deployed['MAE'] > local['MAE'],
               censored=decision['censored'], restricted_first_certificate=decision['first_certificate'],
               restricted_evidence_fraction=decision['restricted_evidence_fraction'],
               verification_origins_read=decision['verification_origins_read'],
               verification_node_values_read=decision['verification_node_values_read'],
               support_node_values=manifest['unique_support_values'],
               total_target_node_values=manifest['unique_support_values']+decision['verification_node_values_read'],
               test_origins=len(truth), material_harm_fraction=float(np.mean(excess > eta)),
               accepted_material_harm_fraction=float(np.mean(excess > eta)) if decision['accepted'] else None,
               positive_scaled_excess_mean=float(np.maximum(excess, 0).mean()),
               positive_scaled_excess_p90=float(np.quantile(np.maximum(excess, 0), .9)),
               unconditional_material_harm_fraction=float(np.mean((transfer_error-local_error)/manifest['support_scale'] > eta)),
               source_optimizer_steps=case['result']['source']['source_optimizer_steps'],
               target_optimizer_steps=case['result']['target']['optimizer_steps'],
               source_elapsed_seconds=case['result']['source']['runtime']['elapsed_seconds'],
               target_elapsed_seconds=case['result']['target']['runtime']['elapsed_seconds'])
    return row, np.column_stack((local_error, deployed_error))


def compare(rows, errors, plan, contract):
    import numpy as np
    from transfer_parallel import block_means

    arms = plan['recipe']['arms']
    domains = plan['recipe']['development_targets']
    by = {(r['dataset'], r['seed'], r['arm']): r for r in rows}
    groups = {}
    panel = {}
    for slug in domains:
        groups[slug] = {}
        columns = []
        for arm in arms:
            selected = [by[slug, seed, arm] for seed in plan['recipe']['seeds']]
            groups[slug][arm] = {key: mean(r[key] for r in selected) for key in
                                 ('local_MAE', 'deployed_MAE', 'unconditional_transfer_MAE',
                                  'restricted_evidence_fraction', 'verification_origins_read',
                                  'material_harm_fraction', 'total_target_node_values')}
            groups[slug][arm].update(accepted=sum(r['accepted'] for r in selected),
                                     useful_acceptance=sum(r['useful_acceptance'] for r in selected))
            averaged = np.mean([errors[slug, seed, arm] for seed in plan['recipe']['seeds']], axis=0)
            if not columns:
                columns.append(averaged[:, 0])
            columns.append(averaged[:, 1])
        panel[slug] = np.column_stack(columns)

    effects = {}
    for control in arms[:-1]:
        differences = [by[slug, seed, arms[-1]]['restricted_evidence_fraction']
                       -by[slug, seed, control]['restricted_evidence_fraction']
                       for slug in domains for seed in plan['recipe']['seeds']]
        effects[control] = {'paired_restricted_evidence_differences': differences,
                            'mean_restricted_evidence_difference': mean(differences),
                            'deployed_relative_MAE_difference_intervals': {}}
    settings = contract['statistics']
    for length in (settings['block_steps'], *settings['sensitivity_blocks']):
        rng = np.random.default_rng(20261111)
        sampled = []
        for slug in domains:
            blocks = block_means(panel[slug], length, settings['paired_bootstrap_replicates'], rng)
            sampled.append(np.divide(blocks[:, 1:], blocks[:, :1], out=np.full_like(blocks[:, 1:], np.nan),
                                     where=blocks[:, :1] > 0))
        macro = np.mean(sampled, axis=0)
        for i, control in enumerate(arms[:-1]):
            values = macro[:, -1]-macro[:, i]
            effects[control]['deployed_relative_MAE_difference_intervals'][str(length)] = (
                np.quantile(values, [.025, .975]).tolist() if np.isfinite(values).all() else None)
    macro = {}
    for arm in arms:
        means = [groups[slug][arm] for slug in domains]
        macro[arm] = {'deployed_relative_MAE': mean(r['deployed_MAE']/r['local_MAE'] for r in means)
                     if all(r['local_MAE'] > 0 for r in means) else None,
                     'restricted_evidence_fraction': mean(r['restricted_evidence_fraction'] for r in means),
                     'material_harm_fraction': mean(r['material_harm_fraction'] for r in means),
                     'acceptance_fraction': sum(r['accepted'] for r in means)/(len(domains)*len(plan['recipe']['seeds'])),
                     'useful_acceptances': sum(r['useful_acceptance'] for r in means)}
    aware = macro[arms[-1]]
    screen = (aware['acceptance_fraction'] >= contract['success_criteria']['activation_floor']
              and aware['useful_acceptances'] > 0 and all(
                  aware['deployed_relative_MAE'] is not None and macro[arm]['deployed_relative_MAE'] is not None
                  and aware['deployed_relative_MAE'] <= .98*macro[arm]['deployed_relative_MAE']
                  and aware['restricted_evidence_fraction'] < macro[arm]['restricted_evidence_fraction']
                  and aware['material_harm_fraction'] <= macro[arm]['material_harm_fraction'] for arm in arms[:-1]))
    return {'groups': groups, 'macro': macro, 'paired_comparisons': effects,
            'supports_held_out_design_review': screen,
            'interval_scope': 'Descriptive paired test-row blocks. Seeds averaged within each domain. Evidence differences are descriptive.'}


def report_summary(result, plan):
    lines = ['# ECTT verification-aware training development pilot', '',
             'All 18 fits were scored with the identical verifier after independent S01 review.', '',
             '| Domain | Source objective | Local MAE | Always-transfer MAE | Deployed MAE | Accepted | Mean verification rows read |',
             '|---|---|---:|---:|---:|---:|---:|']
    for domain in plan['recipe']['development_targets']:
        for arm in plan['recipe']['arms']:
            row = result['comparison']['groups'][domain][arm]
            lines.append(f"| {domain} | {arm} | {row['local_MAE']:.3f} | {row['unconditional_transfer_MAE']:.3f} | "
                         f"{row['deployed_MAE']:.3f} | {row['accepted']}/3 | {row['verification_origins_read']:.1f} |")
    lines += ['', 'Nonacceptance uses local forecasts at every test origin and is censored at available prefix + 1.',
              'Per-seed label counts, harm and accuracy are in pilot_metrics.csv. Paired summaries are in pilot_comparison.json.',
              'Confidence sequences concern observed verification prefixes. Test effects and row-block intervals are descriptive.',
              'Development screen for held-out design review: '+str(result['comparison']['supports_held_out_design_review']), '']
    return '\n'.join(lines)


def score(project, code, report, plan, contract, progress):
    import numpy as np
    from transfer_worker import raw_arrays, raw_paths, save_arrays

    review_sha = admission(project, code, plan)
    identity = fingerprint({'pilot': plan['identity'], 'review': review_sha})
    store = pilot.store_path(project, plan)/'evaluations'/identity
    store.mkdir(parents=True, exist_ok=True)
    outputs = ('SUMMARY.md', 'pilot_metrics.csv', 'pilot_comparison.json', 'decisions.json')
    with (store/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not completed(store, identity):
            cases = frozen_cases(project, plan, contract)
            raw = raw_arrays(raw_paths(project, code, contract), plan['recipe']['development_targets'], contract)
            decisions = {}
            for case in cases:
                task = case['task']
                index = str(task['index'])
                decisions[index] = decide(case, raw[task['dataset']], contract)
                progress(phase='VERIFICATION', decisions_completed=len(decisions), total_cases=len(cases))
            write(store/'decisions.json', {'identity': identity, 'decisions': decisions,
                                          'review_disposition_sha256': review_sha})
            decision_sha = digest(store/'decisions.json')
            rows, errors = [], {}
            for case in cases:
                task = case['task']
                index = str(task['index'])
                start, end = case['manifest']['test_targets']
                truth = np.asarray(raw[task['dataset']][:, start:end].T, dtype=np.float64)
                decision = decisions[index]
                row, values = score_test(case, truth, decision, contract)
                rows.append(row)
                errors[task['dataset'], task['seed'], task['arm']] = values
                save_arrays(store/(index+'_test.npz'), origins=np.arange(start, end), truth=truth,
                            **case['forecasts'], deployed=case['forecasts'][decision['selected_candidate']])
                progress(phase='TEST_SCORING', cases_scored=len(rows), total_cases=len(cases))
            if digest(store/'decisions.json') != decision_sha:
                raise ValueError('Frozen decisions changed during test scoring')
            with (store/'pilot_metrics.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            comparison = compare(rows, errors, plan, contract)
            write(store/'pilot_comparison.json', comparison)
            result = {'scientific_status': 'ECTT_DEVELOPMENT_PILOT_SCORED', 'method_evaluated': True,
                      'S01_independent_review': 'APPROVED_FOR_PINNED_PILOT', 'new_optimizer_steps': 0,
                      'diagnostic_cases_checked': len(rows), 'review_disposition_sha256': review_sha,
                      'decisions_sha256': decision_sha, 'comparison': comparison, 'evaluation_store': str(store)}
            write(store/'result.json', result)
            (store/'SUMMARY.md').write_text(report_summary(result, plan))
            finish(store, identity)
        for name in outputs:
            shutil.copyfile(store/name, report/name)
        return read(store/'result.json')
