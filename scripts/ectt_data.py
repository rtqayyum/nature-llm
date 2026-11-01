"""Shared stage-two raw-flow contract. No network or released model execution."""
import csv
import hashlib
import json
import traceback
from pathlib import Path
import numpy as np
from ectt_common import digest, inside, read, write


def fit_scaler(raw, bounds):
    start, end = bounds
    values = np.asarray(raw[:, start:end], dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Scaler support is empty, negative or nonfinite')
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        raise ValueError('Degenerate support-only min/max; no epsilon substitution')
    return {'minimum': lo, 'maximum': hi, 'fit_rows': [start, end],
            'normalized_clipping': False, 'fit_scope': 'support_only'}


def features(history, scaler, mask=None):
    """B x nodes x 12 raw histories -> raw/trend/mask, no batch coupling."""
    x = np.asarray(history, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] != 12:
        raise ValueError('Expected batch x nodes x 12 histories')
    observed = np.ones(x.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if observed.shape != x.shape or not np.isfinite(x[observed]).all():
        raise ValueError('Invalid mask or observed input')
    lo, hi = scaler['minimum'], scaler['maximum']
    if hi <= lo:
        raise ValueError('Degenerate scaler')
    count = np.cumsum(observed, axis=-1)
    trend = np.divide(np.cumsum(np.where(observed, x, 0), axis=-1), count,
                      out=np.full(x.shape, lo), where=count > 0)
    # A missing raw value is filled by its observed prefix mean. With no prefix,
    # use the support minimum. Its mask remains zero in either case.
    filled = np.where(observed, x, trend)
    return np.stack(((filled-lo)/(hi-lo), (trend-lo)/(hi-lo), observed), axis=-1).astype(np.float32)


def inverse(z, scaler):
    z = np.asarray(z, dtype=np.float64)
    return np.maximum(0, scaler['minimum'] + z*(scaler['maximum']-scaler['minimum']))


def windows(raw, origins, scaler):
    origins = np.asarray(origins, dtype=np.int64)
    if len(origins) == 0 or origins.min() < 12 or origins.max() >= raw.shape[1]:
        raise ValueError('Invalid one-step origins')
    history = np.stack([raw[:, i-12:i] for i in origins])
    truth = np.stack([raw[:, i] for i in origins])
    return features(history, scaler), truth


def verify_profile(row):
    nodes, total, channels = row['shape']
    a, b = row['source_train_raw']
    if not (a == 0 and b == int(.6*total) and channels == 2 and nodes > 0):
        raise ValueError('Source training boundaries differ from the frozen fractions')
    if row['source_train_targets'] != [12, b] or row['source_validation_targets'] != [b, int(.7*total)]:
        raise ValueError('Source target/validation ranges differ')
    cutoff = int(.8*total)
    if row['test_targets'] != [cutoff, total]:
        raise ValueError('Test range differs')
    for q, budget in row['budgets'].items():
        size = int(float(q)*cutoff)
        start, middle = cutoff-size, cutoff-size+size//2
        support = [start, middle]
        verification = [middle, cutoff]
        eligible = size//2 >= 24 and cutoff-middle >= 12
        if budget['raw_rows'] != size or budget['support_raw'] != support or budget['verification_raw'] != verification:
            raise ValueError('Budget/raw split differs: ' + q)
        if budget['eligible_for_learned_adaptation'] != eligible:
            raise ValueError('Budget eligibility differs: ' + q)
        targets = [min(start+12, middle), middle]
        verification_targets = [min(max(middle, start+12), cutoff), cutoff]
        if budget['support_target_rows'] != targets or budget['verification_target_rows'] != verification_targets:
            raise ValueError('Budget target ledger differs: ' + q)


def write_origins(path, dataset, role, bounds):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset', 'role', 'target_row', 'history_start', 'history_end_exclusive'])
        writer.writerows((dataset, role, i, i-12, i) for i in range(*bounds))
    temporary.replace(path)


def audit_raw(path, profile):
    raw_array = np.load(path, mmap_mode='r', allow_pickle=False)
    if raw_array.dtype.hasobject or raw_array.dtype.kind not in 'iuf' or list(raw_array.shape) != profile['shape']:
        raise ValueError('Canonical array dtype/shape mismatch')
    raw = raw_array[..., 0]
    h = hashlib.sha256(json.dumps(list(raw.shape)).encode())
    lower, upper = np.inf, -np.inf
    zero = 0
    # Canonical value digest is independent of the unused second channel and
    # container dtype; fixed little-endian float64, chronological row order.
    for start in range(0, raw.shape[1], 512):
        block = np.asarray(raw[:, start:start+512].T, dtype='<f8', order='C')
        if not np.isfinite(block).all() or np.any(block < 0):
            raise ValueError('Canonical observed raw flow contains negative/nonfinite values')
        h.update(block.tobytes())
        lower, upper = min(lower, float(block.min())), max(upper, float(block.max()))
        zero += int((block == 0).sum())
    return raw, {'canonical_raw_sha256': h.hexdigest(), 'minimum': lower, 'maximum': upper,
                 'zero_observations': zero, 'nonfinite_observed': 0, 'negative_observed': 0,
                 'zero_is_observed': True, 'auxiliary_channel_read_as_feature': False}


def dataset_audit(project, code, store, slug, profile, lock):
    verify_profile(profile)
    source = inside(project, project / 'work/reference/snapshots' / lock['reference_commit'] / profile['path'])
    asset = next(a for a in lock['lfs_files'] if a['path'] == profile['path'])
    if asset['sha256'] != profile['container_sha256']:
        raise ValueError('Scientific contract differs from the acquisition lock')
    if not source.is_file() or source.stat().st_size != asset['size'] or digest(source) != asset['sha256']:
        raise ValueError('Missing/corrupt existing canonical asset; acquisition remains separate: ' + str(source))
    directory = store / 'datasets' / slug
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / 'dataset_audit.json'
    if cached.exists():
        result = read(cached)
        if result['container_sha256'] == profile['container_sha256'] and all(
            digest(inside(directory, directory / name)) == sha for name, sha in result['ledger_hashes'].items()):
            return {**result, 'resumed': True}
        raise ValueError('Partial dataset audit is corrupt; preserve and investigate: ' + str(cached))
    raw, statistics = audit_raw(source, profile)
    ledgers, budgets = {}, {}
    roles = {k: profile[k] for k in ('source_train_targets', 'source_validation_targets', 'test_targets')}
    for q, b in profile['budgets'].items():
        label = q.replace('.', 'p')
        roles[label+'_support'] = b['support_target_rows']
        roles[label+'_verification'] = b['verification_target_rows']
        eligible = b['eligible_for_learned_adaptation']
        result = {'eligible': eligible, 'raw_rows': b['raw_rows'],
                  'support_raw': b['support_raw'], 'verification_raw': b['verification_raw'],
                  'supervised_unique_value_cap': b['raw_rows']*raw.shape[0],
                  'learner_values_consumed_in_this_audit': 0,
                  'support_unique_values': (b['support_raw'][1]-b['support_raw'][0])*raw.shape[0],
                  'verification_unique_values': (b['verification_raw'][1]-b['verification_raw'][0])*raw.shape[0],
                  'node_range': [0, raw.shape[0]], 'ledger_format': 'complete rectangular row x node intervals; no sampled nodes',
                  'infeasible_reason': None if eligible else 'No positive budget or insufficient support/verification; persistence without extra labels',
                  'rolling_history_policy': 'Available histories are shared covariates; not additional fitting/verification access'}
        if eligible:
            scaler = fit_scaler(raw, b['support_raw'])
            first = b['verification_target_rows'][0]
            x, truth = windows(raw, [first], scaler)
            if not np.allclose(inverse((truth-scaler['minimum'])/(scaler['maximum']-scaler['minimum']), scaler), truth, rtol=1e-7, atol=1e-7):
                raise ValueError('Physical inverse failed')
            # All later observations can change without changing the issued
            # first verification history, support-fitted scaler or features.
            small = np.array(raw[:, b['support_raw'][0]:first+2], copy=True)
            offset = first-b['support_raw'][0]
            original = features(small[:, offset-12:offset][None], scaler)
            small[:, offset:] += statistics['maximum']+1000
            if not np.array_equal(original, features(small[:, offset-12:offset][None], scaler)):
                raise ValueError('Future mutation changed issued history features')
            if fit_scaler(small, [0, offset]) != {**scaler, 'fit_rows': [0, offset]}:
                raise ValueError('Future mutation changed support scaler')
            result.update(scaler=scaler, support_only_fit_verified=True, future_mutation_pass=True,
                          features_shape=list(x.shape), inverse_check_pass=True)
        budgets[q] = result
    for role, bounds in roles.items():
        path = directory / 'origin_ledgers' / (role+'.csv')
        write_origins(path, slug, role, bounds)
        ledgers[str(path.relative_to(directory))] = digest(path)
    result = {'dataset': slug, 'container_sha256': profile['container_sha256'], 'source_path': str(source),
              'shape': profile['shape'], 'statistics': statistics, 'budgets': budgets, 'ledger_hashes': ledgers,
              'status': 'NUMERIC_CONTRACT_VALIDATED', 'resumed': False,
              'semantics': 'Released registry describes channel zero as observed flow; original sensor/timestamp provenance remains unverified',
              'timestamps': 'ordered row indices only; no elapsed-time/calendar guarantee'}
    write(cached, result)
    return result


def run(project, code, store, contract, progress):
    lock = read(code / 'config/sources.lock.json')
    if digest(code / 'config/sources.lock.json') != contract['data']['source_lock_sha256']:
        raise ValueError('Acquisition source lock changed')
    # Reference closure is checked from real recorded status, never inferred
    # from a similarly named file or scientific rounding agreement.
    historical = project / 'results/slurm_jobs/610996/job_status.json'
    history = read(historical)
    if history.get('job_id') != '610996' or history.get('stage') != 'reproduction-review' or history.get('shell_exit_code') != 0:
        raise ValueError('Required historical reproduction-review completion is missing/failed')
    rows, failures = {}, {}
    logs = store / 'worker_logs'
    logs.mkdir(exist_ok=True)
    for slug, profile in contract['datasets'].items():
        try:
            rows[slug] = dataset_audit(project, code, store, slug, profile, lock)
            (logs / (slug+'.log')).write_text('NUMERIC_CONTRACT_VALIDATED\nNo inference or training performed.\n')
        except Exception as exc:
            failures[slug] = str(exc)
            (logs / (slug+'.log')).write_text(traceback.format_exc())
        progress(phase='DATA_AUDIT', completed_datasets=len(rows), failed_datasets=len(failures), total_datasets=8,
                 datasets={s: {'state': 'VALIDATED', 'resumed': d['resumed']} for s, d in rows.items()}, failures=failures)
    raw_groups = {}
    for slug, row in rows.items():
        raw_groups.setdefault(row['statistics']['canonical_raw_sha256'], []).append(slug)
    duplicates = [g for g in raw_groups.values() if len(g) > 1]
    result = {'stage': 'ectt-preflight', 'completed_datasets': len(rows), 'failed_datasets': len(failures),
              'datasets': rows, 'failures': failures, 'canonical_raw_duplicate_groups': duplicates,
              'historical_review_status_sha256': digest(historical), 'training_performed': False,
              'model_inference_performed': False, 'raw_sensor_provenance_verified': False,
              'absolute_timestamps_verified': False, 'scientific_status': 'METHOD_NOT_EVALUATED'}
    write(store / 'result.json', result)
    lines = ['# ECTT shared-data admission', '', 'Validated %d/8 domains; failures %d.' % (len(rows), len(failures)), '',
             '| Dataset | State | Resumed |', '|---|---|---|']
    lines += ['| %s | %s | %s |' % (s, 'FAILED' if s in failures else 'NUMERIC_CONTRACT_VALIDATED', rows.get(s, {}).get('resumed', False)) for s in contract['datasets']]
    lines += ['', 'Existing stage-two channel zero only. No new datasets, inference or training.',
              'Absolute timestamps and original sensor provenance remain unverified.',
              'Origin ledgers and compressed unique-value access ledgers: datasets/SLUG/.',
              'Complete planned experiment matrix: experiment_registry.json and experiment_families.csv.',
              'Canonical duplicate groups: '+json.dumps(duplicates), '',
              'Next implemented admission: ectt-stat-check, then ectt-gpu-pilot.',
              'S01 independent mathematical review and benchmark training kernels remain pending.', '']
    lines += [s+': '+reason for s, reason in failures.items()]
    (store / 'SUMMARY.md').write_text('\n'.join(lines))
    if failures or duplicates:
        raise ValueError('Data admission failed: '+json.dumps({'failures': failures, 'duplicates': duplicates}))
    return result
