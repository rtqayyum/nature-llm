"""Full-size, cached-only Qwen recovery with a controlled process restart."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from ectt_common import read, write, digest, inside, completed, finish


def cached_weights(project, contract):
    model = contract['model']
    directory = inside(project, project/'work/models/qwen2_5_1_5b'/model['revision'])
    inventory = read(directory/'PINNED_FILES.json')
    if inventory.get('repository') != model['backbone'] or inventory.get('revision') != model['revision']:
        raise ValueError('Cached Qwen identity mismatch; downloads are not part of this check')
    names = [item['name'] for item in inventory['files']]
    if len(set(names)) != len(names) or 'config.json' not in names or not any(n.endswith('.safetensors') for n in names):
        raise ValueError('Incomplete cached Qwen safe-tensor inventory')
    for item in inventory['files']:
        path = inside(directory, directory/item['name'])
        if not path.is_file() or path.stat().st_size != item['size']:
            raise ValueError('Missing/wrong-size cached Qwen file: '+item['name'])
        if item['sha256']:
            valid = digest(path) == item['sha256']
        else:
            data = path.read_bytes()
            valid = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == item['git_blob_sha1']
        if not valid:
            raise ValueError('Corrupt cached Qwen file: '+item['name'])
    return directory, digest(directory/'PINNED_FILES.json')


def load_latest(store, expected):
    import torch
    pointer = read(store/'LATEST.json'); path = inside(store, store/pointer['file'])
    if digest(path) != pointer['sha256']:
        raise ValueError('Corrupt Qwen recovery checkpoint')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['identity'] != expected or payload['step'] != pointer['step']:
        raise ValueError('Incompatible Qwen recovery checkpoint')
    return payload, path


def equal_state(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) == type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return type(a) == type(b) and a == b


def segment(model, inputs, targets, store, ident, steps, rate, phase, interrupt_after, progress):
    """Generic segment engine; tests use small synthetic models, never fake Qwen."""
    import torch
    from ectt_pilot import fit, restore_trainable
    store.mkdir(parents=True, exist_ok=True)
    before = read(store/'LATEST.json')['step'] if (store/'LATEST.json').exists() else 0
    if phase == 'interrupt' and before >= interrupt_after:
        # A checkpoint survived a prior process exit before receipt writing.
        saved, _ = load_latest(store, ident)
        if before != interrupt_after:
            raise ValueError('Unexpected controlled-interruption cursor')
        restore_trainable(model, saved['model'])
        outcome = {'completed': False, 'optimizer_steps': before, 'new_optimizer_steps': 0,
                   'resumed_from_step': before, 'losses': saved['losses']}
    else:
        outcome = fit(model, inputs, targets, store, ident, steps, rate, progress,
                      stop_after=interrupt_after if phase == 'interrupt' else None)
        saved, _ = load_latest(store, ident)
        outcome.update(optimizer_steps=saved['step'], new_optimizer_steps=saved['step']-before, resumed_from_step=before)
    saved, path = load_latest(store, ident)
    expected_step = interrupt_after if phase == 'interrupt' else steps
    if saved['step'] != expected_step or len(saved['losses']) != expected_step:
        raise ValueError('Qwen recovery segment cursor/loss trajectory differs')
    model.eval()
    with torch.no_grad():
        prediction = model(inputs).detach().cpu()
    prediction_path = store/(phase+'_predictions.pt')
    if before == expected_step and (store/(phase+'_receipt.json')).exists():
        receipt = verify_receipt(store, phase, ident)
        prior = torch.load(store/receipt['predictions'], map_location='cpu', weights_only=True)
        if not equal_state(prediction, prior):
            raise ValueError('Completed Qwen segment reconstruction changed predictions')
        return receipt
    temporary = prediction_path.with_suffix('.tmp'); torch.save(prediction, temporary); temporary.replace(prediction_path)
    receipt = {'identity': ident, 'phase': phase, 'optimizer_steps': expected_step,
               'new_optimizer_steps': outcome['new_optimizer_steps'], 'resumed_from_step': before,
               'checkpoint': path.name, 'checkpoint_sha256': digest(path),
               'predictions': prediction_path.name, 'predictions_sha256': digest(prediction_path),
               'planned_process_exit': phase == 'interrupt', 'pid': os.getpid()}
    write(store/(phase+'_receipt.json'), receipt)
    return receipt


def verify_receipt(store, phase, ident):
    receipt = read(store/(phase+'_receipt.json'))
    if receipt['identity'] != ident or receipt['phase'] != phase:
        raise ValueError('Qwen segment receipt identity differs')
    for path_field, hash_field in (('checkpoint', 'checkpoint_sha256'), ('predictions', 'predictions_sha256')):
        path = inside(store, store/receipt[path_field])
        if digest(path) != receipt[hash_field]:
            raise ValueError('Corrupt completed Qwen recovery segment')
    return receipt


def compare(full, resumed, ident, steps):
    import torch
    a, _ = load_latest(full, ident); b, _ = load_latest(resumed, ident)
    checks = {name: equal_state(a[name], b[name]) for name in ('model', 'optimizer', 'cpu_rng', 'cuda_rng', 'losses', 'step', 'steps', 'identity')}
    receipts = [verify_receipt(full, 'full', ident), verify_receipt(resumed, 'resume', ident)]
    predictions = [torch.load(store/receipt['predictions'], map_location='cpu', weights_only=True)
                   for store, receipt in zip((full, resumed), receipts)]
    checks['predictions'] = equal_state(*predictions)
    checks['cursor_complete'] = a['step'] == b['step'] == steps
    if not all(checks.values()):
        raise ValueError('Full-size Qwen exact recovery failed: '+str(checks))
    return checks


def verify_admission(arm, phase):
    admission = read(arm/(phase+'_admission.json'))
    names = {'single_origin', 'origin_permutation', 'later_companion_mutation', 'node_permutation'}
    for stage in ('pretraining', 'posttraining'):
        rows = admission[stage]
        if len(rows) != 4 or {r['case'] for r in rows} != names or not all(
                r.get('passed') is True and math.isfinite(r['maximum_absolute_difference']) for r in rows):
            raise ValueError('Incomplete/failed Qwen segment invariance admission')
    if admission.get('test_labels_read_by_learner') != 0 or admission.get('verification_labels_read_by_learner') != 0:
        raise ValueError('Qwen recovery data-access scope differs')
    return admission


def worker(project, code, report, store, ident, phase, manifest_sha):
    import numpy as np
    import torch
    from ectt_data import fit_scaler, windows
    from ectt_model import qwen_model, invariance_checks
    recipe = read(code/'config/pretransfer_checks_v1.json')['qwen_recovery']
    contract = read(code/'config/ectt_protocol_v1.json')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or os.environ.get('SLURM_CPUS_PER_TASK') != '5':
        raise ValueError('Qwen recovery requires one visible GPU and five CPUs')
    if not torch.cuda.is_bf16_supported() or torch.__version__ != '2.4.1+cu118' or np.__version__ != '1.26.4':
        raise ValueError('Pinned Qwen BF16 environment differs')
    torch.set_num_threads(5); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(recipe['seed']); torch.cuda.manual_seed_all(recipe['seed'])
    directory = inside(project, project/'work/models/qwen2_5_1_5b'/contract['model']['revision'])
    if digest(directory/'PINNED_FILES.json') != manifest_sha:
        raise ValueError('Qwen cache inventory changed after verification')
    profile = contract['datasets'][recipe['dataset']]
    source = read(code/'config/sources.lock.json')['reference_commit']
    path = inside(project, project/'work/reference/snapshots'/source/profile['path'])
    if digest(path) != profile['container_sha256']:
        raise ValueError('Qwen recovery support data changed')
    raw = np.load(path, mmap_mode='r', allow_pickle=False)[..., 0]
    budget = profile['budgets'][recipe['budget']]; scaler = fit_scaler(raw, budget['support_raw'])
    start, end = budget['support_target_rows']; origins = list(range(start, start+recipe['support_origins']))
    if origins[-1] >= end:
        raise ValueError('Recovery origins leave the paid support block')
    x, y = windows(raw, origins, scaler)
    inputs = torch.from_numpy(x).cuda()
    targets = torch.from_numpy(((y-scaler['minimum'])/(scaler['maximum']-scaler['minimum'])).astype('float32')).cuda()
    runtime_contract = {'study': ident, 'manifest_sha256': manifest_sha, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
        'origins': origins, 'scaler': scaler, 'inputs_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
        'targets_sha256': hashlib.sha256(targets.cpu().numpy().tobytes()).hexdigest()}
    run_identity = hashlib.sha256(json.dumps(runtime_contract, sort_keys=True).encode()).hexdigest()
    marker = store/'runtime_contract.json'
    if marker.exists() and read(marker) != runtime_contract:
        raise ValueError('Qwen recovery runtime/input contract differs; old checkpoints preserved')
    write(marker, runtime_contract)
    model = qwen_model(directory, contract).cuda()
    arm = store/('uninterrupted' if phase == 'full' else 'restarted')
    pre = invariance_checks(model, inputs, atol=1e-3, rtol=1e-3,
        record=lambda rows: write(arm/(phase+'_pretraining_invariance.json'), rows))
    stage_progress = report/'transfer_check_progress.json'
    base_new_steps = read(stage_progress).get('new_optimizer_steps', 0) if stage_progress.exists() else 0
    def progress(**values):
        cumulative = base_new_steps+values.get('optimizer_steps', 0)-values.get('resumed_from_step', 0)
        write(report/'qwen_segment_progress.json', {'segment': phase, 'cumulative_new_optimizer_steps': cumulative, **values})
        print('QWEN_RECOVERY_SEGMENT', phase, json.dumps(values), flush=True)
    began = time.monotonic()
    receipt = segment(model, inputs, targets, arm, run_identity, recipe['updates'],
        contract['training_design']['source_learning_rate'], phase, recipe['interrupt_after'], progress)
    post = invariance_checks(model, inputs, atol=1e-3, rtol=1e-3,
        record=lambda rows: write(arm/(phase+'_posttraining_invariance.json'), rows))
    write(arm/(phase+'_admission.json'), {'pretraining': pre, 'posttraining': post,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'elapsed_seconds': time.monotonic()-began, 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'support_origins': origins, 'test_labels_read_by_learner': 0, 'verification_labels_read_by_learner': 0})
    return 75 if phase == 'interrupt' else 0


def run(project, code, report, store, ident, progress):
    store.mkdir(parents=True, exist_ok=True)
    if completed(store, ident):
        return dict(read(store/'result.json'), new_optimizer_steps=0, reused=True)
    contract = read(code/'config/ectt_protocol_v1.json')
    _, manifest_sha = cached_weights(project, contract)
    new_steps = 0; child_ids = []; admissions = {}
    for phase in ('full', 'interrupt', 'resume'):
        arm = store/('uninterrupted' if phase == 'full' else 'restarted')
        receipt_path = arm/(phase+'_receipt.json')
        if receipt_path.exists() and (arm/(phase+'_admission.json')).exists():
            runtime = read(store/'runtime_contract.json')
            run_ident = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
            receipt = verify_receipt(arm, phase, run_ident)
            admissions[phase] = verify_admission(arm, phase)
            child_ids.append(receipt['pid'])
            continue
        progress(phase='QWEN_RECOVERY', recovery_segment=phase)
        previous_step = read(arm/'LATEST.json')['step'] if (arm/'LATEST.json').exists() else 0
        command = [sys.executable, '-u', str(code/'scripts/pretransfer_checks.py'), '--mode', 'qwen-segment',
            '--project', str(project), '--code', str(code), '--report', str(report), '--store', str(store),
            '--identity', ident, '--segment', phase, '--manifest-sha', manifest_sha]
        result = subprocess.run(command, env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                                             'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
        if result.returncode != (75 if phase == 'interrupt' else 0):
            raise ValueError('Qwen recovery child failed: '+phase+' exit='+str(result.returncode))
        runtime = read(store/'runtime_contract.json')
        run_ident = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
        receipt = verify_receipt(arm, phase, run_ident)
        admissions[phase] = verify_admission(arm, phase)
        new_steps += read(arm/'LATEST.json')['step']-previous_step; child_ids.append(receipt['pid'])
        progress(phase='QWEN_RECOVERY', recovery_segment=phase, new_optimizer_steps=new_steps)
    recipe = read(code/'config/pretransfer_checks_v1.json')['qwen_recovery']
    checks = compare(store/'uninterrupted', store/'restarted', run_ident, recipe['updates'])
    first = verify_receipt(store/'restarted', 'interrupt', run_ident)
    last = verify_receipt(store/'restarted', 'resume', run_ident)
    if first['optimizer_steps'] != recipe['interrupt_after'] or last['resumed_from_step'] != recipe['interrupt_after']:
        raise ValueError('Qwen did not resume from the controlled interruption cursor')
    # Cache verification is read-only; it never refreshes the pinned marker.
    cached_weights(project, contract)
    outcome = {'passed': True, 'checks': checks, 'controlled_process_restart': True,
        'segment_admission': admissions,
        'peak_gpu_allocated_bytes': max(a['peak_gpu_allocated_bytes'] for a in admissions.values()),
        'segment_process_ids': child_ids, 'interrupted_after_update': recipe['interrupt_after'],
        'optimizer_steps_per_trajectory': recipe['updates'], 'engineering_optimizer_steps': 2*recipe['updates'],
        'new_optimizer_steps': new_steps, 'reused': False, 'runtime_contract': runtime,
        'test_evaluation_performed': False, 'method_evaluated': False,
        'scope': 'Full-size Qwen/LoRA fixed-support source-optimizer path; future episodic/adaptation trainers require their own cursor recovery checks'}
    write(store/'result.json', outcome); finish(store, ident)
    return outcome
