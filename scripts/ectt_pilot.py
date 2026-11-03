"""Real support-only Qwen engineering updates, not an ECTT benchmark."""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from ectt_common import digest, inside, read, write


def weights(project, contract, progress):
    from huggingface_hub import HfApi, snapshot_download
    model = contract['model']
    directory = inside(project, project / 'work/models/qwen2_5_1_5b' / model['revision'])
    marker = directory / 'PINNED_FILES.json'
    if marker.exists():
        inventory = read(marker)
        if inventory['revision'] != model['revision']:
            raise ValueError('Model revision marker mismatch')
    else:
        for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
            os.environ.pop(name, None)
        progress(phase='QWEN_METADATA_DOWNLOAD', dataset_downloads=0)
        info = HfApi().model_info(model['backbone'], revision=model['revision'], files_metadata=True)
        if info.sha != model['revision']:
            raise ValueError('Hub resolved a different model revision')
        files = []
        for sibling in info.siblings:
            name = sibling.rfilename
            if name == 'config.json' or name.endswith('.safetensors') or name == 'model.safetensors.index.json':
                inside(directory, directory / name)
                lfs = sibling.lfs
                sha = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
                files.append({'name': name, 'size': sibling.size, 'sha256': sha, 'git_blob_sha1': sibling.blob_id})
        if not files or not any(f['name'].endswith('.safetensors') for f in files):
            raise ValueError('No pinned safe model tensors in Hub inventory')
        if shutil.disk_usage(project).free < sum(f['size'] for f in files)+2*2**30:
            raise RuntimeError('Insufficient project space for Qwen tensors and temporary files')
        snapshot_download(model['backbone'], revision=model['revision'], local_dir=str(directory),
            allow_patterns=[f['name'] for f in files], max_workers=5)
        inventory = {'repository': model['backbone'], 'revision': model['revision'], 'files': files}
    for item in inventory['files']:
        path = inside(directory, directory / item['name'])
        if not path.is_file() or path.stat().st_size != item['size']:
            raise ValueError('Pinned model file missing/wrong size: ' + str(path))
        if item['sha256']:
            valid = digest(path) == item['sha256']
        else:
            data = path.read_bytes()
            valid = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == item['git_blob_sha1']
        if not valid:
            raise ValueError('Pinned model tensor/config digest mismatch: ' + str(path))
    write(marker, inventory)
    return directory


def trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, state):
    current = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(current) != set(state):
        raise ValueError('Trainable checkpoint parameter inventory differs')
    import torch
    with torch.no_grad():
        for name, parameter in current.items():
            if parameter.shape != state[name].shape:
                raise ValueError('Trainable checkpoint shape differs')
            parameter.copy_(state[name].to(parameter))


def fit(model, inputs, targets, store, identity, steps, learning_rate, progress, stop_after=None):
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=.01)
    pointer = store / 'LATEST.json'
    step, losses = 0, []
    if pointer.exists():
        link = read(pointer)
        path = inside(store, store / link['file'])
        if digest(path) != link['sha256']:
            raise ValueError('Pilot checkpoint digest differs')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['identity'] != identity or saved['steps'] != steps:
            raise ValueError('Pilot checkpoint belongs to a different contract')
        restore_trainable(model, saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng'])
        if inputs.is_cuda:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        step, losses = saved['step'], saved['losses']
    initial_step = step
    if not 0 <= step <= steps:
        raise ValueError('Checkpoint optimizer cursor outside pilot budget')
    for i in range(step, steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = torch.nn.functional.mse_loss(prediction, targets)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite pilot loss')
        loss.backward()
        gradients = [p.grad for p in params if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError('Absent/nonfinite pilot gradients')
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        state = {'identity': identity, 'step': i+1, 'steps': steps, 'model': trainable_state(model),
            'optimizer': optimizer.state_dict(), 'losses': losses, 'cpu_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if inputs.is_cuda else []}
        checkpoint = store / ('step-%03d.pt' % (i+1))
        temporary = checkpoint.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(checkpoint)
        write(pointer, {'file': checkpoint.name, 'sha256': digest(checkpoint), 'step': i+1})
        progress(phase='SUPPORT_TRAINING_SMOKE', optimizer_steps=i+1, expected_steps=steps,
                 latest_training_mse=losses[-1], resumed_from_step=initial_step)
        if stop_after is not None and i+1 >= stop_after:
            return {'completed': False, 'steps': i+1, 'losses': losses}
    return {'completed': True, 'optimizer_steps': steps, 'new_optimizer_steps': steps-initial_step,
            'resumed_from_step': initial_step, 'losses': losses}


def run(project, code, store, contract, execution, ident, progress):
    import numpy as np
    import torch
    from ectt_data import fit_scaler, windows
    from ectt_model import qwen_model, invariance_checks
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA GPU; CPU fallback forbidden')
    if int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) != 5:
        raise RuntimeError('Pilot requires the requested five-CPU Slurm allocation')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 capability unavailable; record and amend before changing precision')
    torch.set_num_threads(5)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    recipe = execution['pilot']
    torch.manual_seed(recipe['seed'])
    torch.cuda.manual_seed_all(recipe['seed'])
    profile = contract['datasets'][recipe['dataset']]
    lock = read(code / 'config/sources.lock.json')
    path = inside(project, project / 'work/reference/snapshots' / lock['reference_commit'] / profile['path'])
    if digest(path) != profile['container_sha256']:
        raise ValueError('Canonical data changed after shared-data admission')
    raw = np.load(path, mmap_mode='r', allow_pickle=False)[..., 0]
    budget = profile['budgets'][recipe['budget']]
    scaler = fit_scaler(raw, budget['support_raw'])
    start = budget['support_target_rows'][0]
    origins = list(range(start, start+recipe['window_origins']))
    if origins[-1] >= budget['support_target_rows'][1]:
        raise ValueError('Pilot origin is outside support')
    x, y = windows(raw, origins, scaler)
    directory = weights(project, contract, progress)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    progress(phase='QWEN_LOADING', support_origins=origins, test_labels_read_by_learner=0)
    model = qwen_model(directory, contract).cuda()
    inputs = torch.from_numpy(x).cuda()
    target = torch.from_numpy(((y-scaler['minimum'])/(scaler['maximum']-scaler['minimum'])).astype('float32')).cuda()
    # BF16 kernels can have small batch-shape rounding differences. The
    # engineering tolerance is recorded; failure never changes it silently.
    def check_invariance(label):
        def record(rows):
            write(store / (label+'_invariance.json'), {'atol': 1e-3, 'rtol': 1e-3, 'checks': rows})
            progress(phase='QWEN_INVARIANCE', invariance_phase=label,
                     invariance_checks_completed=len(rows), latest_invariance_check=rows[-1])
        return invariance_checks(model, inputs, atol=1e-3, rtol=1e-3, record=record)
    pre = check_invariance('pretraining')
    began = time.monotonic()
    outcome = fit(model, inputs, target, store, ident, recipe['optimizer_steps'],
                  contract['training_design']['source_learning_rate'], progress)
    post = check_invariance('posttraining')
    # Fresh reconstruction and strict adapter reload must reproduce predictions
    # at the declared precision. Frozen base revision stays pinned.
    model.eval()
    with torch.no_grad():
        expected = model(inputs).cpu()
    state = trainable_state(model)
    parameter_counts = {'total': sum(p.numel() for p in model.parameters()),
        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    del model
    torch.cuda.empty_cache()
    torch.manual_seed(recipe['seed'])
    recovered = qwen_model(directory, contract).cuda()
    restore_trainable(recovered, state)
    recovered.eval()
    with torch.no_grad():
        observed = recovered(inputs).cpu()
    if not torch.equal(observed, expected):
        raise ValueError('Strict pilot reconstruction changed predictions on the same GPU/runtime')
    result = {'stage': 'ectt-gpu-pilot', 'execution_kind': 'REAL_GPU_SUPPORT_SMOKE',
        'dataset': recipe['dataset'], 'seed': recipe['seed'], 'support_origins': origins, 'scaler': scaler,
        'training_performed': outcome['new_optimizer_steps'] > 0, 'fit': outcome,
        'benchmark_experiments_completed': 0, 'new_method_objective_evaluated': False,
        'test_evaluation_performed': False, 'pretraining': pre, 'posttraining': post,
        'strict_reconstruction_pass': True, 'allocated_cpus': 5, 'visible_gpus': 1,
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'parameter_counts': parameter_counts, 'elapsed_training_seconds': time.monotonic()-began,
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'spatial_pool_accumulation_dtype': 'float64', 'spatial_pool_output_dtype': 'float32',
        'scientific_status': 'METHOD_NOT_EVALUATED', 'S01_independent_review': 'PENDING'}
    write(store / 'result.json', result)
    (store / 'SUMMARY.md').write_text('# Qwen numerical engineering pilot\n\n'+json.dumps(result, indent=2)+
        '\n\nReal updates use training-support windows only. No benchmark accuracy or ECTT objective result.\n'+
        'Stop here for independent statistical review and matched baseline implementation.\n')
    return result
