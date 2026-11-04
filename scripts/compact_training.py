"""Atomic compact-model optimizer recovery, fitting and physical forecasts."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from compact_model import CompactTrafficModel
from ectt_common import read, write, digest, completed, finish, inside
from ectt_data import features, inverse


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def configure():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(5)


def data_batch(raw, origins, scaler, device, targets=False):
    history = np.stack([raw[:, i-12:i] for i in origins])
    x = torch.as_tensor(features(history, scaler), dtype=torch.float32, device=device)
    y = None
    if targets:
        truth = np.stack([raw[:, i] for i in origins])
        y = torch.as_tensor((truth-scaler['minimum'])/(scaler['maximum']-scaler['minimum']),
                            dtype=torch.float32, device=device)
    return x, y


def rng_state(generator):
    return {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'sampler': generator.get_state()}


def restore_rng(state, generator):
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['sampler'])


def load_checkpoint(store, expected):
    pointer = read(store/'latest.json')
    path = inside(store, store/pointer['file'])
    if digest(path) != pointer['sha256']:
        raise ValueError('Corrupt compact optimizer checkpoint; preserved for inspection')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['identity'] != expected or payload['step'] != pointer['step']:
        raise ValueError('Incompatible compact optimizer checkpoint')
    return payload


def save_checkpoint(store, expected, step, model, optimizer, generator, losses, elapsed):
    path = store/('step_%06d.pt' % step)
    temporary = path.with_suffix('.tmp')
    payload = {'identity': expected, 'step': step,
               'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
               'optimizer': optimizer.state_dict(), 'rng': rng_state(generator),
               'losses': losses, 'elapsed_training_seconds': elapsed}
    torch.save(payload, temporary); temporary.replace(path)
    write(store/'latest.json', {'file': path.name, 'step': step, 'sha256': digest(path)})


def fit(raw, bounds, scaler, architecture, recipe, seed, store, ident, device, progress, stop_after=None):
    store = Path(store); store.mkdir(parents=True, exist_ok=True)
    if bounds[1] <= bounds[0] or bounds[0] < 12:
        raise ValueError('Empty or invalid compact support targets')
    contract = {'study': ident, 'bounds': bounds, 'scaler': scaler, 'architecture': architecture,
                'recipe': recipe, 'seed': seed, 'torch': torch.__version__,
                'device': torch.cuda.get_device_name() if str(device).startswith('cuda') else 'SYNTHETIC_CPU'}
    expected = fingerprint(contract)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = CompactTrafficModel(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    generator = torch.Generator(device='cpu').manual_seed(seed+1)
    step, losses, previous_seconds = 0, [], 0.
    already_complete = completed(store, expected)
    if (store/'latest.json').exists():
        old = load_checkpoint(store, expected)
        model.load_state_dict(old['model'], strict=True); optimizer.load_state_dict(old['optimizer'])
        restore_rng(old['rng'], generator)
        step, losses, previous_seconds = old['step'], old['losses'], old['elapsed_training_seconds']
        if step != len(losses) or not 0 <= step <= recipe['updates']:
            raise ValueError('Invalid compact recovery cursor')
    elif already_complete:
        raise ValueError('Completed compact fit lacks its optimizer checkpoint')
    resumed = step; began = time.monotonic(); saved_at = began
    model.train()
    while step < recipe['updates']:
        origins = (torch.randint(bounds[0], bounds[1], (recipe['origin_batch'],), generator=generator)).tolist()
        x, y = data_batch(raw, origins, scaler, device, targets=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = torch.nn.functional.mse_loss(prediction, y)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite compact training loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['gradient_clip_norm'])
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite compact training gradient')
        optimizer.step(); step += 1; losses.append(float(loss.detach()))
        now = time.monotonic()
        save = (step % recipe['checkpoint_every_steps'] == 0 or step == recipe['updates']
                or now-saved_at >= recipe['checkpoint_max_seconds'] or step == stop_after)
        if save:
            save_checkpoint(store, expected, step, model, optimizer, generator, losses, previous_seconds+now-began)
            saved_at = now
        if save or step % 10 == 0:
            progress(optimizer_step=step, expected_steps=recipe['updates'],
                     new_optimizer_steps=step-resumed, resumed_from_step=resumed, latest_training_MSE=losses[-1])
        if step == stop_after and step < recipe['updates']:
            return model, {'completed': False, 'optimizer_steps': step, 'new_optimizer_steps': step-resumed}
    model.eval()
    info = {'completed': True, 'optimizer_steps': step, 'new_optimizer_steps': step-resumed,
            'resumed_from_step': resumed, 'losses': losses, 'architecture': architecture,
            'parameters': sum(p.numel() for p in model.parameters()), 'identity': expected,
            'elapsed_training_seconds': load_checkpoint(store, expected)['elapsed_training_seconds']}
    if not already_complete:
        write(store/'fit.json', {k: v for k, v in info.items() if k not in ('new_optimizer_steps', 'resumed_from_step')})
        finish(store, expected)
    progress(optimizer_step=step, expected_steps=recipe['updates'], new_optimizer_steps=step-resumed,
             resumed_from_step=resumed, latest_training_MSE=losses[-1])
    return model, info


def predict(model, raw, bounds, scaler, batch_size, device):
    result = []; model.eval()
    with torch.no_grad():
        for start in range(bounds[0], bounds[1], batch_size):
            origins = list(range(start, min(start+batch_size, bounds[1])))
            x, _ = data_batch(raw, origins, scaler, device)
            result.append(inverse(model(x).float().cpu().numpy(), scaler).astype(np.float64))
    if not result:
        raise ValueError('Empty compact prediction cohort')
    prediction = np.concatenate(result)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite compact predictions')
    return prediction


def equal_state(first, second):
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(equal_state(first[k], second[k]) for k in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(equal_state(a, b) for a, b in zip(first, second))
    return first == second


def recovery_probe(raw, bounds, scaler, architecture, recipe, seed, store, ident, device, progress):
    """Compare complete model/Adam/RNG/cursor after serialized interruption."""
    if completed(store, ident):
        return read(store/'result.json')
    probe = dict(recipe, updates=recipe['recovery_probe_updates'])
    fit(raw, bounds, scaler, architecture, probe, seed, store/'uninterrupted', ident, device,
        lambda **v: progress(probe_arm='uninterrupted', **v))
    # A replay after a finished probe reuses the completed interrupted arm.
    if not (store/'resumed/COMPLETE.json').exists():
        fit(raw, bounds, scaler, architecture, probe, seed, store/'resumed', ident, device,
            lambda **v: progress(probe_arm='interrupted', **v), stop_after=recipe['recovery_probe_split'])
    fit(raw, bounds, scaler, architecture, probe, seed, store/'resumed', ident, device,
        lambda **v: progress(probe_arm='resumed', **v))
    a = load_checkpoint(store/'uninterrupted', fingerprint_from_pointer(store/'uninterrupted'))
    b = load_checkpoint(store/'resumed', fingerprint_from_pointer(store/'resumed'))
    checks = {name: equal_state(a[name], b[name]) for name in ('model', 'optimizer', 'rng', 'step', 'losses', 'identity')}
    if not all(checks.values()):
        raise ValueError('Compact exact optimizer recovery failed: '+str(checks))
    result = {'passed': True, 'checks': checks, 'updates': probe['updates'],
              'interrupted_after': recipe['recovery_probe_split'],
              'scope': 'Compact model only; fresh model/Adam reconstruction on the same device'}
    write(store/'result.json', result); finish(store, ident)
    return result


def fingerprint_from_pointer(store):
    # The pointer hash is verified before opening the local, self-created file.
    pointer = read(store/'latest.json'); path = inside(store, store/pointer['file'])
    if digest(path) != pointer['sha256']:
        raise ValueError('Corrupt recovery probe checkpoint')
    return torch.load(path, map_location='cpu', weights_only=True)['identity']
