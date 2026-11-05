"""Checkpointable source episodes and frozen-backbone support adaptation."""
import time
import uuid
import numpy as np
import torch
from compact_training import data_batch, equal_state, restore_rng, rng_state
from ectt_common import digest, inside, read, write
from ectt_data import fit_scaler, inverse
from shared_baselines import batches, fit_case, scores
from transfer_model import check_invariance, predict_encoded, residual_head


def model_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore_model(model, state):
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if params.keys() != state.keys():
        raise ValueError('Checkpoint trainable keys differ from the model')
    with torch.no_grad():
        for name, p in params.items():
            value = state[name]
            if p.shape != value.shape or p.dtype != value.dtype:
                raise ValueError('Checkpoint trainable tensor differs: '+name)
            p.copy_(value)


def save(directory, identity, model, optimizer, generator, cursor):
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory/'latest.json'
    old = read(pointer)['file'] if pointer.exists() else None
    path = directory/('checkpoint-'+uuid.uuid4().hex+'.pt')
    temp = path.with_suffix('.tmp')
    torch.save({'identity': identity, 'model': model_state(model), 'optimizer': optimizer.state_dict(),
                'rng': rng_state(generator), 'cursor': cursor}, temp)
    temp.replace(path)
    write(pointer, {'file': path.name, 'sha256': digest(path), 'step': cursor['step']})
    if old:
        inside(directory, directory/old).unlink(missing_ok=True)


def load(directory, identity):
    pointer = read(directory/'latest.json')
    path = inside(directory, directory/pointer['file'])
    if digest(path) != pointer['sha256']:
        raise ValueError('Training checkpoint digest failed: '+str(path))
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['identity'] != identity or payload['cursor']['step'] != pointer['step']:
        raise ValueError('Incompatible training checkpoint: '+str(path))
    return payload


def resume(directory, identity, model, optimizer, generator, initial):
    if not (directory/'latest.json').exists():
        return initial
    payload = load(directory, identity)
    restore_model(model, payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    restore_rng(payload['rng'], generator)
    return payload['cursor']


def generator(seed):
    return torch.Generator(device='cpu').manual_seed(seed)


def draw(rng, high, count):
    return torch.randint(high, (count,), generator=rng)


def update(loss, optimizer, clip):
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite training loss')
    loss.backward()
    step(optimizer, clip)


def step(optimizer, clip):
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def episode_bank(task, contract, recipe, raw):
    rng = generator(task['seed'])
    r = recipe['source_episode']
    s, e, q = (r[k] for k in ('support_rows', 'reserved_verification_rows', 'suffix_rows'))
    bank = {}
    for slug in task['sources']:
        p = contract['datasets'][slug]
        start, end = p['source_train_raw']
        last = end-s-e-q
        if last < start or end+q > p['source_validation_targets'][1]:
            raise ValueError('Source bounds cannot fit a frozen episode')

        def bounds(anchor):
            return {'support': [anchor, anchor+s], 'verification': [anchor+s, anchor+s+e],
                    'suffix': [anchor+s+e, anchor+s+e+q]}

        def usable(anchor):
            fitting = raw[slug][:, anchor:anchor+int(.7*s)]
            return fitting.max() > fitting.min()

        anchors = []
        for offset in torch.randperm(last-start+1, generator=rng).tolist():
            anchor = start+offset
            if usable(anchor):
                anchors.append(anchor)
            if len(anchors) == r['training_episodes_per_domain']:
                break
        validation = next(i for i in range(end-s-e, start-1, -1) if usable(i))
        if len(anchors) != r['training_episodes_per_domain']:
            raise ValueError('Insufficient nondegenerate source support blocks: '+slug)
        bank[slug] = {'train': [bounds(i) for i in anchors],
                      'validation': {'support': [validation, validation+s],
                                     'verification': [end-e, end], 'suffix': [end, end+q]}}
    return bank


def support_scale(raw, support, seed, ridge_recipe, compact_recipe, device, path):
    if path.exists():
        return read(path)['scale']
    a, b = support
    split = a+int(.7*(b-a))
    profile = {'budgets': {'1.0': {'eligible_for_learned_adaptation': True, 'support_raw': support,
                                 'support_target_rows': [a+12, b]}}}
    local = fit_case(raw, profile, '1.0', ridge_recipe)
    scale_candidates = {}
    for row in local['candidates']:
        name, value = row['family'], row['validation_MAE']
        if value is not None:
            scale_candidates[name] = min(value, scale_candidates.get(name, value))
    from compact_model import CompactTrafficModel
    from compact_training import predict
    truth = raw[:, split:b].T
    scaler = fit_scaler(raw, [a, split])
    # Keep comparator initialization/sampling independent of the outer RNG.
    devices = [torch.cuda.current_device()] if device == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        for architecture in compact_recipe['configurations']:
            torch.manual_seed(seed)
            rng = generator(seed+1)
            model = CompactTrafficModel(**architecture).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=compact_recipe['learning_rate'],
                                          weight_decay=compact_recipe['weight_decay'])
            for _ in range(compact_recipe['updates']):
                origins = (draw(rng, split-a-12, compact_recipe['origin_batch'])+a+12).tolist()
                x, y = data_batch(raw, origins, scaler, device, targets=True)
                update(torch.nn.functional.mse_loss(model(x), y), optimizer, compact_recipe['gradient_clip_norm'])
            pred = predict(model, raw, [split, b], scaler, compact_recipe['prediction_origin_batch'], device)
            value = scores(pred, truth)['MAE']
            scale_candidates['compact'] = min(value, scale_candidates.get('compact', value))
    family = min(compact_recipe['selection_order'], key=lambda name: scale_candidates.get(name, float('inf')))
    scale = max(1., scale_candidates[family])
    write(path, {'support': support, 'validation': [split, b], 'scale': scale,
                 'family': family, 'validation_MAE': scale_candidates})
    return scale


@torch.no_grad()
def support_embeddings(model, raw, bounds, scaler, microbatch, device):
    model.eval()
    parts, bases, labels = [], [], []
    for ids in batches(bounds, microbatch):
        x, y = data_batch(raw, ids, scaler, device, targets=True)
        encoded = model.encode(x)
        parts.append(encoded.cpu())
        bases.append(model.head(encoded).squeeze(-1).cpu())
        labels.append(y.cpu())
    return torch.cat(parts), torch.cat(bases), torch.cat(labels)


def adapt(model, raw, bounds, scaler, seed, recipe, training, device, progress,
          checkpoint=None, identity=None, stop_after=None, encoded=None):
    embeddings, base, truth = (encoded if encoded is not None else support_embeddings(
        model, raw, bounds, scaler, recipe['origin_microbatch'], device))
    devices = [torch.cuda.current_device()] if device == 'cuda' else []
    # The same source/residual initialization policy applies on every episode.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        head = residual_head(embeddings.shape[-1], device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=training['target_learning_rate'],
                                  weight_decay=training['weight_decay'])
    rng = generator(seed)
    cursor = {'step': 0, 'losses': []}
    if checkpoint is not None:
        cursor = resume(checkpoint, identity, head, optimizer, rng, cursor)
    began = cursor['step']
    last_save = time.monotonic()
    limit = training['target_updates'] if stop_after is None else stop_after
    while cursor['step'] < limit:
        idx = draw(rng, len(embeddings), recipe['target_fit_origin_batch'])
        hidden, estimate, label = (tensor[idx].to(device) for tensor in (embeddings, base, truth))
        loss = (head(hidden).squeeze(-1)+estimate-label).abs().mean()
        update(loss, optimizer, training['gradient_clip_norm'])
        cursor['step'] += 1
        cursor['losses'].append(float(loss.detach()))
        if checkpoint is not None and (cursor['step'] in (limit, training['target_updates'])
                or cursor['step'] % 50 == 0 or time.monotonic()-last_save >= recipe['checkpoint_seconds']):
            save(checkpoint, identity, head, optimizer, rng, cursor)
            last_save = time.monotonic()
            progress(phase='SUPPORT_ADAPTATION', optimizer_step=cursor['step'], expected_steps=limit)
    head.requires_grad_(False)
    return head.eval(), {**cursor, 'new_steps': cursor['step']-began}


def source_fit(model, raw, task, bank, recipe, training, local_recipes, device, directory,
               identity, progress, stop_after=None):
    model.eval()  # deterministic, dropout disabled while training with gradients
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=training['source_learning_rate'], weight_decay=training['weight_decay'])
    rng = generator(task['seed'])
    initial = {'step': 0, 'epoch': 0, 'next_domain': 0, 'stale': 0, 'best': None,
               'best_model': None, 'history': [], 'losses': []}
    cursor = resume(directory, identity, model, optimizer, rng, initial)
    initial_step = cursor['step']
    last_save = time.monotonic()
    family, seed = task['family'], task['seed']
    microbatch = recipe['origin_microbatch']

    def outer(slug, episode, mode):
        values = raw[slug]
        a, b = episode['support']
        scaler = fit_scaler(values, [a, b])
        scale = support_scale(values, [a, b], seed, *local_recipes, device,
                              directory/'scales'/slug/(str(a)+'.json'))
        residual, _ = adapt(model, values, [a+12, b], scaler, seed, recipe, training, device, progress)
        start, end = episode['suffix']
        total = 0.
        for ids in batches([start, end], microbatch):
            x, _ = data_batch(values, ids, scaler, device)
            with torch.set_grad_enabled(mode == 'train'):
                forecast = predict_encoded(model, x, residual)
                forecast = (scaler['minimum']+forecast*(scaler['maximum']-scaler['minimum'])).clamp_min(0)
                truth = torch.as_tensor(values[:, ids].T.copy(), dtype=torch.float32, device=device)
                loss = (forecast-truth).abs().mean()*(len(ids)/(end-start))/scale
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite source suffix loss')
                if mode == 'train':
                    loss.backward()
                total += float(loss.detach())
        return total

    domains = task['sources']
    while cursor['epoch'] < training['source_max_epochs'] and cursor['stale'] < training['source_patience']:
        for i in range(cursor['next_domain'], len(domains)):
            slug = domains[i]
            episodes = bank[slug]['train']
            idx = int(draw(rng, len(episodes), 1)[0])
            loss = outer(slug, episodes[idx], 'train')
            step(optimizer, training['gradient_clip_norm'])
            cursor['step'] += 1
            cursor['next_domain'] = i+1
            cursor['losses'].append([cursor['epoch'], slug, idx, loss])
            if cursor['step'] == stop_after or time.monotonic()-last_save >= recipe['checkpoint_seconds']:
                save(directory, identity, model, optimizer, rng, cursor)
                last_save = time.monotonic()
            progress(phase='SOURCE_TRAINING', family=family, epoch=cursor['epoch']+1,
                     optimizer_step=cursor['step'], source_domain=slug, suffix_loss=loss)
            if cursor['step'] == stop_after:
                return cursor, cursor['step']-initial_step
        values = {slug: outer(slug, bank[slug]['validation'], 'validation') for slug in domains}
        metric = sum(values.values())/len(values)
        cursor['epoch'] += 1
        cursor['next_domain'] = 0
        cursor['history'].append({'epoch': cursor['epoch'], 'validation_normalized_MAE': metric,
                                  'validation_by_source': values})
        if cursor['best'] is None or metric < cursor['best']:
            cursor['best'] = metric
            cursor['best_model'] = model_state(model)
            cursor['stale'] = 0
        else:
            cursor['stale'] += 1
        save(directory, identity, model, optimizer, rng, cursor)
        last_save = time.monotonic()
    return cursor, cursor['step']-initial_step


@torch.no_grad()
def forecast(model, raw, bounds, scaler, recipe, device, residual):
    values = []
    for ids in batches(bounds, recipe['prediction_origin_batch']):
        x, _ = data_batch(raw, ids, scaler, device)
        values.append(predict_encoded(model, x, residual).cpu().numpy())
    z = np.concatenate(values)
    y = inverse(z, scaler)
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite frozen forecast')
    return y


def invariant_probe(model, raw, support, family, device, residual=None):
    a, b = support
    scaler = fit_scaler(raw, support)
    x, _ = data_batch(raw, [a+12, a+13], scaler, device)
    return check_invariance(model, x, family, residual)
