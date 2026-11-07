"""Development-only source loss comparison with the archived update schedule."""
import time
import torch
from compact_training import data_batch
from transfer_train import (adapt, draw, generator, model_state, predict_encoded,
                            resume, save, step, support_scale)
from ectt_data import fit_scaler
from shared_baselines import batches


def training_forecast(physical, arm, mode):
    if arm == 'unclipped' and mode == 'train':
        return physical
    return physical.clamp_min(0)


def fit(model, raw, task, bank, recipe, training, local_recipes, device, directory,
               identity, progress, arm, stop_after=None):
    if arm not in ('clipped', 'unclipped'):
        raise ValueError('Unknown source loss arm')
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
    recurrent = [module for module in model.modules() if isinstance(module, torch.nn.GRU)]

    def outer(slug, episode, mode):
        values = raw[slug]
        a, b = episode['support']
        scaler = fit_scaler(values, [a, b])
        scale = support_scale(values, [a, b], seed, *local_recipes, device,
                              directory/'scales'/slug/(str(a)+'.json'))
        residual, _ = adapt(model, values, [a+12, b], scaler, seed, recipe, training, device, progress)
        # cuDNN needs reserve space from a training-mode GRU forward for
        # backward. The released GRUs have no dropout. Leave GPT-2 in eval.
        if mode == 'train':
            for layer in recurrent:
                layer.train()
        start, end = episode['suffix']
        total = 0.
        for ids in batches([start, end], microbatch):
            x, _ = data_batch(values, ids, scaler, device)
            with torch.set_grad_enabled(mode == 'train'):
                forecast = predict_encoded(model, x, residual)
                physical = scaler['minimum']+forecast*(scaler['maximum']-scaler['minimum'])
                forecast = training_forecast(physical, arm, mode)
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
