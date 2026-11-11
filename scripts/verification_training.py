"""Source-only objectives for the matched verification-aware development pilot."""
import time

import numpy as np
import torch
from torch.nn import functional as F

import shared_baselines as local
from compact_training import data_batch, fit as compact_fit, predict as compact_predict
from ectt_common import digest, read, write
from ectt_data import fit_scaler
from transfer_plan import fingerprint
from transfer_train import adapt, draw, generator, model_state, predict_encoded, resume, save, step
from transfer_worker import save_arrays


ARMS = ('suffix_mae', 'prefix_mae', 'verification_margin')


def prefix_objective(physical, reference, truth, scale, arm, coefficients, verifier):
    """Use the same physical clipping and source labels in both prefix objectives."""
    prediction = physical.double().clamp_min(0)
    error = (prediction-truth).abs().mean(dim=1)/scale
    sizes = coefficients['prefixes']
    if arm == 'prefix_mae':
        return torch.stack([error[:n].mean() for n in sizes]).mean()
    if arm != 'verification_margin':
        raise ValueError('Unknown prefix training objective')
    difference = error-(reference-truth).abs().mean(dim=1)/scale
    envelope = (prediction-reference).abs().mean(dim=1)/scale
    rho, delta = verifier['rho'], verifier['delta_per_stream']
    temperature = coefficients['smooth_hinge_temperature']
    terms = []
    for n in sizes:
        variance = envelope[:n].square().sum()
        width = torch.sqrt((variance+rho)*(torch.log1p(variance/rho)-2*np.log(delta)))/n
        upper = difference[:n].mean()+width
        terms.append(temperature*F.softplus((upper+coefficients['epsilon'])/temperature))
    return torch.stack(terms).mean()


def local_reference(raw, episode, seed, recipes, device, path, identity, with_prefix):
    """Select and refit the frozen local comparator inside source support."""
    ridge_recipe, compact_recipe = recipes
    a, b = episode['support']
    split = a+int(ridge_recipe['selection_support_fraction']*(b-a))
    key = fingerprint({'fit': identity, 'episode': episode, 'prefix': with_prefix})
    receipt = path/'selection.json'
    reference = path/'prefix.npz'
    if receipt.exists():
        selection = read(receipt)
        if selection['identity'] != key:
            raise ValueError('Source comparator cache identity differs')
        if with_prefix:
            if digest(reference) != selection['prefix_sha256']:
                raise ValueError('Source comparator prefix changed')
            with np.load(reference, allow_pickle=False) as values:
                return selection['scale'], values['prediction']
        return selection['scale'], None

    profile = {'budgets': {'1.0': {'eligible_for_learned_adaptation': True,
                                  'support_raw': [a, b], 'support_target_rows': [a+12, b]}}}
    fit = local.fit_case(raw, profile, '1.0', ridge_recipe)
    candidates = {family: min(row['validation_MAE'] for row in fit['candidates'] if row['family'] == family)
                  for family in ('persistence', 'ridge')}
    inner_scaler = fit_scaler(raw, [a, split])
    truth = raw[:, split:b].T
    compact_scores = []
    devices = [torch.cuda.current_device()] if device == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        for index, architecture in enumerate(compact_recipe['configurations']):
            model, _ = compact_fit(raw, [a+12, split], inner_scaler, architecture, compact_recipe, seed,
                                   path/'inner'/str(index), key, device, lambda **_: None)
            prediction = compact_predict(model, raw, [split, b], inner_scaler,
                                         compact_recipe['prediction_origin_batch'], device)
            compact_scores.append(local.scores(prediction, truth)['MAE'])
        best = min(range(len(compact_scores)), key=compact_scores.__getitem__)
        candidates['compact'] = compact_scores[best]
        family = min(compact_recipe['selection_order'], key=candidates.__getitem__)
        scale = max(1., candidates[family])
        prediction = None
        if with_prefix and family == 'compact':
            scaler = fit_scaler(raw, [a, b])
            model, _ = compact_fit(raw, [a+12, b], scaler, compact_recipe['configurations'][best],
                                   compact_recipe, seed, path/'refit', key, device, lambda **_: None)
            prediction = compact_predict(model, raw, episode['verification'], scaler,
                                         compact_recipe['prediction_origin_batch'], device)
        elif with_prefix:
            prediction = local.predict(raw, episode['verification'], fit['models'][family],
                                       ridge_recipe['origin_batch'])
    if with_prefix:
        save_arrays(reference, prediction=prediction)
    write(receipt, {'identity': key, 'support': [a, b], 'inner_validation': [split, b],
                    'family': family, 'scale': scale, 'validation_MAE': candidates,
                    'best_compact_configuration': best,
                    'prefix_sha256': digest(reference) if with_prefix else None})
    return scale, prediction


def physical_forecast(model, residual, raw, ids, scaler, device):
    x, _ = data_batch(raw, ids, scaler, device)
    forecast = predict_encoded(model, x, residual)
    return scaler['minimum']+forecast*(scaler['maximum']-scaler['minimum'])


def prefix_backward(predict, batches, reference, read_truth, scale, arm, coefficients, verifier):
    """Differentiate all prefixes jointly, then replay each bounded GPU batch."""
    # Detach each batch immediately while keeping the same forward kernel mode
    # as the gradient replay. Some kernels change paths under no_grad.
    first = [predict(ids).detach() for ids in batches]
    truth = read_truth()
    leaf = torch.cat(first).detach().requires_grad_(True)
    if leaf.shape != truth.shape or reference.shape != truth.shape or max(coefficients['prefixes']) != len(leaf):
        raise ValueError('Source prefix forecast and label cohorts differ')
    penalty = coefficients['verification_margin_weight']*prefix_objective(
        leaf, reference, truth, scale, arm, coefficients, verifier)
    gradient, = torch.autograd.grad(penalty, leaf)
    if not torch.isfinite(penalty) or not torch.isfinite(gradient).all():
        raise ValueError('Nonfinite source prefix objective')
    offset = 0
    for ids, frozen in zip(batches, first):
        value = predict(ids)
        if not torch.equal(value.detach(), frozen):
            raise ValueError('Source prefix forward replay differs')
        value.backward(gradient[offset:offset+len(ids)])
        offset += len(ids)
    return float(penalty.detach())


def fit(model, raw, task, bank, recipe, training, verifier, recipes, device, directory,
        identity, progress, stop_after=None):
    arm = task['arm']
    if arm not in ARMS or set(raw) != set(task['sources']) or task['dataset'] in raw:
        raise ValueError('Invalid objective or held-out source fold')
    model.eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=training['source_learning_rate'], weight_decay=training['weight_decay'])
    rng = generator(task['seed'])
    cursor = resume(directory, identity, model, optimizer, rng,
                    {'step': 0, 'epoch': 0, 'next_domain': 0, 'stale': 0, 'best': None,
                     'best_model': None, 'history': [], 'losses': []})
    began = cursor['step']
    last_save = time.monotonic()
    recurrent = [module for module in model.modules() if isinstance(module, torch.nn.GRU)]
    coefficients = training['mechanism_coefficients']

    def outer(slug, episode, training_step):
        raw_domain = raw[slug]
        a, b = episode['support']
        prefix = training_step and arm != 'suffix_mae'
        name = 'train' if training_step else 'validation'
        scale, reference = local_reference(raw_domain, episode, task['seed'], recipes, device,
                                           directory/'local'/slug/name/str(a), identity, prefix)
        scaler = fit_scaler(raw_domain, [a, b])
        residual, _ = adapt(model, raw_domain, [a+12, b], scaler, task['seed'], recipe,
                             training, device, progress)
        if training_step:
            for layer in recurrent:
                layer.train()

        def predict(ids):
            return physical_forecast(model, residual, raw_domain, ids, scaler, device)

        start, end = episode['suffix']
        suffix_loss = 0.
        for ids in local.batches([start, end], recipe['origin_microbatch']):
            with torch.set_grad_enabled(training_step):
                value = predict(ids)
                truth = torch.as_tensor(raw_domain[:, ids].T.copy(), dtype=torch.float32, device=device)
                loss = ((value if training_step else value.clamp_min(0))-truth).abs().mean()
                loss = loss*(len(ids)/(end-start))/scale
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite source suffix MAE')
                if training_step:
                    loss.backward()
                suffix_loss += float(loss.detach())
        prefix_loss = 0.
        if prefix:
            start, end = episode['verification']

            def read_labels():
                return torch.as_tensor(raw_domain[:, start:end].T.copy(), dtype=torch.float64, device=device)

            prefix_loss = prefix_backward(predict, list(local.batches([start, end], recipe['origin_microbatch'])),
                                          torch.as_tensor(reference, dtype=torch.float64, device=device),
                                          read_labels, scale, arm, coefficients, verifier)
        return suffix_loss, prefix_loss

    domains = task['sources']
    while cursor['epoch'] < training['source_max_epochs'] and cursor['stale'] < training['source_patience']:
        for i in range(cursor['next_domain'], len(domains)):
            slug = domains[i]
            episodes = bank[slug]['train']
            idx = int(draw(rng, len(episodes), 1)[0])
            suffix, prefix = outer(slug, episodes[idx], True)
            step(optimizer, training['gradient_clip_norm'])
            cursor['step'] += 1
            cursor['next_domain'] = i+1
            cursor['losses'].append([cursor['epoch'], slug, idx, suffix, prefix])
            if cursor['step'] == stop_after or time.monotonic()-last_save >= recipe['checkpoint_seconds']:
                save(directory, identity, model, optimizer, rng, cursor)
                last_save = time.monotonic()
            progress(phase='SOURCE_TRAINING', optimizer_step=cursor['step'], source_domain=slug,
                     epoch=cursor['epoch']+1, suffix_loss=suffix, prefix_loss=prefix)
            if cursor['step'] == stop_after:
                return cursor, cursor['step']-began
        values = {slug: outer(slug, bank[slug]['validation'], False)[0] for slug in domains}
        metric = sum(values.values())/len(values)
        cursor['epoch'] += 1
        cursor['next_domain'] = 0
        cursor['history'].append({'epoch': cursor['epoch'], 'validation_normalized_MAE': metric,
                                  'validation_by_source': values})
        if cursor['best'] is None or metric < cursor['best']:
            cursor.update(best=metric, best_model=model_state(model), stale=0)
        else:
            cursor['stale'] += 1
        save(directory, identity, model, optimizer, rng, cursor)
        last_save = time.monotonic()
    return cursor, cursor['step']-began
