"""One read-only CPU campaign on frozen traffic forecasts, internally reviewed."""
import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path
import sys
import tarfile
from statistics import mean

import numpy as np
from traffic_blend import fit_weight, forecast

ORIGINAL = 'a99dae5b0644ac5baa7f9c7f3fe29b1ea21631c7'
PACKET = '00682af92957218b95346ed6e509faea661afc64b404716ecd67ef580af6f19f'
INVENTORY = '35256a28a0cb7f15834b1e1297824c6642b710bd1d7785979072aaf76ed3557c'
PRIMARY = 'locked_two_sided'
MARGIN = 'verification_margin'
PREFIX = 'prefix_mae'
RULE_2_3 = 'fixed_two_thirds_mean'
MODE = 'Internally reviewed by Codex under the 23 November amendment; independent S01 disposition unavailable.'


def table(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def local_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def cases(project, code, contract):
    import heldout_score as old
    import transfer_diagnostics as diag
    import transfer_plan as plan

    manifest, inventory, _, _ = old.review_inputs(project, code)
    if manifest['review_identity'] != PACKET or manifest['inventory_sha256'] != INVENTORY:
        raise ValueError('The internal review was for a different packet')
    result = []
    for record in inventory:
        meta = old.checked_case(record, manifest['candidate_identity'], contract)
        task = meta['task']
        folder = Path(record['case'])
        result.append((task['dataset'], task['seed'], task['arm'], meta,
                       folder/'verification_candidates.npz', folder/'test_candidates.npz', False))
    matrix, _ = diag.foundation(project, code)
    store = plan.store_path(project, matrix)
    for task in matrix['target']:
        domain = task['dataset']
        if domain not in contract['confirmatory_targets']:
            continue
        folder, artifacts = diag.worker_artifacts(project, task, store, 'target', plan.TARGET)
        checked = lambda file: diag.checked(folder, artifacts, '0.05/'+file)
        meta = old.read(checked('frozen_candidates.json'))
        profile = contract['datasets'][domain]
        budget = profile['budgets']['0.05']
        if (meta['verification_targets'] != budget['verification_target_rows'] or
                meta['support_raw'] != budget['support_raw'] or meta['test_targets'] != profile['test_targets'] or
                meta['verification_labels_used'] != 0 or meta['test_labels_used_for_fitting'] != 0 or
                meta['fallback'] or not np.isfinite(meta['support_scale']) or meta['support_scale'] < 1):
            raise ValueError('Ordinary baseline protocol differs')
        result.append((domain, task['seed'], task['family'], meta, checked('verification_candidates.npz'),
                       checked('test_forecasts.npz'), True))
    expected = set(itertools.product(contract['confirmatory_targets'], contract['seeds'],
                                     ['suffix_mae', PREFIX, MARGIN, 'B03', 'B04']))
    if len(result) != 90 or set(row[:3] for row in result) != expected:
        raise ValueError('Incomplete archived comparison')
    return result


def choose(case, raw, contract):
    import heldout_score as old
    from heldout_thresholds import replay
    domain, seed, arm, meta, path, _, ordinary = case
    start, stop = meta['verification_targets']
    prefix = old.candidate_arrays(path, [start, stop], raw.shape[0])
    n = old.candidates.two_thirds_rows(stop-start)
    truth = raw[:, start:start+n].T.astype('float64')
    local, transfer = prefix['local'], prefix['transfer']
    weights = {'local': [0.,0], 'transfer': [1.,0], 'half': [.5,0],
               'prefix_blend': [fit_weight(local[:n], transfer[:n], truth, meta['support_scale']), n]}
    original, possible_at = {}, None
    if not ordinary:
        original = old.decisions_for(meta, prefix, raw, contract)
        for rule, row in original.items():
            weights[rule] = [float(row['accepted']), row['verification_origins_read']]
        accept_full = original['fixed_budget_mean']['accepted']
        weights['half_after_full_switch'] = [.5 if accept_full else 0.,stop-start]
        # Conditional on the realized sequence of forecast pairs, each row's
        # most favorable loss difference is attained by the hypothetical truth T.
        settings = dict(rho=contract['verifier']['rho'], delta=contract['verifier']['delta_per_stream'],
                        eta=contract['training_design']['mechanism_coefficients']['epsilon'],
                        minimum=contract['partitions']['minimum_verification_origins'])
        best = replay(PRIMARY, np.arange(start,stop), local, transfer, meta['support_scale'],
                      lambda origin: transfer[origin-start], settings, stop-start)
        possible_at = best['verification_origins_read'] if best['accepted'] else None
    return {'weights':weights, 'original':original, 'possible_at':possible_at,
            'local_signature':local_hash(local), 'support_scale':meta['support_scale']}


def reference_forecasts(project, domain, origins, truth):
    from ectt_common import digest, inside, read
    from reference_datasets import PROFILES
    path = project/'results/slurm_jobs/607564/report/datasets'/PROFILES[domain]['slug']/'inference_result.json'
    result = read(path)
    store = inside(project, Path(result['output_store']))
    if (read(store/'completed.json') != result or result.get('dataset') != domain or
            result['training_performed'] or result['strict_load_performed'] is not True or
            digest(store/'forecasts.npz') != result['forecast_file_sha256']):
        raise ValueError('Archived faithful reference failed verification: '+domain)
    with np.load(store/'forecasts.npz', allow_pickle=False) as z:
        ids = z['target_indices']
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(np.diff(ids)<=0):
            raise ValueError('Invalid faithful origin ledger')
        _, i, j = np.intersect1d(origins, ids, assume_unique=True, return_indices=True)
        if len(i) == 0:
            return i, None, len(ids), 'no_common_origins'
        if not np.array_equal(truth[i], z['truth'][j]):
            return i, None, len(ids), 'unequal_physical_truth'
        pred = z['predictions'][j].astype('float64')
        if pred.shape != truth[i].shape or not np.isfinite(pred).all():
            raise ValueError('Invalid faithful predictions')
    return i, pred, len(ids), 'comparable_common_rows'


def contrast(errors, domains, seeds, left, right, contract):
    from transfer_parallel import block_means
    points, panels = {}, {}
    for d in domains:
        panel = np.mean([np.column_stack([errors[d,s,k] for k in ('local',left,right)]) for s in seeds], axis=0)
        panels[d] = panel
        l,a,b = panel.mean(axis=0)
        points[d] = float((a-b)/l) if l else None
    macro = mean(points.values()) if all(v is not None for v in points.values()) else None
    bands = {}
    for length in (72,144,288):
        draws = []
        for i,d in enumerate(domains):
            fold = domains.index('bike') if d in ('bike','taxi') else i
            rng = np.random.default_rng([20261111,length,fold])
            vals = block_means(panels[d],length,contract['statistics']['paired_bootstrap_replicates'],rng)
            draws.append(np.divide(vals[:,1]-vals[:,2],vals[:,0],out=np.full(len(vals),np.nan),where=vals[:,0]>0))
        sample = np.mean(draws,axis=0)
        bands[str(length)] = np.quantile(sample,[.025,.975]).tolist() if np.isfinite(sample).all() else None
    return {'domain_effects':points,'equal_domain_effect':macro,'descriptive_95_intervals':bands,
            'equal_domain_effect_excluding_nyc':(mean(points[d] for d in domains if d not in ('bike','taxi'))
                           if all(points[d] is not None for d in domains if d not in ('bike','taxi')) else None),
            'full_144_row_blocks':{d:len(panels[d])//144 for d in domains}}


def score_panel(project, catalog, choices, raw, contract, out):
    import heldout_score as old
    import transfer_diagnostics as diag
    from shared_baselines import independent_scores
    original_rows, original_errors, rows, errors, overlap, ceiling = [], {}, [], {}, [], []
    refs, local_signatures = {}, {}
    for case in catalog:
        d,s,a,meta,_,path,ordinary = case
        bounds = meta['test_targets']
        truth = raw[d][:,slice(*bounds)].T.astype('float64')
        f = (diag.forecasts(path,bounds,truth.shape[1],include_truth=True) if ordinary
             else old.candidate_arrays(path,bounds,truth.shape[1]))
        if ordinary and not np.array_equal(f.pop('truth'),truth):
            raise ValueError('Historical target truth differs')
        sig = local_hash(f['local'])
        if local_signatures.setdefault((d,s),sig) != sig:
            raise ValueError('Test comparator differs between arms')
        decision = choices['/'.join((d,str(s),a))]
        local_mae = np.abs(f['local']-truth).mean(axis=1)
        errors[d,s,'local'] = local_mae
        if not ordinary:
            rr, cc = old.test_rows(meta,f,truth,decision['original'],contract)
            original_rows.extend(rr)
            original_errors[d,s,a] = cc
            ceiling.append(dict(dataset=d,seed=s,arm=a,possible_row=decision['possible_at'],
                         accepted=decision['original'][PRIMARY]['accepted'],
                         always_transfer_better=bool(np.abs(f['transfer']-truth).mean()<local_mae.mean())))
        if d not in refs:
            try:
                refs[d] = reference_forecasts(project,d,np.arange(*bounds),truth)
            except (OSError, ValueError, KeyError) as exc:
                # Archive availability cannot determine the primary experiment.
                refs[d] = (np.array([],dtype=int),None,None,'archive_check_failed: '+type(exc).__name__)
        idx, ref, ref_count, status = refs[d]
        for rule,(weight,nrows) in decision['weights'].items():
            pred = forecast(f['local'],f['transfer'],weight)
            err = np.abs(pred-truth).mean(axis=1)
            difference = err-local_mae
            scale = meta['support_scale']
            key = a+'/'+rule
            errors[d,s,key] = err
            rows.append(dict(dataset=d,seed=s,arm=a,rule=rule,weight=weight,
                       support_labels=meta['unique_support_values'],verification_rows=nrows,
                       verification_labels=nrows*truth.shape[1],test_rows=len(truth),
                       mean_excess_MAE=float(difference.mean()),material_harm_fraction=float((difference/scale>.01).mean()),
                       **independent_scores(pred,truth)))
            if a==MARGIN and rule in (PRIMARY,'prefix_blend','local'):
                item = dict(dataset=d,seed=s,policy=rule,status=status,common_rows=len(idx),
                            total_new_rows=len(truth),total_reference_rows=ref_count)
                ref_scores = independent_scores(ref,truth[idx]) if ref is not None else {}
                new_scores = independent_scores(pred[idx],truth[idx]) if ref is not None else {}
                for k in ('MAE','RMSE','WMAPE_percent'):
                    item['ours_'+k],item['release_'+k] = new_scores.get(k),ref_scores.get(k)
                overlap.append(item)
    orig = old.summarize(original_rows,original_errors,contract)
    old.write(out/'registered_comparison.json',orig)
    old.write_summary(out/'registered_results.md',orig,54)
    table(out/'registered_metrics.csv',original_rows)
    table(out/'campaign_metrics.csv',rows)
    table(out/'release_checkpoint_overlap.csv',overlap)
    table(out/'certificate_feasibility.csv',ceiling)
    left = MARGIN+'/prefix_blend'
    controls = ['local',MARGIN+'/half',MARGIN+'/half_after_full_switch',MARGIN+'/'+RULE_2_3,PREFIX+'/prefix_blend','suffix_mae/prefix_blend',
                'B03/prefix_blend','B04/prefix_blend','B03/transfer','B04/transfer']
    comparisons = {k:contrast(errors,contract['confirmatory_targets'],contract['seeds'],left,k,contract) for k in controls}
    old.write(out/'blend_comparison.json',comparisons)
    return rows, orig, comparisons, ceiling, overlap


def run(project, code, out):
    import heldout_score as old
    from ectt_common import digest, finish, load_contract, write
    from transfer_worker import raw_arrays, raw_paths
    from transfer_plan import fingerprint

    contract,_ = load_contract(code)
    catalog = cases(project,code,contract)
    raw = raw_arrays(raw_paths(project,code,contract),contract['confirmatory_targets'],contract)
    decisions, signatures = {}, {}
    for case in catalog:
        d,s,a = case[:3]
        decision = choose(case,raw[d],contract)
        signature = (decision['local_signature'],decision['support_scale'])
        if signatures.setdefault((d,s),signature)!=signature:
            raise ValueError('Unequal verification comparator or support scale')
        decisions['/'.join((d,str(s),a))] = decision
    sealed = out/'verification_decisions.json'
    if sealed.exists():
        if old.read(sealed) != decisions:
            raise ValueError('Previously sealed decisions differ')
    else:
        write(sealed,decisions)
    seal = digest(sealed)
    rows, registered, comps, ceiling, overlap = score_panel(project,catalog,old.read(sealed),raw,contract,out)
    if digest(sealed)!=seal:
        raise ValueError('Sealed decisions changed')
    effect = registered['contrasts'][MARGIN+'/'+PRIMARY+' vs '+PREFIX+'/'+PRIMARY]
    manifest = {'review_mode':MODE,'original_snapshot':ORIGINAL,'original_review_identity':PACKET,
                'frozen_inventory_sha256':INVENTORY,'campaign_snapshot':Path(__file__).resolve().parents[1].name,
                'slurm_job':os.environ.get('SLURM_JOB_ID'),'verified_cases':len(catalog),'registered_policy_rows':216,
                'all_policy_rows':len(rows),'decision_sha256':seal,'numpy':np.__version__,'new_training_updates':0}
    write(out/'campaign_receipt.json',manifest)
    def pct(x):
        return 'undefined' if x is None else f'{100*x:+.2f}'
    lines = ['# Traffic forecasting campaign results','',MODE,'',
             f"Registered margin minus prefix effect was {pct(effect['relative_MAE'])} percentage points of local MAE.",
             'Negative values favour the first method. Descriptive paired intervals condition on these datasets.', '',
             '| Quarter-grid margin blend versus control | Difference in percentage points of local MAE | 144-row interval (percentage points) |',
             '|---|---:|---|']
    for k,v in comps.items():
        band = v['descriptive_95_intervals']['144']
        label = ', '.join(pct(x) for x in band) if band else 'undefined'
        lines.append(f"| {k} | {pct(v['equal_domain_effect'])} | {label} |")
    wins = [r for r in ceiling if r['arm']==MARGIN and r['always_transfer_better']]
    impossible = sum(r['possible_row'] is None for r in wins)
    lines += ['',f'{len(wins)} of 18 margin candidates improved unconditional test MAE. In {impossible} of those cases, the locked boundary was unreachable even under the most favorable labels along the realized prediction path.',
              '', 'Checkpoint overlap is descriptive. Coverage, unmatched cohorts, reference metrics and our metrics are in release_checkpoint_overlap.csv.',
              '', 'Unconditional and adaptive baselines, harm and label use are in campaign_metrics.csv.']
    lines += ['', '| Domain | Local MAE | Locked margin MAE | Weighted margin MAE | Weighted prefix control MAE | Verification rows used by weighted margin |',
              '|---|---:|---:|---:|---:|---:|']
    for d in contract['confirmatory_targets']:
        def group(arm,rule,key):
            values = [r[key] for r in rows if (r['dataset'],r['arm'],r['rule']) == (d,arm,rule)]
            return mean(values)
        lines.append(f"| {d} | {group(MARGIN,'local','MAE'):.3f} | {group(MARGIN,PRIMARY,'MAE'):.3f} | {group(MARGIN,'prefix_blend','MAE'):.3f} | {group(PREFIX,'prefix_blend','MAE'):.3f} | {group(MARGIN,'prefix_blend','verification_rows'):.0f} |")
    lines += ['', '| Domain | Common rows / our rows / release rows | Weighted margin MAE | Released checkpoint MAE | Cohort check |',
              '|---|---|---:|---:|---|']
    for d in contract['confirmatory_targets']:
        items = [r for r in overlap if r['dataset']==d and r['policy']=='prefix_blend']
        first = items[0]
        values = [format(mean(r[k] for r in items),'.3f') if first[k] is not None else '—'
                  for k in ('ours_MAE','release_MAE')]
        lines.append(f"| {d} | {first['common_rows']} / {first['total_new_rows']} / {first['total_reference_rows']} | {values[0]} | {values[1]} | {first['status']} |")
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    templates = Path(__file__).resolve().parents[1]/'paper'/'traffic_transfer_draft.md'
    draft = templates.read_text().replace('<!-- CAMPAIGN_RESULTS -->','\n'.join(lines[4:]))
    draft = draft.replace('The six-domain estimates are required before drawing conclusions about generalization.',
        'Across six held-out domains, the primary margin-minus-prefix contrast was '+pct(effect['relative_MAE'])+
        ' percentage points of local MAE under the locked two-sided rule. The secondary weighted margin forecast differed from the local comparator by '+
        pct(comps['local']['equal_domain_effect'])+' percentage points of local MAE. Negative values indicate lower error.')
    draft = draft.replace('The Results section below the insertion marker is populated by the pinned CPU campaign.',
                          'Held-out tables were rendered from the completed internal CPU evaluation.')
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    shutil.copyfile(templates.parent/'figures/development_windows.svg',figures/'development_windows.svg')
    (out/'manuscript_with_results.md').write_text(draft)
    finish(out, fingerprint(manifest))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True)
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID','').isdigit():
        p.error('Use the CPU batch launcher')
    project = args.project.resolve()
    code = project/'work/code_versions'/ORIGINAL
    sys.path.insert(0,str(code/'scripts'))
    from ectt_common import completed, read
    src = Path(__file__).resolve().parents[1]
    out = project/'results/traffic_publication'/src.name
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'COMPLETE.json').exists():
            completed(out,read(out/'COMPLETE.json')['identity'])
        else:
            run(project,code,out)
        bundle = out.parent/(src.name+'_report.tar.gz')
        temp = bundle.with_suffix('.tmp')
        with tarfile.open(temp,'w:gz') as archive:
            for name in ('SUMMARY.md','registered_results.md','registered_comparison.json','registered_metrics.csv',
                         'campaign_metrics.csv','release_checkpoint_overlap.csv','certificate_feasibility.csv',
                         'blend_comparison.json','campaign_receipt.json','manuscript_with_results.md','figures/development_windows.svg','COMPLETE.json'):
                archive.add(out/name,arcname=name)
        temp.replace(bundle)
        print((out/'SUMMARY.md').read_text(),flush=True)
        print('Share this aggregate bundle:',bundle,flush=True)


if __name__=='__main__':
    main()
