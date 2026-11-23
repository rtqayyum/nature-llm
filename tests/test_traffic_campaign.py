"""Exercise the amendment separately from the original held-out admission."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'scripts'))
sys.path.insert(0,str(CODE/'tests'))
from traffic_blend import fit_weight
import traffic_campaign as runner
import test_heldout_score as previous_tests
import heldout_score as old
from ectt_common import digest, read, write
import transfer_worker


class AmendmentTests(unittest.TestCase):
    def test_cost_ties_favor_local_and_interior_solution(self):
        local=np.zeros((24,2))
        transfer=np.ones_like(local)*4
        self.assertEqual(fit_weight(local,transfer,transfer/2,1.),.5)
        self.assertEqual(fit_weight(local,local,local,1.),0.)
        # Every positive weight exactly ties local after the 0.01*w penalty.
        self.assertEqual(fit_weight(local,transfer,transfer,400.),0.)

    def test_truth_and_overlap(self):
        root=tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        project=Path(root.name)
        store=project/'store'
        store.mkdir()
        truth=np.arange(8.).reshape(4,2)
        ids=np.arange(20,24)
        filename=store/'forecasts.npz'
        np.savez(filename,target_indices=ids[1:3],truth=truth[1:3],predictions=truth[1:3]+1)
        result=dict(dataset='pems03',output_store=str(store),forecast_file_sha256=digest(filename),
                    training_performed=False,strict_load_performed=True)
        from reference_datasets import PROFILES
        write(store/'completed.json',result)
        write(project/'results/slurm_jobs/607564/report/datasets'/PROFILES['pems03']['slug']/'inference_result.json',result)
        evaluate=lambda y:runner.reference_forecasts(project,'pems03',ids,y)
        self.assertEqual(evaluate(truth)[-1],'comparable_common_rows')
        self.assertEqual(evaluate(truth+.5)[-1],'unequal_physical_truth')

    def test_full_cpu_path_preserves_primary_and_seals_before_scoring(self):
        fixture = previous_tests.HeldoutScoringTests('test_complete_synthetic_scoring_and_retry')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        contract=fixture.contract
        manifest,records,_,_=old.review_inputs(fixture.project,old_code())
        cases=[]
        for rec in records:
            meta=old.checked_case(rec,manifest['candidate_identity'],contract)
            folder=Path(rec['case'])
            t=meta['task']
            cases.append((t['dataset'],t['seed'],t['arm'],meta,folder/'verification_candidates.npz',folder/'test_candidates.npz',False))
        ordinary=[]
        for d,s,a,m,v,t,_ in cases:
            if a!='suffix_mae':
                continue
            with np.load(t) as z:
                tr=t.parent/'ordinary.npz'
                np.savez(tr,origins=z['origins'],local=z['local'],transfer=z['transfer'],truth=np.ones_like(z['local']))
            ordinary.extend((d,s,f,m,v,tr,True) for f in ('B03','B04'))
        out=fixture.report
        original_score=runner.score_panel
        def check_then_score(*args):
            self.assertEqual(len(read(out/'verification_decisions.json')),90)
            return original_score(*args)
        def refs(project,domain,ids,truth):
            return np.arange(len(ids)),truth,len(ids),'comparable_common_rows'
        with patch.object(runner,'cases',return_value=cases+ordinary),patch.object(runner,'reference_forecasts',side_effect=refs), \
             patch('ectt_common.load_contract',return_value=(contract,{})),patch.object(transfer_worker,'raw_paths',return_value=fixture.paths), \
             patch.object(runner,'score_panel',side_effect=check_then_score):
            runner.run(fixture.project,old_code(),out)
        self.assertEqual(read(out/'campaign_receipt.json')['registered_policy_rows'],216)
        with (out/'campaign_metrics.csv').open() as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),54*9+36*4)
        self.assertTrue((out/'manuscript_with_results.md').exists())
        effect=read(out/'registered_comparison.json')['contrasts']['verification_margin/locked_two_sided vs prefix_mae/locked_two_sided']
        self.assertEqual(effect['relative_MAE'],-1.)
        self.assertNotIn('APPROVED',json.dumps(read(out/'campaign_receipt.json')))


def old_code():
    import heldout_score
    return Path(heldout_score.__file__).resolve().parents[1]
