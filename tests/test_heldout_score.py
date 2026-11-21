"""Exercise the approval boundary and all 216 scored rows with synthetic traffic."""
import csv
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE/'scripts'))
from ectt_common import digest, finish, load_contract, read, write
import heldout_score as scoring
import transfer_worker


class HeldoutScoringTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.project = project = Path(temp.name)
        self.plan = plan = scoring.candidates.matrix(CODE)
        self.contract, _ = load_contract(CODE)
        self.contract['statistics']['paired_bootstrap_replicates'] = 16
        identity = plan['identity']
        sources = project/'inputs'
        sources.mkdir()
        self.paths = {}
        for d in self.contract['confirmatory_targets']:
            path = sources/(d+'.npy')
            np.save(path, np.ones((2,70,1), dtype='float32'))
            self.paths[d] = path
            self.contract['datasets'][d] = {'shape':[2,70,1], 'container_sha256':digest(path),
                'test_targets':[60,70], 'budgets':{'0.05':{'support_raw':[0,18], 'verification_target_rows':[18,54]}}}
        records = []
        for group in plan['tasks']:
            for d in group['targets']:
                task = {**group, 'dataset':d}
                case = project/'outputs/ectt/heldout_gpt2'/identity/'groups'/str(task['index'])/d
                profile = self.contract['datasets'][d]
                meta = {'task':task, 'support_raw':[0,18], 'verification_targets':[18,54],
                        'test_targets':[60,70], 'unique_support_values':36, 'target_container_sha256':profile['container_sha256'],
                        'budget':'0.05', 'support_scale':1.}
                write(case/'frozen_candidates.json', meta)
                for kind in ['verification','test']:
                    origins = np.arange(*meta[kind+'_targets'])
                    local = np.zeros((len(origins),2))
                    value = 1. if group['arm']=='verification_margin' else 3.
                    np.savez(case/(kind+'_candidates.npz'), origins=origins, local=local, transfer=local+value)
                finish(case, identity)
                records.append({'case':str(case), 'task':task, 'manifest_sha256':digest(case/'COMPLETE.json')})
        def job(num, stage):
            jobdir = project/'results/slurm_jobs'/num
            write(jobdir/'job_status.json', dict(stage=stage, execution_state='COMPLETED', shell_exit_code=0))
            return jobdir/'report'
        self.collector = job(scoring.COLLECTION, scoring.candidates.REPORT)
        self.inventory = self.collector/'candidate_inventory.json'
        write(self.inventory, {'identity':identity, 'failures':[], 'records':records})
        write(self.collector/'transfer_result.json', {'identity':identity, 'failed':[], 'diagnostic_cases_checked':54,
              'scientific_status':'HELDOUT_CANDIDATES_READY', 'verification_labels_scored':0, 'test_labels_scored':0})
        packet = job(scoring.PREFLIGHT, scoring.candidates.PROBE)/'S01_packet'
        write(packet/'manifest.json', {'identity':identity, 'tasks':plan['tasks'],
              'files':plan['reviewed_hashes'], 'rules':list(scoring.RULES)})
        (project/'work').mkdir()
        self.report = project/'report'
        self.report.mkdir()

    def test_packet_and_approval_are_bound_to_code_and_inventory(self):
        manifest = scoring.review_inputs(self.project, CODE)[0]
        with patch.object(transfer_worker, 'raw_arrays', side_effect=AssertionError('premature access')):
            archive = scoring.build_packet(self.project, CODE)
            with tarfile.open(archive) as tar:
                self.assertIn('S01/scripts/heldout_score.py', tar.getnames())
                self.assertIn('S01/scripts/heldout_thresholds.py', tar.getnames())
                self.assertIn('S01/tests/test_heldout_thresholds.py', tar.getnames())
                self.assertFalse(any(name.endswith('.npz') for name in tar.getnames()))
                self.assertIn('S01/disposition_template.json', tar.getnames())
            with self.assertRaisesRegex(ValueError, 'pending'):
                scoring.score(self.project, CODE, self.report)
            path = scoring.approval_path(self.project, manifest)
            write(path, {**manifest, 'status':'APPROVED', 'reviewer':'test', 'date':'fixture', 'disposition':'fixture',
                         'independent_review_complete':True, 'inventory_sha256':'other'})
            with self.assertRaisesRegex(ValueError, 'does not approve'):
                scoring.score(self.project, CODE, self.report)
            old_hashes = dict(manifest['reviewed_hashes'])
            del old_hashes['scripts/heldout_thresholds.py']
            stale = {**read(path), 'inventory_sha256': manifest['inventory_sha256'], 'reviewed_hashes': old_hashes}
            write(path, stale)
            with self.assertRaisesRegex(ValueError, 'does not approve'):
                scoring.score(self.project, CODE, self.report)
            inv = read(self.inventory)
            inv['records'] = inv['records'][:-1]
            write(self.inventory, inv)
            with self.assertRaisesRegex(ValueError, 'inventory differs'):
                scoring.review_inputs(self.project, CODE)

    def test_complete_synthetic_scoring_and_retry(self):
        manifest = scoring.review_inputs(self.project, CODE)[0]
        write(scoring.approval_path(self.project, manifest), {**manifest, 'status':'APPROVED',
              'reviewer':'synthetic test', 'date':'fixture', 'disposition':'synthetic only', 'independent_review_complete':True})
        with patch.object(scoring, 'load_contract', return_value=(self.contract, {})), \
                patch.object(transfer_worker, 'raw_paths', return_value=self.paths):
            self.assertEqual(scoring.score(self.project, CODE, self.report), 0)
            with patch.object(scoring, 'evaluate', side_effect=AssertionError('duplicated scoring')):
                self.assertEqual(scoring.score(self.project, CODE, self.report), 0)
        with (self.report/'metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 216)
        fixed = [r for r in rows if r['rule']==scoring.candidates.TWO_THIRDS]
        self.assertEqual(len(fixed), 54)
        for row in fixed:
            self.assertEqual(int(row['verification_rows']), 24)
            self.assertEqual(int(row['verification_values']), 48)
            self.assertEqual(int(row['total_target_values']), 84)
            accepted = row['arm']=='verification_margin'
            self.assertEqual(row['accepted'], str(accepted))
            self.assertEqual(float(row['deployed_MAE']), 0. if accepted else 1.)
            self.assertEqual(int(row['restricted_first_acceptance']), 24 if accepted else 37)
        comparison = read(self.report/'comparison.json')
        contrast = comparison['contrasts']['verification_margin/locked_two_sided vs prefix_mae/locked_two_sided']
        self.assertEqual(contrast['relative_MAE'], -1.)
        self.assertEqual(contrast['relative_MAE_intervals']['144'], [-1.,-1.])
        self.assertEqual(len(comparison['contrasts']), 11)
        self.assertEqual(read(self.report/'transfer_result.json')['frozen_candidates_scored'], 54)

    def test_corrupt_shard_rejected_before_label_access(self):
        manifest, records, _, _ = scoring.review_inputs(self.project, CODE)
        folder = Path(records[-1]['case'])
        (folder/'test_candidates.npz').write_bytes(b'damaged')
        with patch.object(transfer_worker, 'raw_arrays', side_effect=AssertionError('premature access')), \
                self.assertRaisesRegex(ValueError, 'forecast input changed'):
            scoring.evaluate(self.project, CODE, self.report, records, manifest, 'fixture', self.contract)


if __name__ == '__main__':
    unittest.main()
