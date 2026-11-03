"""Synthetic study fixtures and mocked Slurm; no HPC/data/model downloads."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from unittest import mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from ectt_common import read, write, digest, completed, finish, identity
from ectt_data import features, fit_scaler, inverse, verify_profile, audit_raw
from ectt_registry import build, source_domains
from ectt_statistics import confidence_sequence, decision, paired_difference
import ectt_workflow as workflow
import slurm_submit


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name) / 'nature-llm'
        self.code = self.project / 'work/code_versions/fixture'
        self.code.mkdir(parents=True)
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict(os.environ, {'TRAFFIC_MAIL_USER':'study-fixture@example.invalid'}).start()
        mock.patch('job_notifications.queued_receipt',return_value={'state':'SYNTHETIC_TEST_NO_EMAIL_SENT'}).start()
        for name in ('config', 'scripts', 'research'):
            shutil.copytree(ROOT / name, self.code / name, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(ROOT / 'Batch', self.code / 'Batch')

    def synthetic_data(self):
        c = read(self.code / 'config/ectt_protocol_v1.json')
        lock = read(self.code / 'config/sources.lock.json')
        lock['lfs_files'] = []
        for index, (slug, row) in enumerate(c['datasets'].items()):
            total, cutoff = 1500, 1200
            value = np.empty((2, total, 2), dtype=np.float64)
            value[..., 0] = np.arange(2*total).reshape(2, total) % 47 + index*3+1
            value[..., 1] = -999  # Forbidden auxiliary channel must be ignored.
            path = self.project / 'work/reference/snapshots' / lock['reference_commit'] / row['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, value)
            row.update(shape=[2,total,2], container_sha256=digest(path), source_train_raw=[0,900],
                       source_train_targets=[12,900], source_validation_targets=[900,1050], test_targets=[cutoff,total])
            for q, b in row['budgets'].items():
                size = int(float(q)*cutoff)
                start, middle = cutoff-size, cutoff-size+size//2
                b.update(raw_rows=size, support_raw=[start,middle], verification_raw=[middle,cutoff],
                    support_target_rows=[min(start+12,middle),middle],
                    verification_target_rows=[min(max(start+12,middle),cutoff),cutoff],
                    eligible_for_learned_adaptation=size//2 >= 24 and cutoff-middle >= 12)
            lock['lfs_files'].append({'path':row['path'], 'size':path.stat().st_size, 'sha256':digest(path)})
        write(self.code / 'config/sources.lock.json', lock)
        c['data']['source_lock_sha256'] = digest(self.code / 'config/sources.lock.json')
        write(self.code / 'config/ectt_protocol_v1.json', c)
        e = read(self.code / 'config/ectt_execution_v1.json')
        e['scientific_contract_sha256'] = digest(self.code / 'config/ectt_protocol_v1.json')
        write(self.code / 'config/ectt_execution_v1.json', e)
        write(self.project / 'results/slurm_jobs/610996/job_status.json',
              {'job_id':'610996','stage':'reproduction-review','shell_exit_code':0})
        return c

    def test_all_frozen_profiles_including_infeasible_budgets(self):
        c = read(self.code / 'config/ectt_protocol_v1.json')
        for row in c['datasets'].values():
            verify_profile(row)
        corrupt = copy.deepcopy(c['datasets']['pems08'])
        corrupt['test_targets'][0] -= 1
        with self.assertRaises(ValueError): verify_profile(corrupt)

    def test_registry_complete_and_nyc_sources_reused(self):
        matrix = build(self.code)
        self.assertEqual(len(matrix['families']), 35)
        self.assertEqual(len(matrix['source_fit_plan']), 42)
        self.assertEqual(sum(r['state']=='PLANNED_NOT_IMPLEMENTED' for r in matrix['target_run_plan']),132)
        c = read(self.code / 'config/ectt_protocol_v1.json')
        self.assertEqual(source_domains(c,'bike'),source_domains(c,'taxi'))
        self.assertNotIn('taxi',source_domains(c,'bike'))
        self.assertFalse(any(r['implementation']=='IMPLEMENTED' for r in matrix['families'] if r['id'] in ('M01','B03','S01')))

    def test_causal_prefix_mask_zero_and_no_normalized_clipping(self):
        scaler={'minimum':2.,'maximum':10.}
        x=np.arange(24,dtype=float).reshape(1,2,12)
        first=features(x,scaler)
        y=x.copy();y[...,6:]+=1000
        np.testing.assert_array_equal(first[...,:6,:], features(y,scaler)[...,:6,:])
        self.assertLess(first[0,0,0,0],0)
        self.assertGreater(first[0,1,-1,0],1)
        self.assertEqual(first[0,0,0,2],1) # raw zero remains observed
        mask=np.ones_like(x,dtype=bool);mask[...,:2]=False
        missing=features(np.where(mask,x,np.nan),scaler,mask)
        self.assertTrue(np.isfinite(missing).all())
        self.assertTrue((missing[...,:2,2]==0).all())
        np.testing.assert_array_equal(inverse([-1,0,1],scaler),[0,2,10])

    def test_scaler_ignores_heldout_values(self):
        raw=np.arange(100,dtype=float).reshape(2,50)
        scaler=fit_scaler(raw,[0,20]);raw[:,20:]+=1e6
        self.assertEqual(scaler,fit_scaler(raw,[0,20]))
        with self.assertRaises(ValueError):fit_scaler(np.ones((2,50)),[0,20])

    def test_full_synthetic_preflight_resumes_and_detects_corruption(self):
        self.synthetic_data()
        report=self.project/'results/slurm_jobs/70001/report'
        self.assertEqual(workflow.run(self.project,self.code,report,'ectt-preflight'),0)
        store=workflow.stage_store(self.project,self.code,'ectt-preflight')
        self.assertEqual(read(store/'result.json')['completed_datasets'],8)
        self.assertEqual(workflow.next_stage(self.project,self.code),'ectt-stat-check')
        self.assertEqual(workflow.run(self.project,self.code,report,'ectt-preflight'),0)
        ledger=next((store/'datasets').rglob('*.csv'));ledger.write_text('corrupt')
        with self.assertRaises(ValueError):workflow.next_stage(self.project,self.code)

    def test_data_failure_nonzero_preserves_other_partial_audits(self):
        c=self.synthetic_data()
        path=self.project/'work/reference/snapshots'/read(self.code/'config/sources.lock.json')['reference_commit']/c['datasets']['pems03']['path']
        path.write_bytes(b'broken')
        report=self.project/'results/slurm_jobs/70002/report'
        self.assertEqual(workflow.run(self.project,self.code,report,'ectt-preflight'),2)
        store=workflow.stage_store(self.project,self.code,'ectt-preflight')
        self.assertEqual(read(store/'result.json')['completed_datasets'],7)
        self.assertFalse((store/'COMPLETE.json').exists())
        self.assertTrue((store/'worker_logs/pems03.log').exists())

    def test_completed_data_reuse_rechecks_current_inputs_without_mutating_old_report(self):
        c=self.synthetic_data()
        report=self.project/'results/slurm_jobs/70003/report'
        self.assertEqual(workflow.run(self.project,self.code,report,'ectt-preflight'),0)
        store=workflow.stage_store(self.project,self.code,'ectt-preflight')
        marker=(store/'COMPLETE.json').read_bytes()
        path=Path(read(store/'result.json')['datasets']['pems03']['source_path'])
        path.write_bytes(b'corrupt input')
        self.assertEqual(workflow.run(self.project,self.code,report,'ectt-preflight'),2)
        self.assertEqual((store/'COMPLETE.json').read_bytes(),marker)
        self.assertTrue(completed(store,identity(self.code)))

    def test_pair_bound_and_decisions_include_censoring(self):
        rng=np.random.default_rng(7)
        l,t,y=[rng.normal(size=(200,6)) for _ in range(3)]
        d,b=paired_difference(l,t,y,2)
        self.assertTrue((np.abs(d)<=b+1e-12).all())
        rejected=decision(np.zeros(50),np.ones(50))
        self.assertTrue(rejected['censored']);self.assertEqual(rejected['first_certificate'],51)
        accepted=decision(-np.ones(500),np.ones(500))
        self.assertTrue(accepted['accepted']);self.assertGreaterEqual(accepted['first_certificate'],12)
        with self.assertRaises(ValueError):confidence_sequence([2],[1])

    def test_boundary_matches_mixture_equation(self):
        d=np.array([.1,-.2,.2]);b=np.array([.3,.4,.5]);rho=.01;delta=.05
        low,high=confidence_sequence(d,b,rho,delta)
        n=np.arange(1,4);v=np.cumsum(b*b)
        width=np.sqrt((v+rho)*np.log((v+rho)/(rho*delta**2)))/n
        np.testing.assert_allclose(high-low,2*width,rtol=1e-14)
        self.assertEqual(len(confidence_sequence([],[])[0]),0)

    def test_full_statistical_simulation_writes_machine_readable_results(self):
        from ectt_statistics import simulate
        store=self.project/'synthetic_stats';store.mkdir()
        c=read(self.code/'config/ectt_protocol_v1.json')
        e=read(self.code/'config/ectt_execution_v1.json')
        simulate(store,e['statistical_simulation'],c['verifier'],lambda **x:None)
        result=read(store/'result.json')
        self.assertEqual(result['execution_kind'],'SYNTHETIC')
        self.assertTrue(all(r['passed'] for r in result['scenarios']))
        self.assertEqual(result['S01_independent_mathematical_review'],'PENDING_NOT_CLEARED_BY_SIMULATION')

    def fake_slurm(self,args,**kwargs):
        if args[0]=='scontrol':return subprocess.CompletedProcess(args,0,'PartitionName='+args[3]+' MaxTime=12:00:00','')
        if args[0]=='sbatch':return subprocess.CompletedProcess(args,0,'73010\n','')
        if args[0]=='squeue':return subprocess.CompletedProcess(args,0,'','')
        raise AssertionError(args)

    def test_next_first_job_cpu_and_no_array_or_gpu(self):
        with mock.patch.object(slurm_submit,'ROOT',self.project),mock.patch.object(slurm_submit.shutil,'which',return_value='/fake/sbatch'),mock.patch.object(slurm_submit.subprocess,'run',side_effect=self.fake_slurm) as execute:
            slurm_submit.main(['--project',str(self.project),'--code',str(self.code),'--stage','ectt-next'])
        record=read(self.project/'results/slurm_jobs/73010/submission.json')
        self.assertEqual((record['stage'],record['cpus'],record['gpus']),('ectt-preflight',5,0))
        cmd=next(x.args[0] for x in execute.call_args_list if x.args[0][0]=='sbatch')
        self.assertFalse(any(x.startswith('--array') or x.startswith('--gres') for x in cmd))

    def mark(self,stage):
        store=workflow.stage_store(self.project,self.code,stage);store.mkdir(parents=True)
        write(store/'result.json',{'synthetic_fixture':True});finish(store,identity(self.code))

    def test_gpu_pilot_one_gpu_five_cpus_after_prerequisites(self):
        self.mark('ectt-preflight');self.mark('ectt-stat-check')
        with mock.patch.object(slurm_submit,'ROOT',self.project),mock.patch.object(slurm_submit.shutil,'which',return_value='/fake/sbatch'),mock.patch.object(slurm_submit.subprocess,'run',side_effect=self.fake_slurm) as execute:
            slurm_submit.main(['--project',str(self.project),'--code',str(self.code),'--stage','ectt-next'])
        record=read(self.project/'results/slurm_jobs/73010/submission.json')
        self.assertEqual((record['cpus'],record['gpus'],record['partition']),(5,1,'gpuq'))
        self.assertIn('--gres=gpu:1',record['command'])
        self.assertNotIn('--gres=gpu:4',record['command'])

    def test_cannot_skip_prerequisites_or_submit_unimplemented_grid(self):
        with self.assertRaises(ValueError):workflow.require_previous(self.project,self.code,'ectt-gpu-pilot')
        for stage in workflow.STAGES:self.mark(stage)
        with self.assertRaisesRegex(ValueError,'benchmark training kernels'):workflow.next_stage(self.project,self.code)

    def legacy_admissions(self):
        self.assertEqual(workflow.admission_dependencies(self.code), workflow.PREVIOUS_ADMISSION_DEPENDENCIES)
        stores = {}
        for stage in workflow.STAGES[:2]:
            store = self.project / 'outputs/ectt/admission' / workflow.PREVIOUS_ADMISSION_ID / stage
            write(store / 'result.json', {'synthetic_fixture': True})
            finish(store, workflow.PREVIOUS_ADMISSION_ID)
            stores[stage] = store
        return stores

    def test_repaired_pilot_reuses_only_verified_compatible_cpu_admissions(self):
        stores = self.legacy_admissions()
        before = {stage: (store / 'COMPLETE.json').read_bytes() for stage, store in stores.items()}
        self.assertNotEqual(identity(self.code), workflow.PREVIOUS_ADMISSION_ID)
        self.assertEqual(workflow.next_stage(self.project, self.code), 'ectt-gpu-pilot')
        records = workflow.require_previous(self.project, self.code, 'ectt-gpu-pilot')
        self.assertTrue(all(row['reused_compatible_admission'] for row in records.values()))
        for stage, store in stores.items():
            self.assertEqual((store / 'COMPLETE.json').read_bytes(), before[stage])
        # Even a completed old pilot cannot clear this release's new pilot.
        old_pilot = stores['ectt-preflight'].parent / 'ectt-gpu-pilot'
        write(old_pilot / 'result.json', {'synthetic_fixture': True})
        finish(old_pilot, workflow.PREVIOUS_ADMISSION_ID)
        self.assertEqual(workflow.next_stage(self.project, self.code), 'ectt-gpu-pilot')
        with mock.patch.object(slurm_submit, 'ROOT', self.project), mock.patch.object(slurm_submit.shutil, 'which', return_value='/fake/sbatch'), mock.patch.object(slurm_submit.subprocess, 'run', side_effect=self.fake_slurm):
            slurm_submit.main(['--project', str(self.project), '--code', str(self.code), '--stage', 'ectt-next'])
        record = read(self.project / 'results/slurm_jobs/73010/submission.json')
        self.assertEqual((record['stage'], record['cpus'], record['gpus']), ('ectt-gpu-pilot', 5, 1))

    def test_reuse_refuses_corrupt_historical_or_current_completion(self):
        stores = self.legacy_admissions()
        (stores['ectt-preflight'] / 'result.json').write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'corrupt completed artifact'):
            workflow.require_previous(self.project, self.code, 'ectt-gpu-pilot')
        write(stores['ectt-preflight'] / 'result.json', {'synthetic_fixture': True})
        self.mark('ectt-preflight')
        current = workflow.stage_store(self.project, self.code, 'ectt-preflight')
        (current / 'result.json').write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'corrupt completed artifact'):
            workflow.next_stage(self.project, self.code)

    def test_reuse_refuses_any_changed_admission_dependency(self):
        self.legacy_admissions()
        for name in ('config/ectt_protocol_v1.json', 'config/ectt_execution_v1.json',
                     'config/sources.lock.json', 'scripts/ectt_data.py', 'scripts/ectt_common.py',
                     'scripts/ectt_statistics.py', 'scripts/ectt_registry.py'):
            with self.subTest(path=name):
                path = self.code / name
                original = path.read_bytes()
                path.write_bytes(original + b'\n')
                try:
                    self.assertEqual(workflow.next_stage(self.project, self.code), 'ectt-preflight')
                    with self.assertRaisesRegex(ValueError, 'Required stage not completed'):
                        workflow.require_previous(self.project, self.code, 'ectt-gpu-pilot')
                finally:
                    path.write_bytes(original)

    def test_model_only_repair_can_reuse_cpu_admissions_but_never_an_old_pilot(self):
        self.legacy_admissions()
        for name in ('scripts/ectt_model.py', 'scripts/ectt_pilot.py'):
            path = self.code / name
            original = path.read_bytes()
            path.write_bytes(original + b'\n# synthetic model-only repair\n')
            try:
                self.assertEqual(workflow.next_stage(self.project, self.code), 'ectt-gpu-pilot')
                self.assertTrue(all(row['reused_compatible_admission'] for row in
                                    workflow.require_previous(self.project, self.code, 'ectt-gpu-pilot').values()))
            finally:
                path.write_bytes(original)

    def test_study_refuses_silent_email_omission(self):
        with mock.patch.dict(os.environ,{'TRAFFIC_MAIL_USER':''}), mock.patch.object(slurm_submit,'ROOT',self.project),mock.patch.object(slurm_submit.shutil,'which',return_value='/fake/sbatch'),mock.patch.object(slurm_submit.subprocess,'run',side_effect=self.fake_slurm) as execute:
            with self.assertRaisesRegex(ValueError,'require saved email'):
                slurm_submit.main(['--project',str(self.project),'--code',str(self.code),'--stage','ectt-next'])
        self.assertFalse(any(x.args[0][0]=='sbatch' for x in execute.call_args_list))

    def test_active_study_not_last_job_blocks_duplicate_submission(self):
        write(self.project/'results/slurm_jobs/123/submission.json',{'stage':'ectt-preflight'})
        def live(args,**kwargs):
            if args[:3]==['squeue','--noheader','--format=%A']:return subprocess.CompletedProcess(args,0,'123\n','')
            return self.fake_slurm(args,**kwargs)
        with mock.patch.object(slurm_submit,'ROOT',self.project),mock.patch.object(slurm_submit.shutil,'which',return_value='/fake/sbatch'),mock.patch.object(slurm_submit.subprocess,'run',side_effect=live) as execute:
            slurm_submit.main(['--project',str(self.project),'--code',str(self.code),'--stage','ectt-next'])
        self.assertFalse(any(x.args[0][0]=='sbatch' for x in execute.call_args_list))

    def test_monitor_reports_study_progress_without_download_or_zero_training_mislabels(self):
        import slurm_check
        job=self.project/'results/slurm_jobs/73011'
        write(job/'submission.json',{'project':str(self.project),'job_id':'73011','job_name':'ectt-qwen-pilot',
                                    'stage':'ectt-gpu-pilot','partition':'gpuq'})
        write(job/'report/ectt_progress.json',{'phase':'SUPPORT_TRAINING_SMOKE','optimizer_steps':3})
        output=io.StringIO()
        with mock.patch.object(sys,'argv',['Check','--project',str(self.project),'73011']),mock.patch.object(slurm_check,'command',return_value={'stdout':'','stderr':'','exit_code':0}),redirect_stdout(output):
            slurm_check.main()
        self.assertIn('SUPPORT_TRAINING_SMOKE',output.getvalue())
        self.assertNotIn('ASSET WORKERS',output.getvalue())
        self.assertNotIn('reference inference/audits do not train',output.getvalue())


if __name__=='__main__':unittest.main()
