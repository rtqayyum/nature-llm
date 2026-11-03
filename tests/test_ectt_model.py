"""Tiny CPU Qwen/LoRA tests; never official weights or real traffic results."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import importlib.util
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
AVAILABLE=all(importlib.util.find_spec(x) for x in ('torch','transformers','peft'))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires local CPU Torch')
class PoolingPrecisionTests(unittest.TestCase):
    def test_node_permutation_preserves_bf16_tokens_at_real_interface_width(self):
        import torch
        from ectt_model import NumericalTrafficModel
        class Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=False)
            def forward(self, inputs_embeds, **kwargs):
                self.tokens = inputs_embeds.detach().clone()
                return SimpleNamespace(last_hidden_state=inputs_embeds)
        torch.set_num_threads(5)
        for seed, nodes in ((0, 170), (1, 358)):
            with self.subTest(seed=seed, nodes=nodes):
                torch.manual_seed(seed)
                model = NumericalTrafficModel(Recorder(), 1536).eval()
                x = torch.rand(2, nodes, 12, 3); x[..., 2] = 1
                if nodes == 358:
                    x[..., 2] = (torch.rand(2, nodes, 12) > .3).float()
                    x[:, 0, :, 2] = 0
                permutation = torch.randperm(nodes)
                with torch.no_grad():
                    model(x)
                    expected = model.backbone.tokens.reshape(2, nodes, 4, -1).clone()
                    for order in (torch.arange(nodes).flip(0), permutation):
                        model(x[:, order])
                        actual = model.backbone.tokens.reshape(2, nodes, 4, -1)[:, order.argsort()]
                        self.assertTrue(torch.equal(actual, expected),
                                        'Node order changed BF16 input tokens before Qwen')

    def test_invariance_gate_still_rejects_node_order_dependence_and_records_failure(self):
        import torch
        from ectt_model import invariance_checks
        class NodeBiased(torch.nn.Module):
            def forward(self, inputs):
                return inputs[:, :, -1, 0] + torch.arange(inputs.shape[1])
        model = NodeBiased().train()
        rows = []
        with self.assertRaisesRegex(ValueError, 'node_permutation'):
            invariance_checks(model, torch.rand(2, 6, 12, 3), atol=1e-3, rtol=1e-3,
                              record=lambda value: rows.append(value))
        self.assertTrue(model.training)
        self.assertEqual(len(rows[-1]), 4)
        self.assertFalse(rows[-1][-1]['passed'])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires local CPU Torch')
class PilotEntryTests(unittest.TestCase):
    def test_real_entrypoint_imports_before_cuda_admission(self):
        import torch
        from ectt_pilot import run
        # Reach the exact lazy-import path used by Slurm, without a download
        # or GPU. The old broken release raises ImportError before this guard.
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'exactly one visible CUDA GPU'):
                run(None, None, None, None, None, None, lambda **values: None)

    def test_checkpoint_helpers_and_exact_cpu_resume(self):
        import torch
        from ectt_pilot import fit, trainable_state, restore_trainable
        torch.manual_seed(17)
        base = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
        full, partial = copy.deepcopy(base), copy.deepcopy(base)
        x, y = torch.rand(2, 3), torch.rand(2, 1)
        rng = torch.get_rng_state()
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / 'full', Path(temporary) / 'resume'
            a.mkdir(); b.mkdir()
            expected = fit(full, x, y, a, 'synthetic', 3, .001, lambda **values: None)
            torch.set_rng_state(rng)
            fit(partial, x, y, b, 'synthetic', 3, .001, lambda **values: None, stop_after=1)
            recovered = copy.deepcopy(base)
            actual = fit(recovered, x, y, b, 'synthetic', 3, .001, lambda **values: None)
            self.assertEqual(expected['losses'], actual['losses'])
            self.assertEqual(actual['resumed_from_step'], 1)
            self.assertEqual(actual['new_optimizer_steps'], 2)
            for name, expected_tensor in trainable_state(full).items():
                self.assertTrue(torch.equal(expected_tensor, trainable_state(recovered)[name]), name)
            clone = copy.deepcopy(base)
            restore_trainable(clone, trainable_state(recovered))
            clone.eval(); recovered.eval()
            self.assertTrue(torch.equal(clone(x), recovered(x)))


@unittest.skipUnless(AVAILABLE,'Run in the separate local ectt-test environment')
class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.torch=torch
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)

    def make(self,layers=2):
        from transformers import Qwen2Config,Qwen2Model
        from peft import LoraConfig,TaskType,get_peft_model
        from ectt_model import NumericalTrafficModel
        cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=layers,
            num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32,attention_dropout=0.)
        core=get_peft_model(Qwen2Model(cfg),LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
            r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],bias='none'))
        return NumericalTrafficModel(core,16)

    def test_cpu_origin_node_invariance_and_gradients(self):
        from ectt_model import invariance_checks
        t=self.torch;model=self.make()
        x=t.rand(2,3,12,3);x[...,2]=1
        self.assertTrue(all(row['passed'] for row in invariance_checks(model,x)))
        out=model(x);self.assertEqual(out.shape,(2,3))
        out.square().mean().backward()
        got=[name for name,p in model.named_parameters() if p.grad is not None]
        self.assertTrue(any('lora_' in n for n in got))
        self.assertTrue(any('spatial' in n for n in got))
        self.assertFalse(any(p.requires_grad and 'lora_' not in n for n,p in model.backbone.named_parameters()))

    def test_actual_factory_keeps_28_tiny_blocks(self):
        from transformers import Qwen2Config,Qwen2Model,AutoModel
        from ectt_model import qwen_model
        cfg=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=28,
            num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32)
        contract={'model':{'lora':{'rank':8,'alpha':16,'dropout':0.,'modules':['q_proj','v_proj']}}}
        with mock.patch.object(AutoModel,'from_pretrained',return_value=Qwen2Model(cfg)) as loader:
            model=qwen_model(Path(self.tmp.name),contract)
        self.assertEqual(len(model.backbone.base_model.model.layers),28)
        self.assertTrue(loader.call_args.kwargs['local_files_only'])
        self.assertTrue(loader.call_args.kwargs['use_safetensors'])
        self.assertEqual(model(self.torch.rand(2,2,12,3)).shape,(2,2))

    def test_bf16_qwen_all_28_blocks_permutation_and_pool_gradients(self):
        from transformers import Qwen2Config, Qwen2Model, AutoModel
        from ectt_model import qwen_model, invariance_checks
        t = self.torch
        cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=28,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
        contract = {'model': {'lora': {'rank': 8, 'alpha': 16, 'dropout': 0., 'modules': ['q_proj', 'v_proj']}}}
        with mock.patch.object(AutoModel, 'from_pretrained', return_value=Qwen2Model(cfg).to(t.bfloat16)):
            model = qwen_model(Path(self.tmp.name), contract)
        x = t.rand(2, 170, 12, 3); x[..., 2] = 1
        self.assertTrue(all(row['passed'] for row in invariance_checks(model, x, atol=1e-3, rtol=1e-3)))
        model(x).square().mean().backward()
        for name, parameter in model.named_parameters():
            if name.startswith(('projection.', 'spatial.', 'head.')) or 'lora_' in name:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(t.isfinite(parameter.grad).all(), name)

    def test_exact_optimizer_resume_matches_uninterrupted_cpu_fit(self):
        from ectt_pilot import fit
        t=self.torch
        base=self.make();full=copy.deepcopy(base);partial=copy.deepcopy(base)
        x=t.rand(2,3,12,3);x[...,2]=1;y=t.rand(2,3)
        root=Path(self.tmp.name);a=root/'full';b=root/'resume';a.mkdir();b.mkdir()
        first=fit(full,x,y,a,'synthetic-contract',3,.001,lambda **x:None)
        fit(partial,x,y,b,'synthetic-contract',3,.001,lambda **x:None,stop_after=1)
        recovered=copy.deepcopy(base)
        resumed=fit(recovered,x,y,b,'synthetic-contract',3,.001,lambda **x:None)
        self.assertEqual(resumed['resumed_from_step'],1)
        self.assertEqual(resumed['new_optimizer_steps'],2)
        self.assertEqual(first['losses'],resumed['losses'])
        for (n,p),(m,q) in zip(full.named_parameters(),recovered.named_parameters()):
            self.assertEqual(n,m);self.assertTrue(t.equal(p,q),n)
        bad=next(b.glob('step-003.pt'));bad.write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'digest differs'):fit(recovered,x,y,b,'synthetic-contract',3,.001,lambda **x:None)


if __name__=='__main__':unittest.main()
