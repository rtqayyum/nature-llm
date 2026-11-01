"""Tiny CPU Qwen/LoRA tests; never official weights or real traffic results."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import importlib.util

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
AVAILABLE=all(importlib.util.find_spec(x) for x in ('torch','transformers','peft'))


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
