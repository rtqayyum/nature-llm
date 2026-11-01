"""Numerical four-patch interface; no information exchange between origins."""
import torch
from torch import nn


class NumericalTrafficModel(nn.Module):
    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Linear(9, hidden_size)
        self.spatial = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, hidden_size))
        self.head = nn.Linear(hidden_size, 1)

    def encode(self, inputs):
        if inputs.ndim != 4 or inputs.shape[-2:] != (12, 3):
            raise ValueError('Expected origin x node x 12 x 3 features')
        batch, nodes = inputs.shape[:2]
        patches = inputs.reshape(batch, nodes, 4, 9)
        # Mean over time, equivariant mean over available nodes. Never reduce
        # the origin dimension. Context is fused into four temporal tokens.
        node_context = inputs.mean(dim=2)
        weights = inputs[..., 2].sum(dim=2)
        encoded = self.spatial(node_context)
        context = (encoded*weights.unsqueeze(-1)).sum(dim=1)/weights.sum(dim=1).clamp_min(1).unsqueeze(-1)
        tokens = self.projection(patches)+context[:, None, None, :]
        dtype = next(self.backbone.parameters()).dtype
        tokens = tokens.reshape(batch*nodes, 4, -1).to(dtype)
        result = self.backbone(inputs_embeds=tokens, use_cache=False, return_dict=True)
        return result.last_hidden_state[:, -1].float().reshape(batch, nodes, -1)

    def forward(self, inputs):
        return self.head(self.encode(inputs)).squeeze(-1)


def qwen_model(directory, contract):
    from transformers import AutoModel
    from peft import LoraConfig, TaskType, get_peft_model
    backbone = AutoModel.from_pretrained(str(directory), local_files_only=True,
        torch_dtype=torch.bfloat16, use_safetensors=True, trust_remote_code=False,
        attn_implementation='eager')
    if backbone.config.num_hidden_layers != 28 or len(backbone.layers) != 28:
        raise ValueError('All 28 pinned Qwen blocks must remain active')
    backbone.config.use_cache = False
    recipe = contract['model']['lora']
    backbone = get_peft_model(backbone, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
        r=recipe['rank'], lora_alpha=recipe['alpha'], lora_dropout=recipe['dropout'],
        target_modules=recipe['modules'], bias='none'))
    model = NumericalTrafficModel(backbone, backbone.config.hidden_size)
    if any(p.requires_grad and 'lora_' not in name for name, p in backbone.named_parameters()):
        raise ValueError('Unexpected trainable backbone weights')
    return model


def invariance_checks(model, inputs, atol=1e-5, rtol=1e-4):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            reference = model(inputs)
            variants = [('single_origin', model(inputs[:1]), reference[:1]),
                        ('origin_permutation', model(inputs.flip(0)).flip(0), reference)]
            changed = inputs.clone()
            changed[1:] = 100
            variants.append(('later_companion_mutation', model(changed)[:1], reference[:1]))
            variants.append(('node_permutation', model(inputs.flip(1)).flip(1), reference))
            rows = []
            for name, observed, expected in variants:
                passed = torch.isfinite(observed).all() and torch.allclose(observed, expected, atol=atol, rtol=rtol)
                rows.append({'case': name, 'passed': bool(passed),
                             'maximum_absolute_difference': float((observed-expected).abs().max())})
                if not passed:
                    raise ValueError('Forecast invariance failed: ' + str(rows[-1]))
        return rows
    finally:
        model.train(was_training)
