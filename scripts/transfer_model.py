"""Source backbones and fresh, bottleneck-32 target residual heads."""
import torch
from torch import nn


class CausalUTP(nn.Module):
    """Two active pretrained GPT-2 blocks and the released traffic encoder."""
    def __init__(self, released):
        super().__init__()
        self.traffic = released
        self.traffic.gpt2.h = self.traffic.gpt2.h[:2]
        for name in ('wte', 'wpe', 'ln_f'):
            # The released LLMModule calls its blocks directly.
            delattr(self.traffic.gpt2, name)
        del self.traffic.pro_vector
        for p in self.traffic.parameters():
            p.requires_grad_(True)

    def encode(self, inputs):
        m = self.traffic
        a, raw = m.trend_enhancement_module(
            inputs[..., :2].unsqueeze(0), m.f1, m.f2, m.f3, m.f4,
            m.feature_fusion_conv, m.weights, m.rms_norm_conv)
        x, y = m.spatiotemporal_feature_encoding_module(
            a, raw, m.dilated_conv1, m.dilated_conv2, m.dilated_conv3, m.conv1_weights,
            m.conv2_weights, m.conv3_weights, m.gru1, m.gru2, m.gru3, m.gru_weight1,
            m.gru_weight2, m.gru_weight3, m.rms_norm_gru_out, m.mlp, m.fc, m.fcc)
        hidden = torch.cat((x, y), dim=-1)
        for block in m.gpt2.h:
            hidden = block(hidden, use_cache=False)[0]
        return hidden

    def head(self, encoded):
        return self.traffic.output_linear(encoded)

    def forward(self, inputs):
        return self.head(self.encode(inputs)).squeeze(-1)


class OriginIndependentQwen(nn.Module):
    """Keep BF16 kernel shapes fixed as forecast origins are added or removed."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode(self, inputs):
        return torch.cat([self.model.encode(origin) for origin in inputs.split(1)], dim=0)

    def head(self, encoded):
        return self.model.head(encoded)


def create(family, project, code, contract):
    if family == 'B04':
        from ectt_model import qwen_model
        from qwen_recovery import cached_weights
        directory, _ = cached_weights(project, contract)
        return OriginIndependentQwen(qwen_model(directory, contract))
    if family != 'B03':
        raise ValueError('Unknown transfer family')
    from acquire import verify
    from causal_protocol import install_fusion
    from ectt_common import inside, read
    from bootstrap_hpc import blob_sha
    from pems08_infer import module_from
    lock = read(code/'config/sources.lock.json')
    repo = inside(project, project/'work/reference/snapshots'/lock['reference_commit'])
    source = repo/'model/LLM_UTP.py'
    entry = next(row for row in lock['reference_files'] if row['path'] == 'model/LLM_UTP.py')
    if blob_sha(source.read_bytes()) != entry['sha']:
        raise ValueError('Pinned LLM-UTP architecture file differs')
    base = inside(project, project/'work/models/gpt2'/lock['gpt2']['revision'])
    for entry in lock['gpt2']['files']:
        if not verify(base/entry['path'], entry):
            raise ValueError('Pinned GPT-2 file failed verification: '+entry['path'])
    module = module_from(source, '_transfer_author_model')
    released = module.TrafficPredictionModel(1, 12, 64, [1, 2, 4], 12, 0, str(base), 768)
    install_fusion(released, module, source.read_text())
    return CausalUTP(released)


def residual_head(width, device):
    head = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1)).to(device)
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def predict_encoded(model, x, residual=None):
    h = model.encode(x)
    out = model.head(h)
    if residual is not None:
        out = out+residual(h)
    return out.squeeze(-1)


def check_invariance(model, inputs, family, residual=None):
    model.eval()
    with torch.no_grad():
        before = predict_encoded(model, inputs, residual)
        changed = inputs.clone()
        changed[1:] = 100
        variants = [
            ('single_origin', before[:1], predict_encoded(model, inputs[:1], residual)),
            ('origin_permutation', before, predict_encoded(model, inputs.flip(0), residual).flip(0)),
            ('later_companion_mutation', before[:1], predict_encoded(model, changed, residual)[:1])
        ]
        if family == 'B04':
            variants.append(('node_permutation', before,
                             predict_encoded(model, inputs.flip(1), residual).flip(1)))
        result = {}
        for name, expected, observed in variants:
            delta = float((expected-observed).abs().max())
            if not torch.isfinite(observed).all() or not torch.allclose(expected, observed, atol=1e-3, rtol=1e-3):
                raise ValueError(f'Forecast invariance failed: {name} (max difference {delta:.6g})')
            result[name] = delta
        return result
