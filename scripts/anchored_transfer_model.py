"""Matched zero-head controls for the development-only persistence anchor."""
import torch
from torch import nn


class AnchoredTransfer(nn.Module):
    def __init__(self, model, arm):
        super().__init__()
        if arm not in ('direct_zero', 'persistence_residual'):
            raise ValueError('Unknown anchored development condition')
        self.model, self.arm = model, arm
        # Keep the extra, observed last-value feature in BOTH conditions.
        # The same encoder and fresh residual-head dimensions are compared.
        head = (model.traffic.output_linear if hasattr(model, 'traffic')
                else model.model.head)
        # B03 keeps its released Linear/ReLU/Linear prediction module; B04
        # uses one linear layer. Zero only the final scalar output layer.
        if isinstance(head, nn.Sequential):
            head = head[-1]
        if not isinstance(head, nn.Linear) or head.out_features != 1:
            raise ValueError('Expected the admitted scalar numerical output layer')
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)

    def encode(self, inputs):
        return torch.cat((self.model.encode(inputs), inputs[..., -1, :1]), dim=-1)

    def head(self, encoded):
        correction = self.model.head(encoded[..., :-1])
        return correction + encoded[..., -1:] if self.arm == 'persistence_residual' else correction

    def forward(self, inputs):
        return self.head(self.encode(inputs)).squeeze(-1)


def initial_check(model, inputs):
    model.eval()
    with torch.no_grad():
        actual = model(inputs)
        expected = (inputs[..., -1, 0] if model.arm == 'persistence_residual'
                    else torch.zeros_like(actual))
    if not torch.equal(actual, expected):
        raise ValueError('Initial zero-head forecast does not match its declared anchor')
    return {'initial_forecast': 'last_observed_normalized_value' if model.arm == 'persistence_residual'
            else 'zero_normalized_value', 'maximum_normalized_difference': 0.0}
