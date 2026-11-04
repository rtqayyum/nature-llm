"""Small graph-free transformer with origin-independent temporal attention."""
import math
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.output = nn.Linear(width, 3*width), nn.Linear(width, width)
        self.feed = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width//self.heads).permute(2, 0, 3, 1, 4)
        weights = ((q @ k.transpose(-2, -1))/math.sqrt(width//self.heads)).softmax(dim=-1)
        x = x+self.output((weights @ v).transpose(1, 2).reshape(batch, length, width))
        return x+self.feed(self.norm2(x))


class CompactTrafficModel(nn.Module):
    def __init__(self, width, layers, heads):
        super().__init__()
        if width % heads or min(width, layers, heads) < 1:
            raise ValueError('Invalid compact architecture')
        self.projection = nn.Linear(9, width)
        self.position = nn.Parameter(torch.zeros(4, width))
        nn.init.normal_(self.position, std=.02)
        self.spatial = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, width))
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, 1))
        # Start every configuration from the same persistence forecast.
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[-2:] != (12, 3):
            raise ValueError('Expected origin x node x 12 x 3 features')
        batch, nodes = inputs.shape[:2]
        node = self.spatial(inputs.mean(dim=2))
        weights = inputs[..., 2].sum(dim=2).double()
        context = ((node.double()*weights.unsqueeze(-1)).sum(dim=1)
                   /weights.sum(dim=1).clamp_min(1).unsqueeze(-1)).float()
        x = (self.projection(inputs.reshape(batch, nodes, 4, 9))
             +self.position+context[:, None, None, :]).reshape(batch*nodes, 4, -1)
        for block in self.blocks:
            x = block(x)
        residual = self.head(self.norm(x[:, -1])).reshape(batch, nodes)
        return inputs[:, :, -1, 0]+residual
