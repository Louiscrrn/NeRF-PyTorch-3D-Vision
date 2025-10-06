# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Positional Encoding ----------
def posenc(x, L_embed=6):
    rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn((2. ** i) * x))
    return torch.cat(rets, dim=-1)

# ---------- NeRF Model ----------
class NeRFModel(nn.Module):
    def __init__(self, D=8, W=256, L_embed=6):
        super().__init__()
        self.L_embed = L_embed
        self.in_dim = 3 + 3 * 2 * L_embed

        self.layers = nn.ModuleList()
        for i in range(D):
            if i == 0:
                self.layers.append(nn.Linear(self.in_dim, W))
            elif i % 4 == 0:
                self.layers.append(nn.Linear(W + self.in_dim, W))
            else:
                self.layers.append(nn.Linear(W, W))
        self.output_layer = nn.Linear(W, 4)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        inputs = x
        for i, layer in enumerate(self.layers):
            if i % 4 == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
            x = self.relu(layer(x))
        return self.output_layer(x)
