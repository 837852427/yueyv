import torch
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self, dim, layers=3):
        super().__init__()
        self.gates = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(layers)])
        self.projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = x
        for g, p in zip(self.gates, self.projs):
            y = p(out)
            out = g * y + (1-g) * out
        return self.norm(out)

def test():
    rc = ResidualConnection(768, layers=4)
    b = 6
    x = torch.randn(b,768)
    o = rc(x)
    assert o.shape == (b,768)

if __name__ == "__main__":
    test()
