import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.2)

    def forward(self, q, kv):
        out, _ = self.attn(q, kv, kv)
        return self.drop(self.norm(out + q))

class FeatureFusionAndProjection(nn.Module):
    def __init__(self, dsem, dpho, dgly, hidden=512, out_dim=768):
        super().__init__()
        self.proj_sem = nn.Linear(dsem, hidden)
        self.proj_pho = nn.Linear(dpho, hidden)
        self.proj_gly = nn.Linear(dgly, hidden)
        self.cross1 = CrossModalAttention(hidden)
        self.cross2 = CrossModalAttention(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden*3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(0.3)

    def forward(self, sem, pho, gly):
        p = self.proj_pho(pho).unsqueeze(1)
        g = self.proj_gly(gly).unsqueeze(1)
        sp = self.cross1(s, p)
        sg = self.cross2(s, g)
        cat = torch.cat([sp.squeeze(1), sg.squeeze(1), p.squeeze(1), g.squeeze(1), sem], dim=-1)
        out = self.ffn(cat)
        return self.drop(self.norm(out))

def test():
    ff = FeatureFusionAndProjection(768,128,128)
    b = 4
    sem = torch.randn(b,768)
    pho = torch.randn(b,128)
    gly = torch.randn(b,128)
    o = ff(sem, pho, gly)
    assert o.shape == (b,768)

if __name__ == "__main__":
    test()
