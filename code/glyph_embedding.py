# glyph_embedding.py
import torch
import torch.nn as nn

class StrokeCNN(nn.Module):
    def __init__(self, in_ch, hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(hidden*8*8, hidden)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        b, c, h, w = x.shape
        return self.fc(x.view(b, c*h*w))

class GlyphEmbedding(nn.Module):
    def __init__(self, num_rad=214, rad_dim=64, stroke_ch=1, stro_dim=32, final=128):
        super().__init__()
        self.rad_emb = nn.Embedding(num_rad, rad_dim)
        self.stroke_cnn = StrokeCNN(stroke_ch, stro_dim)
        self.gate = nn.Sequential(
            nn.Linear(rad_dim+stro_dim, final),
            nn.Sigmoid()
        )
        self.fusion = nn.Linear(rad_dim+stro_dim, final)
        self.norm = nn.LayerNorm(final)
        self.drop = nn.Dropout(0.2)

    def forward(self, rad, strokes):
        r = self.rad_emb(rad)
        s = self.stroke_cnn(strokes)
        x = torch.cat([r, s], dim=-1)
        g = self.gate(x)
        fused = g * self.fusion(x) + (1-g) * x
        return self.drop(self.norm(fused))

def test():
    model = GlyphEmbedding()
    b = 8
    rad = torch.randint(0,214,(b,))
    strokes = torch.randn(b,1,16,16)
    o = model(rad, strokes)
    assert o.shape == (b,128)

if __name__ == "__main__":
    test()
