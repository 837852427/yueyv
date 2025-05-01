import torch
import torch.nn as nn

class PhonoEncoder(nn.Module):
    def __init__(self, dims, hidden):
        super().__init__()
        self.lstm = nn.LSTM(dims, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(hidden*2, hidden*2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.attn = nn.MultiheadAttention(hidden*2, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

    def forward(self, x):
        o, _ = self.lstm(x)
        c = self.conv(o.transpose(1,2)).transpose(1,2)
        p = self.pool(c.transpose(1,2)).squeeze(-1)
        a, _ = self.attn(o, o, o)
        out = torch.cat([p, a.mean(1)], dim=-1)
        return self.fc(out)

class PhonologicalEmbedding(nn.Module):
    def __init__(self, num_init=21, num_fin=57, num_tone=6, emb=64, hidden=128):
        super().__init__()
        self.init_emb = nn.Embedding(num_init, emb)
        self.fin_emb = nn.Embedding(num_fin, emb)
        self.tone_emb = nn.Embedding(num_tone, emb)
        self.encoder = PhonoEncoder(emb*3, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(0.3)

    def forward(self, init, fin, tone):
        f = self.fin_emb(fin)
        t = self.tone_emb(tone)
        x = torch.cat([i,f,t], dim=-1).unsqueeze(1)
        out = self.encoder(x)
        return self.drop(self.norm(out))

def test():
    model = PhonologicalEmbedding()
    b = 16
    init = torch.randint(0,21,(b,))
    fin = torch.randint(0,57,(b,))
    tone = torch.randint(0,6,(b,))
    o = model(init, fin, tone)
    assert o.shape == (b, 128)

if __name__ == "__main__":
    test()
