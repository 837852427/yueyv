import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))
        self.scale = dim ** -0.5

    def forward(self, x, mask=None):
        scores = torch.matmul(x, self.query) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)

class SemanticEmbedding(nn.Module):
    def __init__(self, pretrained_model_name: str, aggregation: str = "mean", freeze_backbone: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        hidden = self.backbone.config.hidden_size
        self.aggregation = aggregation
        self.attn_pool = AttentionPooling(hidden)
        self.proj1 = nn.Linear(hidden, hidden // 2)
        self.act = nn.GELU()
        self.proj2 = nn.Sequential(
            nn.Linear(hidden // 2, hidden),
            nn.LayerNorm(hidden),
            nn.Dropout(0.2)
        )
        self.cache = {}

    def aggregate(self, x, mask):
        if self.aggregation == "mean":
            return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        if self.aggregation == "max":
            m = x.masked_fill(~mask.unsqueeze(-1), -1e9)
            return m.max(1)[0]
        return self.attn_pool(x, mask)

    def forward(self, texts: list[str]):
        batch = len(texts)
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens["input_ids"]
        mask = tokens["attention_mask"].bool()
        key = tuple(input_ids.flatten().tolist())
        if key in self.cache:
            emb = self.cache[key]
        else:
            out = self.backbone(input_ids, attention_mask=mask)
            last = out.last_hidden_state
            emb = self.aggregate(last, mask)
            self.cache[key] = emb
        h1 = self.proj1(emb)
        a1 = self.act(h1)
        out = self.proj2(a1)
        return out

def test():
    model = SemanticEmbedding("bert-base-chinese", aggregation="attn", freeze_backbone=False)
    x = model(["测试一下", "又一条文本"])
    assert x.shape[1] == model.backbone.config.hidden_size

if __name__ == "__main__":
    test()
