import torch
import torch.nn as nn
from transformers import AutoModel

class LoRALinear(nn.Module):
    def __init__(self, orig: nn.Linear, r=16):
        super().__init__()
        self.orig = orig
        self.r = r
        self.A = nn.Parameter(torch.randn(orig.in_features, r) * 0.01)
        self.B = nn.Parameter(torch.randn(r, orig.out_features) * 0.01)
        self.scaling = orig.in_features ** -0.5

    def forward(self, x):
        return self.orig(x) + (x @ self.A @ self.B) * self.scaling

class LoRAFineTuner(nn.Module):
    def __init__(self, backbone_name: str, r=16, num_labels=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        for name, m in self.backbone.named_modules():
            if isinstance(m, nn.Linear):
                parent, attr = self._find(name)
                setattr(parent, attr, LoRALinear(m, r))
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.Tanh(),
            nn.Linear(hidden//2, num_labels)
        )

    def _find(self, target):
        for module in self.backbone.modules():
            for k, v in module.__dict__.get('._modules', {}).items():
                if v is getattr(self.backbone, target.split('.')[-1], None):
                    return module, k
        raise RuntimeError

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state[:,0]
        return self.classifier(h)

def test():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = LoRAFineTuner("bert-base-chinese")
    x = tok(["测试LoRA"], return_tensors="pt", padding=True)
    o = model(x["input_ids"], x["attention_mask"])
    assert o.shape[-1] == 2

if __name__ == "__main__":
    test()
