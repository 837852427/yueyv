# PGATA.py
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from semantic_embedding import SemanticEmbedding
from phonological_embedding import PhonologicalEmbedding
from glyph_embedding import GlyphEmbedding
from feature_fusion_and_projection import FeatureFusionAndProjection
from residual_connection import ResidualConnection
from lora_fine_tuning import LoRAFineTuner

def run_pipeline(texts, initials, finals, tones, radicals, strokes):
    sem = SemanticEmbedding("bert-base-chinese")(texts)
    pho = PhonologicalEmbedding()(initials, finals, tones)
    gly = GlyphEmbedding()(radicals, strokes)
    fused = FeatureFusionAndProjection(sem.shape[-1], pho.shape[-1], gly.shape[-1])(sem, pho, gly)
    resid = ResidualConnection(fused.shape[-1])(fused)
    tok = torch.randint(0, 100, (len(texts), 16))
    mask = torch.ones_like(tok)
    tuned = LoRAFineTuner("bert-base-chinese")(tok, mask)
    return resid, tuned

if __name__ == "__main__":
    cscd_ds  = load_dataset("my_org/CSCD-NS-CAN", split="train")
    lemon_ds = load_dataset("my_org/LEMON-CAN",  split="train")
    web_ds   = load_dataset("my_org/WebSarcasm-CAN", split="train")

    batch_size = 4
    texts_cscd  = cscd_ds["text"][:batch_size]
    texts_lemon = lemon_ds["text"][:batch_size]
    texts_web   = web_ds["text"][:batch_size]

    texts = texts_cscd + texts_lemon + texts_web

    b = len(texts)
    initials = torch.randint(0, 21, (b,))
    finals   = torch.randint(0, 57, (b,))
    tones    = torch.randint(0, 6,  (b,))
    radicals = torch.randint(0, 214, (b,))
    strokes  = torch.randn(b, 1, 16, 16)

    resid_out, tuned_out = run_pipeline(texts, initials, finals, tones, radicals, strokes)
    print("Residual output shape:", resid_out.shape)
    print("LoRA-tuned output shape:", tuned_out.shape)
