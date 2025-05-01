# PGATA: Phonology-Glyph-Aware Token Aggregation Framework

![img](file:///C:\Users\zsh\AppData\Local\Temp\05dd7acc-2520-488e-833d-c81ccc59ad86.png)

The PGATA framework integrates semantic, phonological, and glyph-level representations through hierarchical embedding modules, followed by feature fusion, residual enhancement, and LoRA-based fine-tuning. It supports context-aware modeling for downstream tasks such as sarcasm detection and intent classification.

------

## Dependencies

- Python >= 3.9
- PyTorch >= 2.6.0

------

## Usage

**Step1**: Navigate to the core code directory that contains the main PGATA pipeline script and component modules.

```bash
cd code
```

**Step2**: Run the PGATA pipeline with default datasets.

```bash
python PGATA.py
```

Alternatively, users may modify its configuration to fit specific requirements, as shown below:

```bash
python PGATA.py \
--backbone_model "bert-base-chinese" \
--batch_size 8 \
--freeze_backbone False \
--semantic_pooling "attn" \
--num_radicals 214 \
--stroke_size 16 \
--phoneme_emb_dim 64 \
--glyph_output_dim 128
```

## Configuration Explanation:

1. **`--backbone_model`**
    The name of the semantic encoder model, supporting any pretrained transformer from HuggingFace (e.g., `bert-base-chinese`).

2. **`--batch_size`**
    Number of examples to be processed in each forward pass.

3. **`--freeze_backbone`**
    Whether to freeze the parameters of the semantic encoder during training.

4. **`--semantic_pooling`**
    Aggregation strategy for semantic representation (supports `mean`, `max`, or `attn`).

5. **`--num_radicals`**
    The number of distinct radicals used in the glyph embedding module.

6. **`--stroke_size`**
    Height and width of stroke images (assumed square, e.g., 16×16).

7. **`--phoneme_emb_dim`**
    Dimensionality of each phoneme (initial/final/tone) embedding vector.

8. **`--glyph_output_dim`**
    Final output dimension of glyph embedding after fusion of radicals and strokes.

## Project Structure

```bash
├── code
│   ├── feature_fusion_and_projection.py    
│   ├── glyph_embedding.py                  
│   ├── lora_fine_tuning.py                 
│   ├── phonological_embedding.py           
│   ├── residual_connection.py              
│   ├── semantic_embedding.py               
│   └── PGATA.py                            
├── PGATA Framework.png                     
├── README.md
```

