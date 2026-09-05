# 🧠 Smart MCQ Solver — Implementation Plan (Updated for Milestone 2)
**Competition**: Smart MCQ Solver | **Metric**: MAP@3 | **Cutoff**: ≥ 0.73 | **Roll No**: 24f1002384

> ⚠️ **No external API keys**. All models run locally on Kaggle T4 GPU.

---

## 📋 Current Status

| Milestone | Status | Deadline |
|---|---|---|
| M0: Setup | ✅ Done | Jun 22 |
| M1: TF-IDF + Word2Vec | ✅ Done | Jun 24 |
| **M2: Enter the Transformers** | **🔄 In Progress** | **Jul 1** |
| M3: RAG Pipelines | ⬜ Pending | Jul 8 |
| M4: Fine-Tuning + LoRA | ⬜ Pending | Jul 15 |
| M5: Ensembling | ⬜ Pending | Jul 21 |

---

## 🎯 Milestone 2 — Exact Steps to Complete

### Pre-requisites (One-time Setup)

You need 3 accounts with API keys stored as Kaggle Secrets:

| Platform | Account URL | Kaggle Secret Name | How to Get Key |
|---|---|---|---|
| **W&B** | https://wandb.ai | `WANDB_API_KEY` | Settings → Danger Zone → API keys |
| **Hugging Face** | https://huggingface.co | `HF_TOKEN` | Settings → Access Tokens → New Token (read) |
| **GitHub** | https://github.com | *(push from local)* | Not needed as Kaggle Secret |

#### Step-by-step setup:
1. **Kaggle Secrets**: Go to your Kaggle notebook → ⚙️ Settings → Secrets → Add:
   - Name: `WANDB_API_KEY`, Value: *(your W&B key)*
   - Name: `HF_TOKEN`, Value: *(your HF read token)*
2. **Enable Internet**: Settings → Internet → On
3. **Enable GPU**: Settings → Accelerator → GPU T4 x1
4. **Attach Dataset**: Add the competition dataset to your notebook

---

### Milestone 2 — Question-by-Question Breakdown

The notebook is structured so each cell directly answers one milestone question:

| Cell | Topic | Question Summary | Key Concept |
|---|---|---|---|
| **2.1** | HF datasets | Load train.csv via `datasets.load_dataset()`, create `combined_text` col, compute `len()` at index 51 | `.map()` function |
| **2.2** | Tokenizer vocab | Load `bert-base-uncased` tokenizer, extract `vocab_size` | Tokenizer config |
| **2.3** | [SEP] token ID | Get integer ID of `[SEP]` token | `convert_tokens_to_ids()` |
| **2.4** | Batch tokenization | Tokenize entire `prompt` column with padding/truncation, get `input_ids` shape | `torch.Size([2000, 128])` |
| **2.5** | Attention heads | Compute dimension per head: `768 / 12` | Transformer architecture |
| **2.6** | BERT output shape | Tokenize row index 0 prompt, pass through model, get `last_hidden_state.shape` | `AutoModel` forward pass |
| **2.7** | [CLS] embedding | Extract [CLS] vector (index 0), sum first 5 values | Feature extraction |
| **2.8** | Attention weight | Load BERT with `output_attentions=True`, find attention from [CLS] to "fusion" token | Attention matrix |
| **2.9** | Sentence similarity | Use `all-MiniLM-L6-v2`, encode prompt + option B at index 0, compute `cos_sim()` | Sentence-BERT |
| **2.10** | Full pipeline comparison | TF-IDF vs MiniLM MAP@3 on all train, count improvement | Ranking pipeline |
| **2.11** | Zero-shot (softmax) | `bart-large-mnli` on index 1 prompt with A/B/C as labels | `pipeline("zero-shot-classification")` |
| **2.12** | Zero-shot (sigmoid) | Same but `multi_label=True`, compute difference in prob sums | Softmax vs Sigmoid |
| **2.13** | Generative SLM | `flan-t5-small` text2text-generation on row 0 | `pipeline("text2text-generation")` |

---

## 📊 Dataset Facts (Quick Reference)

| Property | Value |
|---|---|
| Train rows | 2,000 (IDs 1–2000) |
| Test rows | 500 (IDs 1–500) |
| Columns (train) | id, prompt, A, B, C, D, E, answer |
| Zero-indexed row 0 | id=1, Martin Heidegger question |
| Zero-indexed row 1 | id=2, Accelerator-based light-ion fusion |
| Zero-indexed row 51 | id=52 |

---

## 🏗️ Three Required Models (Overall Project)

| # | Model | Type | Milestone |
|---|---|---|---|
| 1 | TF-IDF + Word2Vec Cosine Ranker | From scratch | M1 ✅ |
| 2 | DeBERTa-v3-base Fine-Tuned MCQ | Pretrained | M4 |
| 3 | RoBERTa + LoRA | Additional | M4 |

---

## 📁 Project File Structure

```
DL_Gen_AI_Quiz_solver/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── milestone1_tfidf_w2v.ipynb     (M1)
│   └── milestone2_transformers.ipynb  (M2) ← CURRENT
├── mcq_solver_kaggle.ipynb            (Main competition notebook)
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
└── README.md
```

---

## 📈 W&B Experiment Tracking Plan

| Run Name | Model | Metrics |
|---|---|---|
| `model1_tfidf_w2v` | TF-IDF + Word2Vec | val_map3, top1_acc |
| `m2_minilm_pipeline` | all-MiniLM-L6-v2 | map3, improvement_count |
| `model2_deberta` | DeBERTa fine-tuned | train_loss, val_map3, fold |
| `model3_lora` | RoBERTa + LoRA | train_loss, val_map3 |

---

## 🔧 Tech Stack (No External API Keys)

```
# All run locally on Kaggle T4 GPU
datasets              # Hugging Face datasets library
transformers>=4.40    # BERT, RoBERTa, BART, Flan-T5
sentence-transformers # all-MiniLM-L6-v2
torch>=2.1
scikit-learn>=1.3     # TF-IDF, cosine_similarity
wandb                 # Experiment tracking
```

---

## ⏱️ Milestone 2 Runtime Estimate (Kaggle T4)

| Step | Time |
|---|---|
| HF datasets loading + map | 1 min |
| BERT tokenization (2000 rows) | 2 min |
| BERT forward pass + attention | 1 min |
| Sentence-BERT embeddings (all train) | 5–8 min |
| Full MAP@3 pipeline comparison | 8–12 min |
| Zero-shot classification | 2 min |
| Flan-T5 generation | 1 min |
| **Total** | **~20–30 min** |
