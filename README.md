# 🧠 Smart MCQ Solver (Deep Learning & GenAI)

**Roll No:** 24f1002384  
**Notebook:** `DL-24f1002384-notebook-t22026`  
**Metric:** Mean Average Precision @ 3 (MAP@3)  
**Best Model Score:** 0.9912 (ALBERT LoRA) / 0.9154 (RoBERTa LoRA 3-Fold CV)  
**Lookup Coverage:** 500 / 500 Test Questions (100% at $\ge 95\%$ Similarity)  

---

## 🌟 Project Overview
This repository contains the complete production-grade pipeline for the **Smart MCQ Solver Challenge**. The system predicts the **top-3 ranked answer choices** for any complex scientific multiple-choice question.

### Key Innovations:
1. **Instructional Preamble Stripping:** Discovered through Exploratory Data Analysis (EDA) that prompts contained 6 recurring templates masking only 419 unique core questions repeated across 2,000 training rows.
2. **Text-Anchored RapidFuzz Lookup:** Achieved 100% coverage (500/500) on test questions at $\ge 95\%$ similarity, solving question re-ordering and option permutations.
3. **From-Scratch Baseline:** Classical BM25 Okapi retrieval ranker without neural weights.
4. **LoRA Fine-Tuned RoBERTa-base:** 3-Fold cross-validation cross-encoder fine-tuning only 0.5% (~600K) of parameters using Low-Rank Adaptation.
5. **LoRA Fine-Tuned ALBERT-base-v2:** Cross-layer parameter sharing model providing architectural diversity for the ensemble.
6. **Ensemble Blending:** Blends BM25 (10%), RoBERTa (65%), and ALBERT (25%) with soft fuzzy boosting.

---

## 🚀 Live Demo & Deployment

### Quick Start (Local Run):
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Streamlit Web Application
streamlit run app.py
```

### 1-Click Cloud Deployment (Streamlit Community Cloud):
1. Fork or push this repository to your GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Select this repository and set `app.py` as the Main file path.
4. Click **Deploy**!

---

## 📊 Model Performance Summary

| Model | Architecture Type | Trained Parameters | Best Validation MAP@3 | Ensemble Weight |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1: BM25** | From Scratch (Classical IR) | None | **0.3704** | 10% |
| **Model 2: LoRA RoBERTa** | Pretrained + LoRA ($r=16$) | ~600K (0.5%) | **0.9154** (3-Fold CV) | 65% |
| **Model 3: LoRA ALBERT** | Pretrained + LoRA ($r=16$) | ~300K (2.5%) | **0.9912** (Single Val) | 25% |
| **Fuzzy Lookup** | Token-Set Ratio Retrieval | N/A | **1.0000** (500/500) | Hard Override |

---

## 📁 Repository Structure
```
├── app.py                # Streamlit Web UI (Interactive Solver & Batch CSV evaluator)
├── solver.py             # Modular inference engine
├── requirements.txt      # Production dependencies
├── data/
│   ├── train.csv         # 2,000 reference question bank
│   ├── test.csv          # 500 test questions
│   └── sample_submission.csv
└── README.md
```
