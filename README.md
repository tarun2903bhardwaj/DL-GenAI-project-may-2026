# Smart MCQ Solver

**Author:** Tarun Bhardwaj  
**Roll No:** 24f1002384  
**Competition:** Smart MCQ Solver Challenge (DL / Gen-AI)  
**Notebook ID:** `DL-24f1002384-notebook-t22026`  
**Evaluation Metric:** Mean Average Precision @ 3 (MAP@3)  

**Live Web Application:** [https://dl-genai-project-may-2026-fehzhelckazxabjvty8bcb.streamlit.app/](https://dl-genai-project-may-2026-fehzhelckazxabjvty8bcb.streamlit.app/)

---

## 1. Project Overview

The objective of this project is to predict the top-3 ranked answer options for 5-choice multiple-choice questions (MCQs) evaluated under the MAP@3 competition metric.

The system combines:
1. Preamble-stripped fuzzy string retrieval.
2. A from-scratch information retrieval baseline (BM25 Okapi).
3. Two parameter-efficient fine-tuned transformer cross-encoders (RoBERTa-base and ALBERT-base-v2 via LoRA).
4. A weighted probability ensemble with hard and soft lookup overrides.

---

## 2. Key EDA Findings

- **Preamble Artifacts:** Training prompts were wrapped in 6 recurring instructional templates (e.g., *"Pick the best possible answer:"*, *"Select the most accurate option:"*), accounting for ~25% of prompt token length.
- **Question Duplication:** Stripping preambles revealed that the 2,000 training rows contain only **419 unique core questions** (repeated up to 17 times each). All duplicates had 100% consistent ground-truth labels.
- **Test Set Overlap:** 485 out of 500 test questions (97%) were exact core matches to training questions. RapidFuzz `token_set_ratio` matched **500/500 (100%)** test questions at $\ge 95\%$ similarity.
- **Option Length Bias:** Correct options averaged 28.7 words vs 25.7 words for incorrect options, ruling out trivial length-based heuristics.

---

## 3. Model Architectures & Results

| Model | Architecture | Type | Parameters Trained | Best Validation MAP@3 | Ensemble Weight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1** | **BM25 Okapi** | Classical IR (From Scratch) | 0 (no neural weights) | **0.3704** | 10% |
| **Model 2** | **LoRA RoBERTa-base** | Cross-Encoder (Pretrained 1) | ~600K (0.5% of total) | **0.9154** (3-Fold CV Mean) | 65% |
| **Model 3** | **LoRA ALBERT-base-v2** | Cross-Encoder (Pretrained 2) | ~300K (2.5% of total) | **0.9912** (Val split) | 25% |
| **Lookup** | **RapidFuzz `token_set_ratio`** | Text Retrieval | N/A | **1.0000** (500/500 coverage) | Priority Override |

*Note: DeBERTa-v3 was dropped after producing NaN training loss across all folds in V1.*

### Detailed Validation Breakdown:
- **LoRA RoBERTa (3-Fold Stratified CV):**
  - Fold 1: 0.8751
  - Fold 2: 0.9093
  - Fold 3: 0.9617
  - Mean CV MAP@3: **0.9154**
  - Inference: 3 fold checkpoints $\times$ 2 Test-Time Augmentation (TTA) prompt templates = 6 forward passes.
- **LoRA ALBERT (80/20 Stratified Split):**
  - Epoch 1: 0.8933 | Epoch 2: 0.9629 | Epoch 3: **0.9912**

---

## 4. Decision & Ensemble Pipeline

For each test question:
1. **Preamble Stripping:** Regular expressions remove prefixes (*"Pick the best..."*) and trailing clauses (*"...carefully"*).
2. **Priority 1 — High-Confidence Fuzzy Override:** If RapidFuzz similarity $\ge 95\%$, the matched training answer is assigned to Rank 1. Remaining ranks are filled using BM25 option scores.
3. **Priority 2 — Probability Ensemble:** If no high match exists, probabilities are blended:
   $$\text{Score} = 0.10 \times P_{\text{BM25}} + 0.65 \times P_{\text{RoBERTa}} + 0.25 \times P_{\text{ALBERT}}$$
4. **Priority 3 — Soft Boost:** For similarities between 80% and 94%, a $+0.08$ boost is added to the candidate class before final ranking.

---

## 5. Repository Structure

```
├── app.py                              # Streamlit web application
├── solver.py                           # Standalone inference engine
├── requirements.txt                    # Python deployment dependencies
├── SmartMCQ_Report_Tarun.pdf           # Project report (PDF)
├── SmartMCQ_Report_Tarun.docx          # Project report (Word)
├── DL-24f1002384-notebook-t22026       # Kaggle training notebook (saved checkpoint)
├── dl-24f1002384-notebook-t22026.ipynb # Kaggle notebook with outputs
├── check_reorder.py                    # Option permutation verification script
└── data/
    ├── train.csv                       # 2,000 reference question bank
    ├── test.csv                        # 500 test questions
    └── sample_submission.csv           # Submission template
```
