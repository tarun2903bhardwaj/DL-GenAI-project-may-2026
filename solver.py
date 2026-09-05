"""
Smart MCQ Solver - Inference Engine
Combines:
1. Preamble Stripping & RapidFuzz Fuzzy Lookup (Hard override >=95, Soft boost 80-94)
2. BM25 Classical IR Ranker (From-scratch baseline with embedded fallback)
3. LoRA RoBERTa-base Cross-Encoder (3-Fold CV)
4. LoRA ALBERT-base-v2 Cross-Encoder
5. Weighted Ensemble (BM25 10%, RoBERTa 65%, ALBERT 25%)
"""

import os
import re
import math
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# Classical IR BM25 with pure-Python fallback
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:
        """Pure-Python fallback implementation of BM25Okapi."""
        def __init__(self, corpus, k1=1.5, b=0.75):
            self.k1 = k1
            self.b = b
            self.corpus = corpus
            self.corpus_size = len(corpus)
            self.avgdl = sum(len(x) for x in corpus) / max(self.corpus_size, 1)
            self.doc_freqs = []
            self.idf = {}
            self.doc_len = [len(x) for x in corpus]
            df = {}
            for doc in corpus:
                frequencies = {}
                for word in doc:
                    frequencies[word] = frequencies.get(word, 0) + 1
                self.doc_freqs.append(frequencies)
                for word in frequencies.keys():
                    df[word] = df.get(word, 0) + 1
            for word, freq in df.items():
                self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

        def get_scores(self, query):
            score = [0.0] * self.corpus_size
            for q in query:
                q_idf = self.idf.get(q, 0.0)
                for i, doc_f in enumerate(self.doc_freqs):
                    freq = doc_f.get(q, 0)
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_len[i] / max(self.avgdl, 1e-6)))
                    score[i] += q_idf * (numerator / max(denominator, 1e-6))
            return score

CHOICES = list('ABCDE')
LABEL2IDX = {c: i for i, c in enumerate(CHOICES)}
IDX2LABEL = {i: c for c, i in LABEL2IDX.items()}

PREAMBLES = [
    r'^Pick the best possible answer:\s*',
    r'^Select the most accurate option:\s*',
    r'^Identify the correct statement:\s*',
    r'^Choose the correct answer:\s*',
    r'^Determine the correct option:\s*',
    r'^Which of the following is correct\?\s*',
    r'\s*among the listed options\.?\s*$',
    r'\s*from the following choices\.?\s*$',
    r'\s*carefully\.?\s*$',
]

def strip_preambles(text: str) -> str:
    """Remove instructional preambles and suffixes to isolate the core question."""
    text = str(text).strip()
    for p in PREAMBLES:
        text = re.sub(p, '', text, flags=re.IGNORECASE).strip()
    return text

def tokenize_bm25(text: str) -> list:
    """Lowercase, strip non-alphanumeric, and split into tokens."""
    return re.sub(r'[^a-z0-9\s]', ' ', str(text).lower()).split()

BORDA_RANK_WEIGHTS = [0.50, 0.30, 0.15, 0.04, 0.01]

def bm25_to_prob(top3_letters: list) -> np.ndarray:
    """Convert BM25 ranked letters to pseudo-probability vector."""
    prob = np.zeros(5)
    top3_idxs = set()
    for rank, letter in enumerate(top3_letters):
        idx = LABEL2IDX[letter]
        prob[idx] = BORDA_RANK_WEIGHTS[rank]
        top3_idxs.add(idx)
    for c in CHOICES:
        if LABEL2IDX[c] not in top3_idxs:
            prob[LABEL2IDX[c]] = BORDA_RANK_WEIGHTS[3]
    return prob / prob.sum()

def softmax_np(x):
    """Numerically stable row-wise softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


class SmartMCQSolver:
    """
    Modular solver that implements the complete competition pipeline.
    Gracefully handles environments with or without deep learning weights.
    """
    def __init__(self, train_csv_path='data/train.csv', models_dir='models'):
        self.train_csv_path = train_csv_path
        self.models_dir = models_dir
        self.train_df = None
        self.train_cores = []
        self.train_labels = []
        self.train_rows = []
        
        # Load train reference database if available
        if os.path.exists(train_csv_path):
            self.train_df = pd.read_csv(train_csv_path)
            self.train_df['core'] = self.train_df['prompt'].apply(strip_preambles)
            self.train_cores = self.train_df['core'].tolist()
            self.train_labels = self.train_df['answer'].tolist()
            self.train_rows = self.train_df.to_dict('records')
        
        # Check for PyTorch & LoRA model checkpoints
        self.has_torch = False
        self.device = 'cpu'
        self.roberta_models = []
        self.albert_model = None
        self.roberta_tokenizer = None
        self.albert_tokenizer = None
        
        self._init_models()

    def _init_models(self):
        """Attempt to initialize PyTorch and load LoRA weights if files exist."""
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoTokenizer, AutoModel
            from peft import get_peft_model, LoraConfig, TaskType
            
            self.has_torch = True
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Check for saved weights
            roberta_paths = [
                os.path.join(self.models_dir, f'roberta_fold{i}.pt')
                for i in range(1, 4)
                if os.path.exists(os.path.join(self.models_dir, f'roberta_fold{i}.pt'))
            ]
            albert_path = os.path.join(self.models_dir, 'albert_lora.pt')
            
            if roberta_paths or os.path.exists(albert_path):
                class LoRAMCQModel(nn.Module):
                    def __init__(self, model_name, lora_r=16, lora_alpha=32):
                        super().__init__()
                        base = AutoModel.from_pretrained(model_name)
                        peft_config = LoraConfig(
                            task_type=TaskType.FEATURE_EXTRACTION,
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            lora_dropout=0.05,
                            target_modules=['query', 'value']
                        )
                        self.encoder = get_peft_model(base, peft_config)
                        self.dropout = nn.Dropout(0.1)
                        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

                    def forward(self, input_ids, attention_mask):
                        B, N, L = input_ids.shape
                        ids = input_ids.view(B * N, L)
                        mask = attention_mask.view(B * N, L)
                        out = self.encoder(input_ids=ids, attention_mask=mask)
                        cls = out.last_hidden_state[:, 0, :]
                        cls = cls.to(self.classifier.weight.dtype)
                        logits = self.classifier(self.dropout(cls)).view(B, N)
                        return logits

                if roberta_paths:
                    self.roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
                    for p in roberta_paths:
                        m = LoRAMCQModel('roberta-base').to(self.device)
                        m.load_state_dict(torch.load(p, map_location=self.device))
                        m.eval()
                        self.roberta_models.append(m)

                if os.path.exists(albert_path):
                    self.albert_tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
                    m = LoRAMCQModel('albert-base-v2').to(self.device)
                    m.load_state_dict(torch.load(albert_path, map_location=self.device))
                    m.eval()
                    self.albert_model = m
                    
        except Exception as e:
            pass

    def score_bm25(self, core_q: str, options: dict) -> tuple:
        """Score options with BM25 Okapi baseline."""
        q_tokens = tokenize_bm25(core_q)
        corpus = [tokenize_bm25(str(options.get(c, ''))) for c in CHOICES]
        bm25 = BM25Okapi(corpus)
        raw_scores = np.array(bm25.get_scores(q_tokens))
        top3_letters = [IDX2LABEL[i] for i in np.argsort(raw_scores)[::-1][:3]]
        return raw_scores, top3_letters

    def predict_single(self, prompt: str, options: dict) -> dict:
        """
        Predict top-3 options for a single question.
        Returns:
            dict containing 'top3', 'strategy', 'confidence', 'breakdown', 'core_prompt'
        """
        core_q = strip_preambles(prompt)
        
        # 1. Fuzzy Matching against Reference Bank
        fuzzy_match_info = None
        if self.train_cores:
            res = process.extractOne(core_q, self.train_cores, scorer=fuzz.token_set_ratio)
            if res:
                score, match_idx = res[1], res[2]
                matched_row = self.train_rows[match_idx]
                train_correct_letter = matched_row['answer']
                train_correct_text = str(matched_row.get(train_correct_letter, '')).strip()
                
                # Check option text similarity to account for option permutations
                best_option_by_text = None
                best_text_sim = 0
                for c in CHOICES:
                    opt_text = str(options.get(c, '')).strip()
                    sim = fuzz.ratio(train_correct_text, opt_text)
                    if sim > best_text_sim:
                        best_text_sim = sim
                        best_option_by_text = c
                
                # Determine chosen letter
                resolved_letter = best_option_by_text if best_text_sim >= 85 else train_correct_letter
                
                fuzzy_match_info = {
                    'score': score,
                    'matched_core': self.train_cores[match_idx],
                    'train_answer_letter': train_correct_letter,
                    'resolved_letter': resolved_letter,
                    'text_match_sim': best_text_sim
                }
                
                # Hard Override (>= 95 similarity)
                if score >= 95:
                    remaining = [c for c in CHOICES if c != resolved_letter]
                    # Sort remaining using BM25
                    _, bm25_top3 = self.score_bm25(core_q, options)
                    ordered_remaining = [c for c in bm25_top3 if c != resolved_letter]
                    top3 = [resolved_letter] + ordered_remaining[:2]
                    
                    return {
                        'top3': top3,
                        'strategy': 'Fuzzy Lookup Override (Similarity >= 95%)',
                        'confidence': round(score / 100.0, 4),
                        'fuzzy_info': fuzzy_match_info,
                        'bm25_top3': bm25_top3,
                        'ensemble_used': False,
                        'core_prompt': core_q
                    }

        # 2. BM25 Scoring
        bm25_scores, bm25_top3 = self.score_bm25(core_q, options)
        p_bm25 = bm25_to_prob(bm25_top3)
        
        # 3. LoRA Transformer Scoring (if weights present)
        p_roberta = None
        p_albert = None
        
        if self.has_torch and (self.roberta_models or self.albert_model):
            import torch
            try:
                if self.roberta_models and self.roberta_tokenizer:
                    input_ids_rob, mask_rob = [], []
                    for c in CHOICES:
                        ans_text = str(options.get(c, ''))
                        enc = self.roberta_tokenizer(
                            f'Question: {core_q}', f'Answer: {ans_text}',
                            truncation='longest_first', max_length=256,
                            padding='max_length', return_tensors='pt'
                        )
                        input_ids_rob.append(enc['input_ids'].squeeze(0))
                        mask_rob.append(enc['attention_mask'].squeeze(0))
                    
                    t_ids_rob = torch.stack(input_ids_rob).unsqueeze(0).to(self.device)
                    t_mask_rob = torch.stack(mask_rob).unsqueeze(0).to(self.device)
                    
                    r_logits = np.zeros(5)
                    with torch.no_grad():
                        for m in self.roberta_models:
                            l = m(t_ids_rob, t_mask_rob).cpu().numpy()[0]
                            r_logits += l / len(self.roberta_models)
                    p_roberta = softmax_np(r_logits)
                    
                if self.albert_model and self.albert_tokenizer:
                    input_ids_alb, mask_alb = [], []
                    for c in CHOICES:
                        ans_text = str(options.get(c, ''))
                        enc = self.albert_tokenizer(
                            f'Question: {core_q}', f'Answer: {ans_text}',
                            truncation='longest_first', max_length=256,
                            padding='max_length', return_tensors='pt'
                        )
                        input_ids_alb.append(enc['input_ids'].squeeze(0))
                        mask_alb.append(enc['attention_mask'].squeeze(0))
                    t_ids_alb = torch.stack(input_ids_alb).unsqueeze(0).to(self.device)
                    t_mask_alb = torch.stack(mask_alb).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        a_logits = self.albert_model(t_ids_alb, t_mask_alb).cpu().numpy()[0]
                    p_albert = softmax_np(a_logits)
            except Exception as e:
                pass

        # 4. Ensemble Blending
        # Weights: BM25: 10%, RoBERTa: 65%, ALBERT: 25%
        if p_roberta is not None and p_albert is not None:
            p_final = 0.10 * p_bm25 + 0.65 * p_roberta + 0.25 * p_albert
            strategy_name = 'Full Ensemble (BM25 10% + RoBERTa 65% + ALBERT 25%)'
        elif p_roberta is not None:
            p_final = 0.20 * p_bm25 + 0.80 * p_roberta
            strategy_name = 'Ensemble (BM25 20% + RoBERTa 80%)'
        else:
            p_final = p_bm25
            strategy_name = 'BM25 Classical IR Baseline'

        # Soft boost (80 <= similarity < 95)
        if fuzzy_match_info and 80 <= fuzzy_match_info['score'] < 95:
            boost_idx = LABEL2IDX[fuzzy_match_info['resolved_letter']]
            p_final[boost_idx] += 0.08
            p_final = p_final / p_final.sum()
            strategy_name += ' + Fuzzy Soft Boost'

        top3_indices = np.argsort(p_final)[::-1][:3]
        top3 = [IDX2LABEL[i] for i in top3_indices]
        confidence = float(p_final[top3_indices[0]])

        return {
            'top3': top3,
            'strategy': strategy_name,
            'confidence': round(confidence, 4),
            'fuzzy_info': fuzzy_match_info,
            'probabilities': {c: round(float(p_final[i]), 4) for i, c in enumerate(CHOICES)},
            'ensemble_used': True,
            'core_prompt': core_q
        }

    def predict_dataframe(self, df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        """
        Run inference over an entire DataFrame and produce Kaggle submission format.
        """
        results = []
        total = len(df)
        for idx, row in df.iterrows():
            q_id = row.get('id', idx)
            prompt = str(row['prompt'])
            options = {c: str(row.get(c, '')) for c in CHOICES}
            
            pred = self.predict_single(prompt, options)
            results.append({
                'ID': q_id,
                'Prediction': ' '.join(pred['top3']),
                'Top_1': pred['top3'][0],
                'Strategy': pred['strategy'],
                'Confidence': pred['confidence']
            })
            if progress_callback:
                progress_callback(len(results) / total)
                
        return pd.DataFrame(results)
