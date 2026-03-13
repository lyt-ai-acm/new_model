# Chinese Homophone Normalization + Sentiment Analysis — Project Specification

## Overview

This project implements an end-to-end NLP pipeline for Chinese social media text that:
1. Detects and normalizes homophone substitutions (e.g., "辣鸡" → "垃圾")
2. Performs 3-class sentiment analysis (pos / neu / neg)
3. Fuses Top-10 normalized candidates for robust sentiment prediction

## System Architecture

```
Raw Text
   │
   ▼
[Tokenizer] (jieba word segmentation)
   │
   ▼
[SuspiciousWordDetector] (scores tokens for homophone likelihood)
   │
   ▼
[CandidateGenerator] (fetches top-10 homophones per suspicious token)
   │
   ▼
[BeamSearchExpander] (beam=50, expands candidate sentences)
   │
   ▼
[CandidateRanker] (LM score + prior + edit cost → Top-10 with softmax weights)
   │
   ▼
[SentimentPredictor] (RoBERTa or RBMA → [P(neg), P(neu), P(pos)] per candidate)
   │
   ▼
[SentimentFuser] (weighted or mean fusion → final label)
```

## Module Details

### lm/kenlm_scorer.py
- **KenLMScorer**: wraps KenLM binary model; falls back to Laplace-smoothed unigram scoring
- Input: tokenized list of strings
- Output: normalized log-probability score

### homophone/
| Module | Class | Purpose |
|--------|-------|---------|
| tokenizer.py | Tokenizer | jieba/char tokenization |
| detect.py | SuspiciousWordDetector | Score tokens: 2×has_candidate + 1.5×low_freq + 1×context + 0.5×shape |
| candidates.py | CandidateGenerator | Lookup homophone dictionary, optionally filter by high-freq vocab |
| beam.py | BeamSearchExpander | Beam search over suspicious positions (beam_size=50) |
| rank.py | CandidateRanker | Score = α·LM + β·Prior − λ·EditCost; apply δ threshold |
| normalize.py | HomophoneNormalizer | Full pipeline → Top-K (sentence, weight) pairs |

### sentiment/
| Module | Class | Purpose |
|--------|-------|---------|
| model_roberta.py | RoBERTaSentimentClassifier | Chinese RoBERTa + linear head |
| model_rbma.py | RBMASentimentClassifier | RoBERTa + BiLSTM + MultiheadAttention |
| train.py | - | Training loop with AdamW + linear warmup |
| predict.py | SentimentPredictor | Batch inference → probabilities + labels |

### fusion/fuse.py
- **SentimentFuser**: `mean` or `weighted` fusion over candidate probability distributions

### eval/
- **metrics.py**: Accuracy, Macro-F1, per-class P/R/F1
- **confusion.py**: 3×3 confusion matrix (matplotlib/seaborn)

### pipeline/run_infer.py
- End-to-end inference: loads all components, runs normalization + sentiment fusion

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| m | 2 | Max suspicious positions per sentence |
| beam_size | 50 | Beam search width |
| top_k | 10 | Top-K candidates after ranking |
| α (alpha) | 1.0 | LM score weight |
| β (beta) | 0.1 | Prior score weight |
| λ (lambda) | 1.0 | Edit cost penalty |
| δ (delta) | 0.3 | Replacement threshold |
| τ (tau) | 2.0 | Softmax temperature |

## Experiment Design (Ablation Study)

| ID | Description | Normalization | Fusion |
|----|-------------|--------------|--------|
| E0 | Baseline | None | — |
| E1 | Top-1 normalization | Top-1 only | Weighted |
| E2 | Top-10 mean fusion | Top-10 | Mean |
| E3 | Top-10 weighted fusion | Top-10 | Weighted |

## Data Format

### Sentiment Data (CSV)
```
id,text,label
1,这个功能真的很好用,pos
```
Labels: `pos` / `neu` / `neg`

### Homophone Dictionary (JSON)
```json
{"垃圾": ["垃圾", "拉圾", "辣鸡", ...]}
```

### Lexicon Files (TXT)
One entry per line. high_freq_words.txt may include optional `\tcount` column.

## Setup & Usage

See README.md for installation and usage instructions.
