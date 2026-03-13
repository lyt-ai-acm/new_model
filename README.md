# Chinese Homophone Normalization + Sentiment Analysis

A complete NLP pipeline for robust Chinese sentiment analysis that handles homophone substitutions common in social media text (e.g., "辣鸡" → "垃圾", "开新" → "开心").

## Features

- **Homophone Normalization**: Detects suspicious tokens, generates Top-10 candidate corrections using a homophone dictionary, and ranks them with an LM + prior + edit-cost scorer
- **Two Sentiment Models**: Chinese RoBERTa baseline and RBMA (RoBERTa + BiLSTM + Multi-head Attention)
- **Candidate Fusion**: Aggregates sentiment predictions across normalized candidates (mean or weighted)
- **Ablation Study**: Compare E0 (no normalization) → E3 (Top-10 weighted fusion)

## Project Structure

```
.
├── data/
│   ├── sentiment/          # train/val/test CSV files
│   ├── lexicon/            # high-freq words, sentiment lexicons, negation/degree words
│   └── homophones/         # chinese_homophones_top10.json
├── lm/                     # KenLM scorer (with unigram fallback)
├── homophone/              # tokenizer, detect, candidates, beam, rank, normalize
├── sentiment/              # model_roberta, model_rbma, train, predict
├── fusion/                 # candidate fusion strategies
├── eval/                   # metrics and confusion matrix
├── pipeline/               # end-to-end inference
├── scripts/                # data prep, LM training, augmentation, ablation
├── requirements.txt
├── PROJECT_SPEC.md
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Optional: Install KenLM for n-gram language model support:
```bash
git clone https://github.com/kpu/kenlm.git
mkdir kenlm/build && cd kenlm/build
cmake .. && make -j4
sudo make install
pip install https://github.com/kpu/kenlm/archive/master.zip
```

## Quick Start

### 1. Validate Data
```bash
python scripts/01_prepare_corpus.py
```

### 2. Build Vocabulary Frequency
```bash
python scripts/03_build_vocab_freq.py --input data/sentiment/train.csv --output data/lexicon/vocab_freq.txt
```

### 3. Generate Noisy Training Data
```bash
python scripts/04_generate_noisy_pairs.py
```

### 4. Train Sentiment Model
```bash
# RoBERTa baseline
python -m sentiment.train --model_type roberta --epochs 5 --output_dir checkpoints/

# RBMA model
python -m sentiment.train --model_type rbma --epochs 5 --output_dir checkpoints/
```

### 5. Run Inference
```bash
# Single sentence
python -m pipeline.run_infer --text "辣鸡产品根本不能用" --model_checkpoint checkpoints/best_roberta.pt

# Batch inference
python -m pipeline.run_infer --input_file data/sentiment/test.csv --model_checkpoint checkpoints/best_roberta.pt --output_json results/predictions.json
```

### 6. Run Ablation Study
```bash
python scripts/05_run_ablation.py --model_checkpoint checkpoints/best_roberta.pt
```

## Pipeline Overview

```
Raw Text → Tokenize → Detect Suspicious Tokens → Generate Homophone Candidates
       → Beam Search → Rank (LM + Prior + EditCost) → Top-10 Candidates
       → Sentiment Prediction (per candidate) → Fusion → Final Label
```

## Model Details

### RoBERTa Baseline
- Base: `hfl/chinese-roberta-wwm-ext`
- Head: Dropout + Linear(768, 3)

### RBMA (RoBERTa + BiLSTM + Multi-head Attention)
- Base: `hfl/chinese-roberta-wwm-ext`
- BiLSTM: 2 layers, hidden=256, bidirectional
- Multi-head Attention: 8 heads, embed_dim=512
- Head: Dropout + Linear(512, 3)

## Evaluation Metrics
- Accuracy
- Macro-F1
- Per-class Precision / Recall / F1 (neg / neu / pos)

## License

MIT
