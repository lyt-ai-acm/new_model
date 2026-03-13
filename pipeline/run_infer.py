"""
End-to-end inference pipeline:
  raw text -> homophone normalization (Top10) -> sentiment fusion -> label
"""
import argparse
import json
import os
import sys
from typing import List, Dict, Optional


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_set(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_freq_dict(path: str) -> Dict[str, float]:
    if not os.path.exists(path):
        return {}
    freq = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t")
                freq[parts[0]] = float(parts[1]) if len(parts) > 1 else 1.0
    return freq


def build_pipeline(
    homophone_dict_path: str,
    high_freq_words_path: str,
    negation_words_path: str,
    degree_words_path: str,
    sentiment_pos_path: str,
    sentiment_neg_path: str,
    model_checkpoint: Optional[str] = None,
    model_name: str = "hfl/chinese-roberta-wwm-ext",
    model_type: str = "roberta",
    lm_model_path: Optional[str] = None,
    fusion_strategy: str = "weighted",
    m: int = 2,
    beam_size: int = 50,
    top_k: int = 10,
    alpha: float = 1.0,
    beta: float = 0.1,
    lambda_: float = 1.0,
    delta: float = 0.3,
    tau: float = 2.0,
):
    from homophone.normalize import HomophoneNormalizer
    from lm.kenlm_scorer import KenLMScorer
    from fusion.fuse import SentimentFuser

    homophone_dict = load_json(homophone_dict_path)
    high_freq_words = load_set(high_freq_words_path)
    negation_words = load_set(negation_words_path)
    degree_words = load_set(degree_words_path)
    sentiment_pos = load_set(sentiment_pos_path)
    sentiment_neg = load_set(sentiment_neg_path)
    
    # Build word freq from high_freq_words (uniform if no freq info)
    word_freq = {w: 1.0 for w in high_freq_words}
    
    lm_scorer = KenLMScorer(model_path=lm_model_path, word_freq=word_freq)
    
    normalizer = HomophoneNormalizer(
        homophone_dict=homophone_dict,
        high_freq_words=high_freq_words,
        negation_words=negation_words,
        degree_words=degree_words,
        word_freq=word_freq,
        sentiment_pos=sentiment_pos,
        sentiment_neg=sentiment_neg,
        lm_scorer=lm_scorer,
        m=m,
        beam_size=beam_size,
        top_k=top_k,
        alpha=alpha,
        beta=beta,
        lambda_=lambda_,
        delta=delta,
        tau=tau,
    )
    
    fuser = SentimentFuser(strategy=fusion_strategy)
    
    # Load sentiment model if checkpoint provided
    sentiment_predictor = None
    if model_checkpoint and os.path.exists(model_checkpoint):
        if model_type == "roberta":
            from sentiment.model_roberta import RoBERTaSentimentClassifier
            model = RoBERTaSentimentClassifier.load(model_checkpoint, model_name=model_name)
        elif model_type == "rbma":
            from sentiment.model_rbma import RBMASentimentClassifier
            model = RBMASentimentClassifier.load(model_checkpoint, model_name=model_name)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        from sentiment.predict import SentimentPredictor
        sentiment_predictor = SentimentPredictor(model, model_name=model_name)
    
    return normalizer, sentiment_predictor, fuser


def infer(
    text: str,
    normalizer,
    sentiment_predictor,
    fuser,
) -> Dict:
    """
    Run end-to-end inference on a single text.
    Returns {label, prob, all_probs, candidates}
    """
    # Normalize: get Top-10 candidates with weights
    candidates = normalizer.normalize(text)
    candidate_texts = [c for c, _ in candidates]
    candidate_weights = [w for _, w in candidates]
    
    if sentiment_predictor is None:
        return {
            "label": "unknown (no model loaded)",
            "candidates": candidates,
        }
    
    # Predict probabilities for each candidate
    all_probs = sentiment_predictor.predict_proba(candidate_texts)
    
    # Fuse predictions
    result = fuser.fuse(all_probs, weights=candidate_weights)
    result["candidates"] = [(text, weight) for text, weight in candidates]
    
    return result


def main():
    parser = argparse.ArgumentParser(description="End-to-end Chinese sentiment inference with homophone normalization")
    parser.add_argument("--text", type=str, help="Input text to analyze")
    parser.add_argument("--input_file", type=str, help="CSV file with 'text' column")
    parser.add_argument("--homophone_dict", default="data/homophones/chinese_homophones_top10.json")
    parser.add_argument("--high_freq_words", default="data/lexicon/high_freq_words.txt")
    parser.add_argument("--negation_words", default="data/lexicon/negation_words.txt")
    parser.add_argument("--degree_words", default="data/lexicon/degree_words.txt")
    parser.add_argument("--sentiment_pos", default="data/lexicon/sentiment_lexicon_pos.txt")
    parser.add_argument("--sentiment_neg", default="data/lexicon/sentiment_lexicon_neg.txt")
    parser.add_argument("--model_checkpoint", default=None)
    parser.add_argument("--model_name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--model_type", default="roberta", choices=["roberta", "rbma"])
    parser.add_argument("--lm_model_path", default=None)
    parser.add_argument("--fusion_strategy", default="weighted", choices=["mean", "weighted"])
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambda_", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    
    normalizer, predictor, fuser = build_pipeline(
        homophone_dict_path=args.homophone_dict,
        high_freq_words_path=args.high_freq_words,
        negation_words_path=args.negation_words,
        degree_words_path=args.degree_words,
        sentiment_pos_path=args.sentiment_pos,
        sentiment_neg_path=args.sentiment_neg,
        model_checkpoint=args.model_checkpoint,
        model_name=args.model_name,
        model_type=args.model_type,
        lm_model_path=args.lm_model_path,
        fusion_strategy=args.fusion_strategy,
        m=args.m,
        beam_size=args.beam_size,
        top_k=args.top_k,
        alpha=args.alpha,
        beta=args.beta,
        lambda_=args.lambda_,
        delta=args.delta,
        tau=args.tau,
    )
    
    if args.text:
        result = infer(args.text, normalizer, predictor, fuser)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    elif args.input_file:
        import pandas as pd
        df = pd.read_csv(args.input_file)
        results = []
        for _, row in df.iterrows():
            r = infer(row["text"], normalizer, predictor, fuser)
            results.append(r)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output_json}")
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))
    
    else:
        print("Please provide --text or --input_file")
        parser.print_help()


if __name__ == "__main__":
    main()
