"""
Step 5: Ablation study runner.
Runs experiments E0-E5 and prints comparison table.

Requires:
  - A trained sentiment model checkpoint
  - Test data
  - Homophone normalization pipeline
"""
import os
import json
import argparse
import pandas as pd
from typing import List, Dict

from eval.metrics import compute_metrics, print_metrics
from eval.confusion import plot_confusion_matrix


def load_predictions(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment(
    experiment_id: str,
    test_csv: str,
    normalizer,
    predictor,
    fuser,
    use_normalization: bool = True,
    top_k: int = 10,
    strategy: str = "weighted",
) -> Dict:
    """Run a single experiment and return metrics."""
    df = pd.read_csv(test_csv)
    y_true = df["label"].tolist()
    y_pred = []
    
    for _, row in df.iterrows():
        text = str(row["text"])
        
        if not use_normalization or predictor is None:
            # E0: no normalization
            if predictor is not None:
                preds = predictor.predict([text])
                y_pred.append(preds[0]["label"])
            else:
                y_pred.append("neu")  # dummy
            continue
        
        # Normalize
        candidates = normalizer.normalize(text)
        candidate_texts = [c for c, _ in candidates[:top_k]]
        candidate_weights = [w for _, w in candidates[:top_k]]
        
        if experiment_id == "E1":
            # Top1 only
            candidate_texts = candidate_texts[:1]
            candidate_weights = [1.0]
        
        all_probs = predictor.predict_proba(candidate_texts)
        
        if strategy == "mean" or experiment_id == "E2":
            result = fuser.fuse(all_probs, weights=None)
        else:
            result = fuser.fuse(all_probs, weights=candidate_weights)
        
        y_pred.append(result["label"])
    
    metrics = compute_metrics(y_true, y_pred)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", default="data/sentiment/test.csv")
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--model_name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--model_type", default="roberta")
    parser.add_argument("--homophone_dict", default="data/homophones/chinese_homophones_top10.json")
    parser.add_argument("--high_freq_words", default="data/lexicon/high_freq_words.txt")
    parser.add_argument("--negation_words", default="data/lexicon/negation_words.txt")
    parser.add_argument("--degree_words", default="data/lexicon/degree_words.txt")
    parser.add_argument("--sentiment_pos", default="data/lexicon/sentiment_lexicon_pos.txt")
    parser.add_argument("--sentiment_neg", default="data/lexicon/sentiment_lexicon_neg.txt")
    parser.add_argument("--output_dir", default="results/")
    args = parser.parse_args()
    
    from pipeline.run_infer import build_pipeline
    normalizer, predictor, fuser_weighted = build_pipeline(
        homophone_dict_path=args.homophone_dict,
        high_freq_words_path=args.high_freq_words,
        negation_words_path=args.negation_words,
        degree_words_path=args.degree_words,
        sentiment_pos_path=args.sentiment_pos,
        sentiment_neg_path=args.sentiment_neg,
        model_checkpoint=args.model_checkpoint,
        model_name=args.model_name,
        model_type=args.model_type,
        fusion_strategy="weighted",
    )
    
    from fusion.fuse import SentimentFuser
    fuser_mean = SentimentFuser(strategy="mean")
    
    experiments = [
        ("E0: No Normalization", False, "weighted"),
        ("E1: Top1 Normalization", True, "weighted"),
        ("E2: Top10 Mean Fusion", True, "mean"),
        ("E3: Top10 Weighted Fusion", True, "weighted"),
    ]
    
    os.makedirs(args.output_dir, exist_ok=True)
    results_summary = []
    
    for name, use_norm, strategy in experiments:
        fuser = fuser_mean if strategy == "mean" else fuser_weighted
        metrics = run_experiment(
            experiment_id=name.split(":")[0].strip(),
            test_csv=args.test_csv,
            normalizer=normalizer,
            predictor=predictor,
            fuser=fuser,
            use_normalization=use_norm,
            strategy=strategy,
        )
        print_metrics(metrics, experiment_name=name)
        results_summary.append({
            "experiment": name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "neg_f1": metrics["per_class"]["neg"]["f1"],
            "neu_f1": metrics["per_class"]["neu"]["f1"],
            "pos_f1": metrics["per_class"]["pos"]["f1"],
        })
    
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(args.output_dir, "ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nAblation summary saved to {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
