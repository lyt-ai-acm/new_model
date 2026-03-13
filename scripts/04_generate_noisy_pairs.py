"""
Step 4: Generate noisy/homophone-corrupted sentence pairs for data augmentation.
For each sentence in train.csv, randomly replace 1-2 words with their homophones.
Outputs augmented_train.csv with extra noisy examples.
"""
import os
import json
import random
import argparse
import pandas as pd
import jieba
from typing import List, Dict


def load_homophone_dict(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def corrupt_sentence(text: str, homophone_dict: Dict[str, List[str]], p_replace: float = 0.3) -> str:
    tokens = list(jieba.cut(text, cut_all=False))
    new_tokens = []
    for tok in tokens:
        if tok in homophone_dict and random.random() < p_replace:
            candidates = [c for c in homophone_dict[tok] if c != tok]
            if candidates:
                tok = random.choice(candidates)
        new_tokens.append(tok)
    return "".join(new_tokens)


def generate_noisy_pairs(
    input_csv: str,
    homophone_dict_path: str,
    output_csv: str,
    n_augment: int = 2,
    p_replace: float = 0.3,
    seed: int = 42,
):
    random.seed(seed)
    homophone_dict = load_homophone_dict(homophone_dict_path)
    df = pd.read_csv(input_csv)
    
    augmented_rows = []
    for _, row in df.iterrows():
        for i in range(n_augment):
            noisy = corrupt_sentence(str(row["text"]), homophone_dict, p_replace)
            augmented_rows.append({
                "id": f"{row['id']}_aug{i+1}",
                "text": noisy,
                "label": row["label"],
            })
    
    aug_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([df, aug_df], ignore_index=True)
    combined.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved {len(combined)} rows ({len(df)} original + {len(aug_df)} augmented) to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="data/sentiment/train.csv")
    parser.add_argument("--homophone_dict", default="data/homophones/chinese_homophones_top10.json")
    parser.add_argument("--output_csv", default="data/sentiment/augmented_train.csv")
    parser.add_argument("--n_augment", type=int, default=2)
    parser.add_argument("--p_replace", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_noisy_pairs(args.input_csv, args.homophone_dict, args.output_csv, args.n_augment, args.p_replace, args.seed)


if __name__ == "__main__":
    main()
