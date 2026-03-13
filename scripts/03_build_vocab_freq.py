"""
Step 3: Build vocabulary frequency table from training corpus.
Reads train.csv (or a plain text file), tokenizes with jieba,
and outputs a TSV of word\tcount.
"""
import os
import sys
import argparse
from collections import Counter
import pandas as pd
import jieba


def build_freq(input_path: str, output_path: str, top_k: int = 100000, is_csv: bool = True, text_col: str = "text"):
    counter = Counter()
    
    if is_csv:
        df = pd.read_csv(input_path)
        texts = df[text_col].tolist()
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    
    for text in texts:
        tokens = list(jieba.cut(str(text), cut_all=False))
        counter.update(tokens)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for word, count in counter.most_common(top_k):
            if word.strip():
                f.write(f"{word}\t{count}\n")
    
    print(f"Saved {min(top_k, len(counter))} words to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sentiment/train.csv")
    parser.add_argument("--output", default="data/lexicon/high_freq_words.txt")
    parser.add_argument("--top_k", type=int, default=100000)
    parser.add_argument("--is_csv", action="store_true", default=True)
    parser.add_argument("--text_col", default="text")
    args = parser.parse_args()
    build_freq(args.input, args.output, args.top_k, args.is_csv, args.text_col)


if __name__ == "__main__":
    main()
