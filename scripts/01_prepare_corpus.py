"""
Step 1: Prepare and validate sentiment corpus.
Checks train/val/test.csv for required format and prints statistics.
"""
import os
import pandas as pd
from collections import Counter


DATA_DIR = "data/sentiment"
FILES = ["train.csv", "val.csv", "test.csv"]
REQUIRED_COLS = {"id", "text", "label"}
VALID_LABELS = {"pos", "neu", "neg"}


def check_file(path: str):
    df = pd.read_csv(path)
    assert REQUIRED_COLS.issubset(df.columns), f"Missing columns in {path}: {REQUIRED_COLS - set(df.columns)}"
    
    invalid = set(df["label"].unique()) - VALID_LABELS
    assert not invalid, f"Invalid labels in {path}: {invalid}"
    
    dist = Counter(df["label"])
    avg_len = df["text"].str.len().mean()
    print(f"{os.path.basename(path)}: {len(df)} rows, label dist={dict(dist)}, avg_text_len={avg_len:.1f}")


def main():
    for fname in FILES:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            check_file(path)
        else:
            print(f"WARNING: {path} not found")


if __name__ == "__main__":
    main()
