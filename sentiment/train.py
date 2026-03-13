"""
Training script for Chinese sentiment classifiers (RoBERTa / RBMA).
"""
import os
import argparse
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from sentiment.model_roberta import RoBERTaSentimentClassifier
from sentiment.model_rbma import RBMASentimentClassifier


LABEL2ID = {"neg": 0, "neu": 1, "pos": 2}


class SentimentDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        df = pd.read_csv(csv_path)
        self.texts = df["text"].tolist()
        self.labels = [LABEL2ID[l] for l in df["label"].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros((1, self.max_length), dtype=torch.long)).squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_ds = SentimentDataset(args.train_csv, tokenizer, max_length=args.max_length)
    val_ds = SentimentDataset(args.val_csv, tokenizer, max_length=args.max_length)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    if args.model_type == "roberta":
        model = RoBERTaSentimentClassifier(model_name=args.model_name, dropout=args.dropout)
    elif args.model_type == "rbma":
        model = RBMASentimentClassifier(model_name=args.model_name, dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    best_val_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, f"best_{args.model_type}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved checkpoint to {ckpt_path}")
    
    print(f"Training done. Best val_acc={best_val_acc:.4f}")


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            out = model(**batch)
            preds = out["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/sentiment/train.csv")
    parser.add_argument("--val_csv", default="data/sentiment/val.csv")
    parser.add_argument("--model_name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--model_type", default="roberta", choices=["roberta", "rbma"])
    parser.add_argument("--output_dir", default="checkpoints/")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
