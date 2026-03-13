"""
Inference for trained sentiment classifiers.
"""
import torch
from typing import List, Dict
from transformers import AutoTokenizer


LABEL2ID = {"neg": 0, "neu": 1, "pos": 2}
ID2LABEL = {0: "neg", 1: "neu", 2: "pos"}


class SentimentPredictor:
    """
    Wraps a trained sentiment model for inference.
    Returns 3-class probabilities for each input sentence.
    """
    
    def __init__(self, model, model_name: str = "hfl/chinese-roberta-wwm-ext", max_length: int = 128, device: str = "cpu"):
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        """Returns list of [P(neg), P(neu), P(pos)] for each text."""
        results = []
        for text in texts:
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                logits = out["logits"]
                probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
            results.append(probs)
        return results
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """Returns list of {label, prob, all_probs} dicts."""
        all_probs = self.predict_proba(texts)
        results = []
        for probs in all_probs:
            label_id = int(torch.tensor(probs).argmax().item())
            results.append({
                "label": ID2LABEL[label_id],
                "prob": probs[label_id],
                "all_probs": {"neg": probs[0], "neu": probs[1], "pos": probs[2]},
            })
        return results
