"""
RoBERTa-based three-class Chinese sentiment classifier.
"""
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class RoBERTaSentimentClassifier(nn.Module):
    """
    Fine-tuned Chinese RoBERTa for 3-class sentiment classification (pos/neu/neg).
    """
    
    LABEL2ID = {"neg": 0, "neu": 1, "pos": 2}
    ID2LABEL = {0: "neg", 1: "neu", 2: "pos"}
    
    def __init__(self, model_name: str = "hfl/chinese-roberta-wwm-ext", num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits}
    
    @classmethod
    def load(cls, checkpoint_path: str, model_name: str = "hfl/chinese-roberta-wwm-ext"):
        model = cls(model_name=model_name)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        return model
