"""
RBMA: RoBERTa + BiLSTM + Multi-head Attention for Chinese sentiment classification.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class RBMASentimentClassifier(nn.Module):
    """
    RoBERTa-BiLSTM-Multi-head Attention (RBMA) for 3-class sentiment.
    
    Architecture:
      [CLS..SEP] -> RoBERTa -> BiLSTM -> Multi-head Attention -> Linear -> 3-class softmax
    """
    
    LABEL2ID = {"neg": 0, "neu": 1, "pos": 2}
    ID2LABEL = {0: "neg", 1: "neu", 2: "pos"}
    
    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        num_labels: int = 3,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        
        bilstm_out = lstm_hidden * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=bilstm_out,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bilstm_out, num_labels)
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq_out = outputs.last_hidden_state  # (B, L, H)
        
        lstm_out, _ = self.bilstm(seq_out)  # (B, L, 2*lstm_hidden)
        
        key_padding_mask = (attention_mask == 0)  # (B, L)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding_mask)
        
        # Mean pooling over sequence
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (attn_out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        
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
