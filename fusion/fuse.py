"""
Fusion strategies for combining Top-10 normalized candidate predictions.
"""
import math
from typing import List, Dict, Tuple

_UNIFORM_PROB = 1.0 / 3


class SentimentFuser:
    """
    Fuses sentiment probability distributions from multiple candidates
    into a single final prediction.
    
    Supports:
      - 'mean': simple average
      - 'weighted': weighted by candidate scores (softmax weights)
    """
    
    LABELS = ["neg", "neu", "pos"]
    
    def __init__(self, strategy: str = "weighted"):
        assert strategy in ("mean", "weighted"), f"Unknown strategy: {strategy}"
        self.strategy = strategy
    
    def fuse(
        self,
        candidate_probs: List[List[float]],
        weights: List[float] = None,
    ) -> Dict:
        """
        Args:
            candidate_probs: list of [P(neg), P(neu), P(pos)] for each candidate
            weights: softmax weights from normalizer (used for 'weighted' strategy)
        
        Returns:
            {label, prob, all_probs}
        """
        if not candidate_probs:
            return {"label": "neu", "prob": _UNIFORM_PROB, "all_probs": {"neg": _UNIFORM_PROB, "neu": _UNIFORM_PROB, "pos": _UNIFORM_PROB}}
        
        n = len(candidate_probs)
        
        if self.strategy == "mean":
            fused = [sum(p[i] for p in candidate_probs) / n for i in range(3)]
        
        elif self.strategy == "weighted":
            if weights is None:
                weights = [1.0 / n] * n
            # Re-normalize weights
            w_sum = sum(weights)
            w = [wi / w_sum for wi in weights]
            fused = [sum(w[j] * candidate_probs[j][i] for j in range(n)) for i in range(3)]
        
        best_idx = fused.index(max(fused))
        label = self.LABELS[best_idx]
        
        return {
            "label": label,
            "prob": fused[best_idx],
            "all_probs": {"neg": fused[0], "neu": fused[1], "pos": fused[2]},
        }
    
    def fuse_batch(
        self,
        batch_candidate_probs: List[List[List[float]]],
        batch_weights: List[List[float]] = None,
    ) -> List[Dict]:
        """Fuse a batch of candidate probability sets."""
        if batch_weights is None:
            batch_weights = [None] * len(batch_candidate_probs)
        return [
            self.fuse(cps, ws)
            for cps, ws in zip(batch_candidate_probs, batch_weights)
        ]
