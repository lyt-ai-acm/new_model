"""
KenLM-based n-gram language model scorer for Chinese text.
Falls back to unigram frequency scoring when KenLM is unavailable.
"""
import math
import os
from typing import List, Optional, Dict

class KenLMScorer:
    """
    Wraps a KenLM language model for scoring tokenized Chinese sentences.
    Falls back to unigram frequency model if KenLM is not available.
    """
    def __init__(self, model_path: Optional[str] = None, word_freq: Optional[Dict[str, float]] = None):
        self.model = None
        self.word_freq = word_freq or {}
        self._freq_sum = sum(self.word_freq.values())
        self._use_kenlm = False
        
        if model_path and os.path.exists(model_path):
            try:
                import kenlm
                self.model = kenlm.Model(model_path)
                self._use_kenlm = True
            except ImportError:
                pass
    
    def score(self, tokens: List[str]) -> float:
        """
        Score a tokenized sentence. Returns log probability (normalized by length).
        """
        if not tokens:
            return 0.0
        sentence = " ".join(tokens)
        if self._use_kenlm and self.model is not None:
            return self.model.score(sentence) / len(tokens)
        else:
            return self._unigram_score(tokens)
    
    def _unigram_score(self, tokens: List[str]) -> float:
        total = 0.0
        vocab_size = max(len(self.word_freq), 1)
        denom = self._freq_sum + vocab_size
        for w in tokens:
            freq = self.word_freq.get(w, 0)
            prob = (freq + 1) / denom
            total += math.log(prob)
        return total / len(tokens)
    
    def is_ready(self) -> bool:
        return self._use_kenlm or len(self.word_freq) > 0
