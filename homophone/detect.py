"""
Suspicious word detection: identifies the top-m most likely noisy/homophone words
in a tokenized sentence.
"""
import math
from typing import List, Dict, Tuple, Set

class SuspiciousWordDetector:
    """
    Scores each token for suspiciousness (possible homophone/typo).
    
    Scoring formula:
      S = 2*has_candidate + 1.5*low_freq + 1.0*context + 0.5*shape
    """
    
    def __init__(
        self,
        homophone_dict: Dict[str, List[str]],
        high_freq_words: Set[str],
        negation_words: Set[str],
        degree_words: Set[str],
        low_freq_threshold: int = 0,
    ):
        self.homophone_dict = homophone_dict
        self.high_freq_words = high_freq_words
        self.negation_words = negation_words
        self.degree_words = degree_words
        self.low_freq_threshold = low_freq_threshold
    
    def score_token(self, token: str, position: int, tokens: List[str]) -> float:
        """Compute suspiciousness score for a single token."""
        has_candidate = 1.0 if token in self.homophone_dict else 0.0
        low_freq = 1.0 if token not in self.high_freq_words else 0.0
        
        # Context: check nearby tokens for negation/degree words
        context_window = tokens[max(0, position-2):position+3]
        has_context = any(
            t in self.negation_words or t in self.degree_words
            for t in context_window if t != token
        )
        context = 1.0 if has_context else 0.0
        
        # Shape: contains digits, repeated chars, or unusual symbols
        shape = self._shape_score(token)
        
        # Weights reflect relative importance: homophone match > freq > context > shape
        return 2.0 * has_candidate + 1.5 * low_freq + 1.0 * context + 0.5 * shape
    
    def _shape_score(self, token: str) -> float:
        if not token:
            return 0.0
        has_digit = any(c.isdigit() for c in token)
        has_repeat = len(token) > 1 and len(set(token)) < len(token)
        has_unusual = any(not c.isalnum() and ord(c) < 128 for c in token)
        return float(has_digit or has_repeat or has_unusual)
    
    def detect(self, tokens: List[str], m: int = 2) -> List[Tuple[int, str]]:
        """
        Returns the top-m suspicious (position, token) pairs.
        Skips single-character punctuation and very short tokens.
        """
        scored = []
        for i, tok in enumerate(tokens):
            if len(tok) < 1 or tok in {"，", "。", "！", "？", "、", "；", "：", " "}:
                continue
            s = self.score_token(tok, i, tokens)
            if s > 0:
                scored.append((i, tok, s))
        scored.sort(key=lambda x: -x[2])
        return [(i, tok) for i, tok, _ in scored[:m]]
