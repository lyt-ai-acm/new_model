"""
Candidate sentence ranking using n-gram LM score + prior + edit cost.
"""
import math
from typing import List, Dict, Tuple, Set, Optional

class CandidateRanker:
    """
    Ranks candidate token sequences using:
      Score(x') = alpha * LM(x') + beta * Prior(x') - lambda_ * EditCost(x', x)
    
    Then applies a replacement threshold delta:
      If best_score - original_score < delta, keep original as top candidate.
    """
    
    def __init__(
        self,
        lm_scorer,
        word_freq: Dict[str, float],
        sentiment_pos: Set[str],
        sentiment_neg: Set[str],
        alpha: float = 1.0,
        beta: float = 0.1,
        lambda_: float = 1.0,
        kappa: float = 0.5,
        delta: float = 0.3,
        epsilon: float = 1e-8,
    ):
        self.lm_scorer = lm_scorer
        self.word_freq = word_freq
        self.sentiment_pos = sentiment_pos
        self.sentiment_neg = sentiment_neg
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.kappa = kappa
        self.delta = delta
        self.epsilon = epsilon
    
    def score(self, candidate_tokens: List[str], original_tokens: List[str]) -> float:
        lm_score = self.lm_scorer.score(candidate_tokens)
        prior_score = self._prior(candidate_tokens)
        edit_cost = self._edit_cost(candidate_tokens, original_tokens)
        return self.alpha * lm_score + self.beta * prior_score - self.lambda_ * edit_cost
    
    def _prior(self, tokens: List[str]) -> float:
        total = 0.0
        total_freq = sum(self.word_freq.values()) + 1.0
        for t in tokens:
            freq = self.word_freq.get(t, 0)
            total += math.log(freq / total_freq + self.epsilon)
            is_sentiment = 1.0 if (t in self.sentiment_pos or t in self.sentiment_neg) else 0.0
            total += self.kappa * is_sentiment
        return total
    
    def _edit_cost(self, candidate: List[str], original: List[str]) -> int:
        """Number of positions that differ."""
        min_len = min(len(candidate), len(original))
        diff = sum(1 for i in range(min_len) if candidate[i] != original[i])
        diff += abs(len(candidate) - len(original))
        return diff
    
    def rank(
        self,
        candidates: List[List[str]],
        original_tokens: List[str],
        top_k: int = 10,
    ) -> List[Tuple[List[str], float]]:
        """
        Returns top_k (tokens, score) pairs, applying delta threshold.
        """
        scored = [(seq, self.score(seq, original_tokens)) for seq in candidates]
        scored.sort(key=lambda x: -x[1])
        
        # Find original score
        orig_key = tuple(original_tokens)
        orig_score = next(
            (s for seq, s in scored if tuple(seq) == orig_key),
            self.score(original_tokens, original_tokens)
        )
        
        # Delta threshold: if best improvement < delta, put original first
        best_score = scored[0][1] if scored else orig_score
        if best_score - orig_score < self.delta:
            # Ensure original is in results and put it first
            without_orig = [(seq, s) for seq, s in scored if tuple(seq) != orig_key]
            result = [(original_tokens, orig_score)] + without_orig
        else:
            result = scored
        
        return result[:top_k]
