"""
End-to-end homophone normalization: tokenize -> detect -> candidates -> beam -> rank -> output Top10.
"""
import math
from typing import List, Tuple, Dict, Set, Optional

from homophone.tokenizer import Tokenizer
from homophone.detect import SuspiciousWordDetector
from homophone.candidates import CandidateGenerator
from homophone.beam import BeamSearchExpander
from homophone.rank import CandidateRanker


class HomophoneNormalizer:
    """
    Full pipeline for homophone normalization.
    Given a raw sentence, returns Top-10 candidate normalized sentences with weights.
    """
    
    def __init__(
        self,
        homophone_dict: Dict[str, List[str]],
        high_freq_words: Set[str],
        negation_words: Set[str],
        degree_words: Set[str],
        word_freq: Dict[str, float],
        sentiment_pos: Set[str],
        sentiment_neg: Set[str],
        lm_scorer,
        m: int = 2,
        beam_size: int = 50,
        top_k: int = 10,
        alpha: float = 1.0,
        beta: float = 0.1,
        lambda_: float = 1.0,
        delta: float = 0.3,
        tau: float = 2.0,
        tokenizer_mode: str = "default",
    ):
        self.tokenizer = Tokenizer(mode=tokenizer_mode)
        self.detector = SuspiciousWordDetector(
            homophone_dict, high_freq_words, negation_words, degree_words
        )
        self.candidate_gen = CandidateGenerator(
            homophone_dict, high_freq_words
        )
        self.beam_expander = BeamSearchExpander(beam_size=beam_size)
        self.ranker = CandidateRanker(
            lm_scorer, word_freq, sentiment_pos, sentiment_neg,
            alpha=alpha, beta=beta, lambda_=lambda_, delta=delta
        )
        self.m = m
        self.top_k = top_k
        self.tau = tau
    
    def normalize(self, text: str) -> List[Tuple[str, float]]:
        """
        Args:
            text: raw Chinese sentence
        
        Returns:
            List of (normalized_sentence, weight) tuples, length <= top_k.
            Weights sum to 1.0 (softmax over scores).
        """
        tokens = self.tokenizer.tokenize(text)
        suspicious = self.detector.detect(tokens, m=self.m)
        
        if not suspicious:
            return [(text, 1.0)]
        
        candidates_map = {}
        for pos, tok in suspicious:
            cands = self.candidate_gen.get_candidates(tok)
            candidates_map[pos] = cands
        
        beam_results = self.beam_expander.expand(tokens, suspicious, candidates_map)
        ranked = self.ranker.rank(beam_results, tokens, top_k=self.top_k)
        
        # Compute softmax weights
        scores = [s for _, s in ranked]
        weights = self._softmax(scores, self.tau)
        
        results = []
        for (tok_seq, _), w in zip(ranked, weights):
            sentence = self.tokenizer.detokenize(tok_seq)
            results.append((sentence, w))
        
        return results
    
    def _softmax(self, scores: List[float], tau: float) -> List[float]:
        if not scores:
            return []
        scaled = [s / tau for s in scores]
        max_s = max(scaled)
        exps = [math.exp(s - max_s) for s in scaled]
        total = sum(exps)
        if total == 0 or total != total:  # guard against zero or NaN
            return [1.0 / len(scores)] * len(scores)
        return [e / total for e in exps]
