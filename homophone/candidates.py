"""
Candidate generation: for each suspicious word, retrieve top-10 homophone candidates.
"""
from typing import List, Dict, Set, Optional

class CandidateGenerator:
    """
    Fetches homophone candidates for a given token from the homophone dictionary.
    """
    
    def __init__(
        self,
        homophone_dict: Dict[str, List[str]],
        high_freq_words: Optional[Set[str]] = None,
        max_candidates: int = 10,
    ):
        self.homophone_dict = homophone_dict
        self.high_freq_words = high_freq_words
        self.max_candidates = max_candidates
    
    def get_candidates(self, token: str) -> List[str]:
        """
        Returns up to max_candidates homophone candidates for the given token.
        Always includes the original token.
        Optionally filters to high-freq words only.
        """
        candidates = list(self.homophone_dict.get(token, []))
        
        # Ensure the original token is included
        if token not in candidates:
            candidates.insert(0, token)
        
        # Optional: filter to high-freq words (keep original token always)
        if self.high_freq_words:
            filtered = [c for c in candidates if c == token or c in self.high_freq_words]
            if not filtered:
                filtered = [token]
            candidates = filtered
        
        return candidates[:self.max_candidates]
