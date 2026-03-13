"""
Beam search to enumerate candidate normalized sentences.
Expands homophone candidates at suspicious positions without combinatorial explosion.
"""
from typing import List, Tuple, Dict

class BeamSearchExpander:
    """
    Uses beam search to generate candidate sentences by replacing
    suspicious tokens with their homophone candidates.
    """
    
    def __init__(self, beam_size: int = 50):
        self.beam_size = beam_size
    
    def expand(
        self,
        tokens: List[str],
        suspicious_positions: List[Tuple[int, str]],
        candidates_map: Dict[int, List[str]],
    ) -> List[List[str]]:
        """
        Args:
            tokens: original tokenized sentence
            suspicious_positions: list of (position, original_token)
            candidates_map: {position: [candidate1, candidate2, ...]}
        
        Returns:
            List of candidate token sequences (beam_size at most)
        """
        # Start beam with original tokens
        beam: List[List[str]] = [list(tokens)]
        
        for pos, orig_token in suspicious_positions:
            cands = candidates_map.get(pos, [orig_token])
            new_beam: List[List[str]] = []
            for existing_seq in beam:
                for cand in cands:
                    new_seq = list(existing_seq)
                    new_seq[pos] = cand
                    new_beam.append(new_seq)
            # Deduplicate
            seen = set()
            deduped = []
            for seq in new_beam:
                key = tuple(seq)
                if key not in seen:
                    seen.add(key)
                    deduped.append(seq)
            beam = deduped[:self.beam_size]
        
        return beam
