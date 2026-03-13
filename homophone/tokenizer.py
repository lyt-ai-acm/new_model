"""
Tokenizer wrapper for consistent tokenization across the pipeline.
Uses jieba for Chinese word segmentation.
"""
from typing import List

class Tokenizer:
    def __init__(self, mode: str = "default"):
        """
        mode: 'default' uses jieba, 'char' uses character-level splitting
        """
        self.mode = mode
        if mode == "default":
            try:
                import jieba
                self._jieba = jieba
            except ImportError:
                raise ImportError("jieba is required. Install with: pip install jieba")
    
    def tokenize(self, text: str) -> List[str]:
        if self.mode == "default":
            return list(self._jieba.cut(text, cut_all=False))
        elif self.mode == "char":
            return list(text)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def detokenize(self, tokens: List[str]) -> str:
        return "".join(tokens)
