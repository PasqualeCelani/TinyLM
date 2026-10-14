from abc import ABC, abstractmethod
from typing import List


class AutoregressiveLanguageModel(ABC):
    def __init__(self, vocab : dict, configs : dict):
        self.vocab = vocab
        self.configs = configs  

    @abstractmethod
    def train(self, corpus : List[int]) -> None:
        """Training method on tokenized corpus"""

    @abstractmethod
    def next_token_distribution(self, generated : List[int]) -> dict:
        """Returns the next token distribution based on the current generated text"""

    @abstractmethod
    def log_probability(self, text : List[int]):
        """Returns total log probability of the current text"""