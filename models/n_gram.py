from collections import Counter
import math
from .base import AutoregressiveLanguageModel
from . import register


@register("ngram")
class NGram(AutoregressiveLanguageModel):

    def __init__(self, vocab, config):
        super().__init__(vocab, config)
        self.n = config.get("n", 5)
        self.delta = config.get("delta", 0.00000001)
        self.ngram = {}

    def train(self, corpus):
        nominator =  Counter()
        denominator = Counter()

        for i in range(self.n-1, len(corpus)):
            denominator[tuple(corpus[i-(self.n-1):i])] += 1
            nominator[tuple(corpus[i-(self.n-1):i+1])] += 1

        for key, value in nominator.items():
            self.ngram[key] = (value+self.delta)/(denominator[key[:-1]]+self.delta*len(self.vocab))
        
        self.denominator = denominator

    def _prob(self, seq):
        p = self.ngram.get(seq)
        if p is None:
            p = (0 + self.delta) / (
                self.denominator[seq[:-1]] + self.delta * len(self.vocab)
            )
        return p

    def next_token_distribution(self, generated):
        context = tuple(generated[-(self.n - 1):])
        return {
            w_id: self._prob(context + (w_id,))
            for w_id in self.vocab.values()
        }
    
    def get_context_size(self):
        return self.n-1

    def log_probability(self, text):
        total = 0.0
        for i in range(self.n - 1, len(text)):
            seq = tuple(text[i - (self.n - 1):i + 1])
            total += math.log(self._prob(seq))
        return total