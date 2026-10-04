from collections import Counter
import math

class NGram():
    def __init__(self, n, V):
        self.n = n
        self.ngram = {}
        self.V = V
    
    def train(self, corpus):
        nominator =  Counter()
        denominator = Counter()

        for i in range(self.n-1, len(corpus)):
            denominator[tuple(corpus[i-(self.n-1):i])] += 1
            nominator[tuple(corpus[i-(self.n-1):i+1])] += 1

        for key, value in nominator.items():
            self.ngram[key] = value/denominator[key[:-1]] if denominator[key[:-1]] != 0 else 0
    
    def get_next_token_id(self, generated):
        next_id = None
        max_p = 0

        for w, w_id in self.V.items():
            seq = tuple(generated[-(self.n-1):])+ (w_id,)
            p = self.ngram[seq] if seq in self.ngram.keys() else 0
            if p > max_p:
                max_p = p
                next_id = w_id

        return next_id
    
    def log_probability(self, text):
        total = 0.0
        for i in range(self.n - 1, len(text)):
            seq = tuple(text[i-(self.n-1):i+1])
            p = self.ngram.get(seq)
            if p is None or p == 0:
                return float("-inf") 
            total += math.log(p)
        return total