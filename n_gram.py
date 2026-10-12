from collections import Counter
import math

class NGram():
    def __init__(self, n, V):
        self.n = n
        self.ngram = {}
        self.V = V
        self.delta = 0.00000001
    
    def train(self, corpus):
        nominator =  Counter()
        denominator = Counter()

        for i in range(self.n-1, len(corpus)):
            denominator[tuple(corpus[i-(self.n-1):i])] += 1
            nominator[tuple(corpus[i-(self.n-1):i+1])] += 1

        for key, value in nominator.items():
            self.ngram[key] = (value+self.delta)/(denominator[key[:-1]]+self.delta*len(self.V))
        
        self.denominator = denominator

    
    def get_next_token_id(self, generated):
        next_id = None
        max_p = 0

        for w, w_id in self.V.items():
            seq = tuple(generated[-(self.n-1):])+ (w_id,)
            p = self.ngram.get(seq)
            if p is None:
                p = (0 + self.delta) / (self.denominator[seq[:-1]] + self.delta * len(self.V))

            if p > max_p:
                max_p = p
                next_id = w_id
        
        if next_id == 0:
            return None

        return next_id
    
    def log_probability(self, text):
        total = 0.0
        for i in range(self.n - 1, len(text)):
            seq = tuple(text[i-(self.n-1):i+1])
            p = self.ngram.get(seq)
            if p is None:
                p = (0 + self.delta) / (self.denominator[seq[:-1]] + self.delta * len(self.V))
            total += math.log(p)
        return total