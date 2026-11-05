from .base import AutoregressiveLanguageModel
from . import register
from tinygrad import Tensor, Device, Context, nn, TinyJit
from tqdm import tqdm
import numpy as np
import math
from tinygrad.nn.optim import SGD

class RNN():
    def __init__(self, lm_config):
        self.vocab_size = lm_config["vocab_size"]
        self.embeeding_dim = lm_config["embeeding_dim"]

        self.E = Tensor.normal(self.vocab_size, self.embeeding_dim, mean=0, std=0.02)

    def _self_orthogonal_init(tensor):
        pass
        
    
    def __call__(self, block_t):
        embedded = self.E[block_t]


@register("rnnlm")
class RNNLM(AutoregressiveLanguageModel):

    def __init__(self, vocab, config):
        super().__init__(vocab, config)

        config["params"]["vocab_size"] = len(vocab)
        self.lm = RNN(config["params"])

        self.embeeding_dim = config["params"]["embeeding_dim"]

        self.epochs = config["training"]["epochs"]
        self.batch_size = config["training"]["batch_size"]
        self.eval_steps = config["training"]["eval_steps"]
        self.eval_batch_size = config["training"]["eval_batch_size"]

        self.opt = SGD(
            nn.state.get_parameters(self.lm), 
            lr=config["training"]["lr"], 
            weight_decay=config["training"]["weight_decay"]
        )  

    
    def _pad_corpus(self, corpus):
        pad_corpus = []
        max_seq_len = max([len(row) for row in corpus])
        for row in corpus:
            seq_len = len(row)
            seq_diff = max_seq_len-seq_len
            pad_row = row+([self.vocab["<||pad||>"]]*seq_diff)
            pad_corpus.append(pad_row)
        return pad_corpus
    
    @TinyJit
    def _train_step(self, block_t):
        pass


    def train(self, corpus, val_corpus=None):
        Tensor.manual_seed(42)
        ppl = float('inf')
        best_ppl = float('inf')

        print(f"[INFO] used device: {Device.DEFAULT}")
        print(f"[INFO] vocabulary size: {len(self.vocab)}")
        print(f"[INFO] embeeding dimension: {self.embeeding_dim}")
        print(f"[INFO] Number of model parameters: {round(self.get_model_param_number()/1000000, 2)}M")
        print(f"[INFO] batch size: {self.batch_size}")

        pad_corpus = np.asarray(self._pad_corpus(corpus))

        if val_corpus is not None:
            pad_corpus_val = np.asarray(self._pad_corpus(val_corpus))

        rng = np.random.default_rng(42)
        with Context(TRAINING=1):
            for epoch in range(self.epochs):
                epoch_loss, n_batches = 0.0, 0

                usable = len(pad_corpus) - (len(pad_corpus) % self.batch_size)
                perm = rng.permutation(usable)
                progress_bar = tqdm(
                    range(0, usable, self.batch_size),
                    desc=f"Epoch {epoch+1}",
                    unit="batch"
                )

                for i in progress_bar:
                    block_t = Tensor(pad_corpus[perm[i:i + self.batch_size]])
                    loss = self._train_step(block_t)

                    loss_val = loss.item()
                    epoch_loss += loss_val
                    n_batches += 1

                    train_ppl = math.exp(epoch_loss / n_batches)  

                    msg = (
                        f"Epoch {epoch+1} | Loss: {loss_val:.3f} | "
                        f"train_ppl: {train_ppl:.2f} | eval_ppl: {ppl:.2f}"
                    )

                    if i % 100 == 0:
                        progress_bar.set_description(msg)

                    if val_corpus is not None and i % self.eval_steps == 0:
                        logp = self.log_probability(pad_corpus_val)
                        ppl = math.exp(logp)
                        if ppl < best_ppl:
                            best_ppl = ppl
                            self._save_model_weights()

            print(f"[Epoch {epoch+1}] train_ppl: {math.exp(epoch_loss / n_batches):.2f} | eval_ppl: {ppl:.2f} | epoch_loss: {epoch_loss/n_batches}")
    
    def get_model_param_number(self):
        return sum(math.prod(param.shape) for param in nn.state.get_parameters(self.lm))

    def next_token_distribution(self, generated):
        pass

    def log_probability(self, blocks):
        pass

    def get_context_size(self):
        pass
    