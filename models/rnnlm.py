from .base import AutoregressiveLanguageModel
from . import register
from tinygrad import Tensor, Device, Context, nn, TinyJit
from tqdm import tqdm
import numpy as np
import math
from tinygrad.nn.optim import SGD

class RNN():
    def __init__(self, lm_config, batch_size, pad_index):
        self.vocab_size = lm_config["vocab_size"]
        self.embeeding_dim = lm_config["embeeding_dim"]
        self.hiddend_dim = lm_config["hiddend_dim"]
        self.batch_size = batch_size
        self.pad_index = pad_index

        self.E = Tensor.normal(self.vocab_size, self.embeeding_dim, mean=0, std=0.001)
        self.W_e = self._orthogonal_init(self.embeeding_dim, self.hiddend_dim)
        self.W_h = self._orthogonal_init(self.hiddend_dim, self.hiddend_dim)
        self.bias = Tensor.zeros(self.batch_size, self.hiddend_dim)
        
        self.lin = nn.Linear(self.hiddend_dim, self.vocab_size)


    def _orthogonal_init(self, rows, cols, gain=1.0):
        if rows < cols:
            return self._orthogonal_init(cols, rows, gain).transpose()
        q, r = Tensor.randn(rows, cols).qr()   
        q = q[:, :cols]                       
        q = q * r.diagonal().sign()           
        return q * gain
        
    
    def __call__(self, block_t):
        embedded = self.E[block_t]                   
        T = block_t.shape[1]
        h_t = Tensor.zeros(block_t.shape[0], self.hiddend_dim)
        hs = []
        for t in range(T - 1):
            X_t = embedded[:, t]                      
            h_t = (h_t @ self.W_h + X_t @ self.W_e + self.bias).tanh()
            hs.append(h_t)
        H = hs[0].stack(*hs[1:], dim=1)                
        logits = self.lin(H)                          
        return logits.sparse_categorical_crossentropy(block_t[:, 1:], ignore_index=self.pad_index)
        


@register("rnnlm")
class RNNLM(AutoregressiveLanguageModel):

    def __init__(self, vocab, config):
        super().__init__(vocab, config)

        config["params"]["vocab_size"] = len(vocab)
        self.batch_size = config["training"]["batch_size"]
        self.block_len = config["training"]["block_len"]
        self.lm = RNN(config["params"], self.batch_size, self.vocab["<||pad||>"])

        self.embeeding_dim = config["params"]["embeeding_dim"]
        self.hiddend_dim = config["params"]["hiddend_dim"]

        self.epochs = config["training"]["epochs"]
        self.eval_steps = config["training"]["eval_steps"]
        self.eval_batch_size = config["training"]["eval_batch_size"]

        self.opt = SGD(
            nn.state.get_parameters(self.lm), 
            lr=config["training"]["lr"], 
            weight_decay=config["training"]["weight_decay"]
        )  

    
    def _prepare_corpus(self, corpus, block_len):
        pad = self.vocab["<||pad||>"]
        blocks = []
        for row in corpus:
            for i in range(0, len(row), block_len):
                chunk = row[i:i + block_len]
                if len(chunk) < 2:
                    continue
                blocks.append(chunk + [pad] * (block_len - len(chunk)))
        return np.asarray(blocks)
    
    @TinyJit
    def _train_step(self, block_t):
        loss = self.lm(block_t)
        self.opt.zero_grad()
        loss.backward()
        return loss.realize(*self.opt.schedule_step()) 


    def train(self, corpus, val_corpus=None):
        Tensor.manual_seed(42)
        ppl = float('inf')
        best_ppl = float('inf')

        print(f"[INFO] used device: {Device.DEFAULT}")
        print(f"[INFO] vocabulary size: {len(self.vocab)}")
        print(f"[INFO] embeeding dimension: {self.embeeding_dim}")
        print(f"[INFO] hidden layer dimension: {self.hiddend_dim}")
        print(f"[INFO] Number of model parameters: {round(self.get_model_param_number()/1000000, 2)}M")
        print(f"[INFO] batch size: {self.batch_size}")
        print(f"[INFO] block length: {self.block_len}")

        pad_corpus = self._prepare_corpus(corpus, self.block_len)

        if val_corpus is not None:
            pad_corpus_val = self._prepare_corpus(val_corpus, self.block_len)
        

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
    