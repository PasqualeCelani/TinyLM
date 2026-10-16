from .base import AutoregressiveLanguageModel
from . import register
from tinygrad import Tensor, nn, Device, Context, TinyJit
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import safe_save, get_state_dict, safe_load, load_state_dict
import math
from tqdm import tqdm
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class BengioLMModel():
    def __init__(self, lm_config):
        self.vocab_size = lm_config["vocab_size"]
        self.embeeding_dim = lm_config["embeeding_dim"]
        self.hiddend_dim = lm_config["hiddend_dim"]
        self.ctx_size = lm_config["ctx_size"]

        self.E = Tensor.normal(self.vocab_size, self.embeeding_dim, mean=0, std=0.05)
        self.lin1 = nn.Linear(self.ctx_size * self.embeeding_dim, self.hiddend_dim)
        self.lin2 = nn.Linear(self.hiddend_dim, self.vocab_size) 
    
    def __call__(self, windows):
        embedded = self.E[windows] # (num_windows, ctx, emb_dim)
        X = embedded.reshape(embedded.shape[0], self.ctx_size * self.embeeding_dim) # (num_windows, ctx*emb)

        a = self.lin1(X).tanh() # (num_windows, hiden_dim)
        s = self.lin2(a).T  # (vocab_size, num_windows)
        p = s.log_softmax(axis=0) # (vocab_size, num_windows)

        return p
    


@register("bengiolm")
class BengioLM(AutoregressiveLanguageModel):
    
    def __init__(self, vocab, config):
        super().__init__(vocab, config)

        config["params"]["vocab_size"] = len(vocab)
        self.lm = BengioLMModel(config["params"])

        self.embeeding_dim = config["params"]["embeeding_dim"]
        self.hiddend_dim = config["params"]["hiddend_dim"]
        self.ctx_size = config["params"]["ctx_size"]  

        self.epochs = config["training"]["epochs"]
        self.batch_size = config["training"]["batch_size"]
        self.num_windows = self.batch_size // self.ctx_size
        self.eval_steps = config["training"]["eval_steps"]

        self.opt = SGD(nn.state.get_parameters(self.lm), lr=config["training"]["lr"])     

    @TinyJit
    def _train_step(self, block_t):
        windows = block_t[:-1].reshape(self.num_windows, self.ctx_size) # (num_windows, ctx)
        Y = block_t[self.ctx_size::self.ctx_size] # (num_windows,)
        idx = Tensor.arange(self.num_windows) # (num_windows,)
        p = self.lm(windows)
        loss = -(p[Y, idx].mean())
        self.opt.zero_grad()
        loss.backward()
        return loss.realize(*self.opt.schedule_step()) 
    
    @TinyJit
    def _eval_step(self, X_t, Y_t):
        p = self.lm(X_t)                     # (vocab, B) log-probs
        return p[Y_t, Tensor.arange(X_t.shape[0])].sum()

    def _save_model_weights(self):
        state_dict = get_state_dict(self.lm)
        safe_save(state_dict, "model.safetensors")
        print("[SUCCESS] model weight have been saved at ./model.safetensors")

    def load_model_weights(self, weight_path):
        state_dict = safe_load(weight_path)
        load_state_dict(self.lm, state_dict)

    
    def train(self, corpus, val_corpus=None):
        Tensor.manual_seed(42)
        ppl = float('inf')
        best_ppl = float('inf')

        print(f"[INFO] used device: {Device.DEFAULT}")
        print(f"[INFO] vocabulary size: {len(self.vocab)}")
        print(f"[INFO] embeeding dimension: {self.embeeding_dim}")
        print(f"[INFO] hidden layer dimension: {self.hiddend_dim}")
        print(f"[INFO] context size: {self.ctx_size}")
        print(f"[INFO] Number of model parameters: {self.get_model_param_number()}")
        print(f"[INFO] batch size: {self.batch_size}")

        with Context(TRAINING=1):
            for epoch in range(self.epochs):
                epoch_loss, n_batches = 0.0, 0

                progress_bar = tqdm(
                    range(0, len(corpus), self.num_windows*self.ctx_size+1), 
                    desc=f"Epoch {epoch+1}",
                    unit="batch"
                )

                for i in progress_bar:
                    block_t = Tensor(corpus[i:i + (self.num_windows * self.ctx_size+1)])
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
                        logp = self.log_probability(val_corpus)
                        n_pred = len(val_corpus) - self.ctx_size
                        ppl = math.exp(-logp / n_pred)
                        if ppl < best_ppl:
                            best_ppl = ppl
                            self._save_model_weights()

                print(f"[Epoch {epoch+1}] train_ppl: {math.exp(epoch_loss / n_batches):.2f} | eval_ppl: {ppl:.2f} | epoch_loss: {epoch_loss/n_batches}")

        

    def get_model_param_number(self):
        return sum(math.prod(param.shape) for param in nn.state.get_parameters(self.lm))

    
    def next_token_distribution(self, generated):
        X = Tensor(generated[-self.ctx_size:]).reshape(1, self.ctx_size)
        logp = self.lm(X).exp().reshape(-1).numpy()
        return {
            w_id: logp[w_id]
            for w_id in self.vocab.values()
        }

    def get_context_size(self):
        return self.ctx_size+1

    def log_probability(self, text):
        ctx = self.ctx_size
        if len(text) <= ctx:
            return 0.0

        flat = np.asarray(text, dtype=np.int32)
        X = sliding_window_view(flat, ctx)    # (n, ctx)
        Y = flat[ctx:]
        n = Y.shape[0]

        B = 1024                             
        parts = []
        for i in range(0, n - B + 1, B):
            x = Tensor(np.ascontiguousarray(X[i:i + B]))
            y = Tensor(Y[i:i + B])
            parts.append(self._eval_step(x, y))

        total = sum(parts)                  

        r = n % B                           
        if r:
            x = Tensor(np.ascontiguousarray(X[n - r:]))
            y = Tensor(Y[n - r:])
            p = self.lm(x)
            total = total + p[y, Tensor.arange(r)].sum()

        return total.item()                 
           