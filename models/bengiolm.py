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
        self.eval_steps = config["training"]["eval_steps"]
        self.eval_batch_size = config["training"]["eval_batch_size"]

        self.opt = SGD(
            nn.state.get_parameters(self.lm), 
            lr=config["training"]["lr"], 
            weight_decay=config["training"]["weight_decay"]
        )     

    @TinyJit
    def _train_step(self, block_t):
        windows = block_t[:, :-1]
        Y =  block_t[:, -1]
        idx = Tensor.arange(Y.shape[0])
        p = self.lm(windows)
        loss = -(p[Y, idx].mean())
        self.opt.zero_grad()
        loss.backward()
        return loss.realize(*self.opt.schedule_step()) 
    
    @TinyJit
    def _eval_step(self, block_t):
        windows = block_t[:, :-1]
        Y =  block_t[:, -1]
        idx = Tensor.arange(Y.shape[0])
        p = self.lm(windows)
        loss = -(p[Y, idx].mean())
        return loss.realize()

    def _save_model_weights(self):
        state_dict = get_state_dict(self.lm)
        safe_save(state_dict, "model.safetensors")
        print("[SUCCESS] model weight have been saved at ./model.safetensors")

    def load_model_weights(self, weight_path):
        state_dict = safe_load(weight_path)
        load_state_dict(self.lm, state_dict)

    def _create_blocks(self, corpus):
        first_row_np = np.asarray(corpus[0], dtype=np.int32)
        blocks =  sliding_window_view(first_row_np, self.ctx_size+1)
        for i in range(1, len(corpus)):
            row_np = np.asarray(corpus[i], dtype=np.int32)
            block = sliding_window_view(row_np, self.ctx_size+1)
            blocks = np.concat((blocks, block), axis=0)
        return blocks

    
    def train(self, corpus, val_corpus=None):
        Tensor.manual_seed(42)
        ppl = float('inf')
        best_ppl = float('inf')

        print(f"[INFO] used device: {Device.DEFAULT}")
        print(f"[INFO] vocabulary size: {len(self.vocab)}")
        print(f"[INFO] embeeding dimension: {self.embeeding_dim}")
        print(f"[INFO] hidden layer dimension: {self.hiddend_dim}")
        print(f"[INFO] context size: {self.ctx_size}")
        print(f"[INFO] Number of model parameters: {round(self.get_model_param_number()/1000000, 2)}M")
        print(f"[INFO] batch size: {self.batch_size}")

        blocks = self._create_blocks(corpus)

        if val_corpus is not None:
            blocks_val = self._create_blocks(val_corpus)
        
        rng = np.random.default_rng(42)
        with Context(TRAINING=1):
            for epoch in range(self.epochs):
                epoch_loss, n_batches = 0.0, 0
                
                usable = len(blocks) - (len(blocks) % self.batch_size)
                perm = rng.permutation(usable)
                progress_bar = tqdm(
                    range(0, usable, self.batch_size),
                    desc=f"Epoch {epoch+1}",
                    unit="batch"
                )

                for i in progress_bar:
                    block_t = Tensor(blocks[perm[i:i + self.batch_size]])
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
                        logp = self.log_probability(blocks_val)
                        ppl = math.exp(logp)
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

    def log_probability(self, blocks):
        B = self.eval_batch_size
        usable = len(blocks) - (len(blocks) % B)
        total, n = 0.0, 0
        for i in range(0, usable, B):
            block_t = Tensor(blocks[i:i + B])
            loss = self._eval_step(block_t).item()
            total += loss * B
            n += B
        return total / n  
           