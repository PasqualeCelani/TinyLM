from tinygrad import Tensor, nn
from tinygrad.nn.state import safe_save, get_state_dict
import math


class TinyLM():
    def __init__(self, lm_config):
        self.vocab_size = lm_config["vocab_size"]
        self.embeeding_dim = lm_config["embeeding_dim"]
        self.hiddend_dim = lm_config["hiddend_dim"]
        self.ctx_size = lm_config["ctx_size"]

        self.E = Tensor.normal(self.vocab_size, self.embeeding_dim, mean=0, std=1)
        self.lin1 = nn.Linear(self.ctx_size * self.embeeding_dim, self.hiddend_dim)
        self.lin2 = nn.Linear(self.hiddend_dim, self.vocab_size) 
    
    def __call__(self, windows):
        embedded = self.E[windows] # (num_windows, ctx, emb_dim)
        X = embedded.reshape(embedded.shape[0], self.ctx_size * self.embeeding_dim) # (num_windows, ctx*emb)

        a = self.lin1(X).tanh() # (num_windows, hiden_dim)
        s = self.lin2(a).T  # (vocab_size, num_windows)
        p = s.log_softmax(axis=0) # (vocab_size, num_windows)

        return p

def save_model_weights(lm):
    state_dict = get_state_dict(lm)
    safe_save(state_dict, "model.safetensors")
    print("[SUCCESS] model weight have been saved at ./model.safetensors")

def get_model_param_number(lm):
    return sum(math.prod(param.shape) for param in nn.state.get_parameters(lm))