MODEL_REGISTRY = {}

#class decorator, for example @register(ngram)
def register(name):
    def wrapper(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' already registered")
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper

def build_model(name, vocab, params):
    if name not in MODEL_REGISTRY:
        from .n_gram import NGram  
        from .bengiolm import BengioLM
        from .rnnlm import RNNLM
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](vocab, params)