from tokenizer import Tokenizer
from tinygrad import Tensor, Device, nn, Context, TinyJit
from tinygrad.nn.optim import SGD
from tqdm import tqdm


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

    
def pack_corpus(rows, V):
    corpus = []
    for row in rows:
        row.insert(0, V["<||start||>"])
        row.append(V["<||end||>"])
        corpus += row
    return corpus

def main():
    Tensor.manual_seed(42)

    tokenizer = Tokenizer(n_workers=2)
    V, id_to_token, _ = tokenizer.load_training_from_file("./dataset/tokenized/tokenizer_train.json")
    rows = tokenizer.get_text_from_parquet("./dataset/tokenized/unsupervised-00000-of-00001_tokenized.parquet")

    print("[INFO] packing corpus...")
    packed_corpus = pack_corpus(rows, V)
    print("[SUCCESS] corpus packed.")

    print(f"[INFO] used device: {Device.DEFAULT}")

    lm_config = {
        "vocab_size" : len(V),
        "embeeding_dim" : 128,
        "hiddend_dim" : 64,
        "ctx_size" : 2
    }
    lm = TinyLM(lm_config)
    opt = SGD([lm.lin1.weight, lm.lin2.weight, lm.E], lr=1e-3)
    epochs = 3

    block_size = 20
    num_windows = block_size // lm_config["ctx_size"]

    @TinyJit
    def train_step(block_t):
        windows = block_t[:-1].reshape(num_windows, lm_config["ctx_size"]) # (num_windows, ctx)
        Y = block_t[lm_config["ctx_size"]::lm_config["ctx_size"]] # (num_windows,)
        idx = Tensor.arange(num_windows) # (num_windows,)
        p = lm(windows)
        loss = -(p[Y, idx].mean())
        opt.zero_grad()
        loss.backward()
        return loss.realize(*opt.schedule_step()) 


    with Context(TRAINING=1):
        for epoch in range(epochs):

            progress_bar = tqdm(
                range(0, len(packed_corpus), num_windows*lm_config["ctx_size"]+1), 
                desc=f"Epoch {epoch+1}",
                unit="batch"
            )

            for i in progress_bar:
                block_t = Tensor(packed_corpus[i:i+ (num_windows * lm_config["ctx_size"]+1)]) # (num_windows*ctx,)
                loss =  train_step(block_t)

                progress_bar.set_description(
                    f"Epoch {epoch+1} | Loss: {loss.item():.3f}"
                )



if __name__ == "__main__":
    main()