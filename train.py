from tokenizer import Tokenizer
from tinygrad import Tensor, Device, nn, Context, TinyJit
from tinygrad.nn.optim import SGD
from tqdm import tqdm
from model import TinyLM, save_model_weights, get_model_param_number

    
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

    lm_config = {
        "vocab_size" : len(V),
        "embeeding_dim" : 128,
        "hiddend_dim" : 64,
        "ctx_size" : 4
    }

    print(f"[INFO] used device: {Device.DEFAULT}")
    print(f"[INFO] vocabulary size: {len(V)}")
    print(f"[INFO] embeeding dimension: {lm_config["embeeding_dim"]}")
    print(f"[INFO] hidden layer dimension: {lm_config["hiddend_dim"]}")
    print(f"[INFO] context size: {lm_config["ctx_size"]}")


    lm = TinyLM(lm_config)
    print(f"[INFO] Number of model parametes: {get_model_param_number(lm)}")
   
    opt = SGD(nn.state.get_parameters(lm), lr=0.2)
    epochs = 1

    block_size = 1024
    print(f"[INFO] block size: {block_size}")
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

                if i % 100 == 0:
                    progress_bar.set_description(
                        f"Epoch {epoch+1} | Loss: {loss.item():.3f}"
                    )
    
    save_model_weights(lm)



if __name__ == "__main__":
    main()