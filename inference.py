from tokenizer import Tokenizer
from tinygrad import Tensor, Device
from tinygrad.nn.state import safe_load, load_state_dict
from model import TinyLM, get_model_param_number
from config import get_config


def next_token(logp, method="dist_sampling"):
    if method not in ["dist_sampling", "greedy"]:
        raise RuntimeError(f"[ERROR] the provided method: {method} is not in available.")
    
    if method == "dist_sampling":
        temperature = 0.75
        return logp.exp().flatten().pow(1/temperature).multinomial(num_samples=1).item()
    elif method == "greedy":
        return logp.exp().argmax(axis=0).item()

def main():
    Tensor.manual_seed(42)

    text = "First time I"

    tokenizer = Tokenizer(n_workers=2)
    V, id_to_token, merge_rules = tokenizer.load_training_from_file("./dataset/tokenized/tokenizer_train.json")
    row = tokenizer.tokenize_text(text, merge_rules, V)[0]

    lm_config = get_config("config.json")
    lm_config["vocab_size"] = len(V)

    print(f"[INFO] used device: {Device.DEFAULT}")
    print(f"[INFO] vocabulary size: {len(V)}")
    print(f"[INFO] embeeding dimension: {lm_config["embeeding_dim"]}")
    print(f"[INFO] hidden layer dimension: {lm_config["hiddend_dim"]}")
    print(f"[INFO] context size: {lm_config["ctx_size"]}")


    lm = TinyLM(lm_config)
    print(f"[INFO] Number of model parameters: {get_model_param_number(lm)}")

    state_dict = safe_load("./weights/model-2-ctx.safetensors")
    load_state_dict(lm, state_dict)

    ids = [V["<||start||>"]] + row  
    generated = list(ids)
    decoding_max_token = 1024

    print("="*100)
    print("+"*47+"TinyLM"+"+"*47)
    print("="*100)
    print(text, end=" ", flush=True)

    for _ in range(decoding_max_token):
        X = Tensor(generated[-lm_config["ctx_size"]:]).reshape(1, lm_config["ctx_size"])

        logp = lm(X)
        next_id = next_token(logp, method="dist_sampling")

        if next_id == V["<||end||>"]:
            break

        print(id_to_token[next_id], end=" ", flush=True)
        generated.append(next_id) 

    print("\n"+"="*100)
    print("="*100)

if __name__ == "__main__":
    main()