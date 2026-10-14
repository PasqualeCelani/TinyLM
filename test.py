from tokenizer import Tokenizer
from models import build_model
import math
from util.config import load_config

def pack_corpus(rows, V):
    corpus = []
    for row in rows:
        row.insert(0, V["<||start||>"])
        row.append(V["<||end||>"])
        corpus += row
    return corpus

def main():
    config = load_config("./config.json")

    tokenizer = Tokenizer(n_workers=config["tokenizer"]["concurent_workers"])
    V, id_to_token, merge_rules = tokenizer.load_training_from_file(config["tokenizer"]["path"])
    rows = tokenizer.get_text_from_parquet(config["tokenizer"]["training_dataset_path"])

    packed_corpus = pack_corpus(rows, V)

    print("[INFO] training...")
    model_config = config["model"]["n-gram"]
    model = build_model("ngram", V, model_config)
    model.train(packed_corpus)
    print("[SUCESS] training have been completed.")

    rows = tokenizer.get_text_from_parquet(config["tokenizer"]["validation_dataset_path"])
    packed_text = pack_corpus(rows, V)
    
    logp = model.log_probability(packed_text)
    n_predicted = len(packed_text) - getattr(model, "n", 1)
    perplexity =  math.exp(-logp / n_predicted)

    
    print(f"[INFO] Perplexity: {perplexity}")

if __name__ == "__main__":
    main()