from tokenizer import Tokenizer
from n_gram import NGram
import math

def pack_corpus(rows, V):
    corpus = []
    for row in rows:
        row.insert(0, V["<||start||>"])
        row.append(V["<||end||>"])
        corpus += row
    return corpus

def main():
    tokenizer = Tokenizer(n_workers=2)

    V, id_to_token, merge_rules = tokenizer.load_training_from_file("./dataset/tokenized/tokenizer_train.json")
    rows = tokenizer.get_text_from_parquet("./dataset/tokenized/party-dataset-train_tokenized.parquet")

    packed_corpus = pack_corpus(rows, V)

    print("[INFO] training...")
    n_gram = NGram(4, V)
    n_gram.train(packed_corpus)
    print("[SUCESS] training have been completed.")

    V, id_to_token, merge_rules = tokenizer.load_training_from_file("./dataset/tokenized/tokenizer_validation.json")
    rows = tokenizer.get_text_from_parquet("./dataset/tokenized/party-dataset-validation_tokenized.parquet")

    packed_corpus = pack_corpus(rows, V)
    sequence_length = len(packed_corpus)

    logp = n_gram.log_probability(packed_corpus)
    n_predicted = len(packed_corpus) - (n_gram.n - 1)
    perplexity = math.exp(-logp / n_predicted)
    
    print(f"[INFO] Perplexity: {perplexity}")

if __name__ == "__main__":
    main()