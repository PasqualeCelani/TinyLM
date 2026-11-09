from tokenizer import Tokenizer
from models import build_model
from util.config import load_config



def add_delimeters(rows, V):
    return [[V["<||start||>"]] + row + [V["<||end||>"]]  for row in rows]
    

def main():
    config = load_config("./config.json")

    tokenizer = Tokenizer(n_workers=config["tokenizer"]["concurent_workers"])
    V, id_to_token, merge_rules = tokenizer.load_training_from_file(config["tokenizer"]["path"])

    rows_training = tokenizer.get_text_from_parquet(config["tokenizer"]["training_dataset_path"])
    
    corpus_training = add_delimeters(rows_training, V)


    model_config = config["model"]["rnnlm"]

    corpus_validation = None
    if model_config["training"]["eval_steps"] > 0:
        rows_validation = tokenizer.get_text_from_parquet(config["tokenizer"]["validation_dataset_path"])
        corpus_validation = add_delimeters(rows_validation, V)
    
    model = build_model("rnnlm", V, model_config)
    model.train(corpus_training, corpus_validation)

if __name__ == "__main__":
    main()