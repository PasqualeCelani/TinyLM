import argparse

import sys
import pathlib
sys.path.append("../")

from tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Tokenizer script of parquet file.")

    parser.add_argument("-nw", "--n-workers", help="Number of concurent workers used.", type=int , default=2)
    parser.add_argument("-m", "--merges", help="Number of BPE merges iterations.", type=int , default=15000)
    parser.add_argument("-i", "--input-path", help="Input path of the parquet file.", type=str, required=True)
    parser.add_argument("-s", "--save-path", help="Output path for saving the parquet file.", type=str, required=True)
    parser.add_argument("-t", "--tokenizer-path", help="Path of a trained tokenizer", type=str)

    args = parser.parse_args()

    tokenizer = Tokenizer(n_workers=args.n_workers)

    if not args.tokenizer_path:
        V, id_to_token, merge_rules = tokenizer.train(args.input_path, merges=args.merges)
        tokenizer.save_training(V, id_to_token, merge_rules)
    else:
        V, id_to_token, merge_rules = tokenizer.load_training_from_file(args.tokenizer_path)

    tokenizer.tokenize_parquet(args.input_path, merge_rules, V, save_path=args.save_path)

if __name__ == "__main__":
    main()