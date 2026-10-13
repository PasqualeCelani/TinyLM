import argparse

import sys
import pathlib
sys.path.append("../")

from tokenizer import Tokenizer

def main():
    parser = argparse.ArgumentParser(description="Token counter util script.")

    parser.add_argument("-nw", "--n-workers", help="Number of concurent workers used.", type=int , default=2)
    parser.add_argument("-t", "--tokenizer-path", help="Path of a trained tokenizer", type=str, required=True)
    parser.add_argument("-i", "--input-path", help="Input path of the parquet file.", type=str, required=True)

    args = parser.parse_args()

    tokenizer = Tokenizer(n_workers=args.n_workers)
    V, id_to_token, merge_rules = tokenizer.load_training_from_file(args.tokenizer_path)
    rows = tokenizer.get_text_from_parquet(args.input_path)

    print(f"[INFO] tokens in dataset: {round(sum([len(row) for row in rows])/1000000, 2)}M")


if __name__ == "__main__":
    main()