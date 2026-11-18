import argparse

import sys
import pathlib
sys.path.append("../")

from tokenizer import Tokenizer

def main():
    parser = argparse.ArgumentParser(description="Token counter util script.")
    parser.add_argument("-i", "--input-path", help="Input path of the tokenized parquet file.", type=str, required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer()
    rows = tokenizer.get_text_from_parquet(args.input_path)

    print(f"[INFO] tokens in dataset: {round(sum([len(row) for row in rows])/1000000, 2)}M")


if __name__ == "__main__":
    main()