import pyarrow.json as paj
import pyarrow.parquet as pq
import pyarrow as pa
import argparse

def main():
    parser = argparse.ArgumentParser(description="normalize all text to lower case.")
    
    parser.add_argument("-i", "--input-path", help="Input path of the jsonl dataset.", type=str, required=True)
    parser.add_argument("-s", "--save-path", help="Output path for jsonl dataset file.", type=str, required=True)

    args = parser.parse_args()

    table = paj.read_json(args.input_path)
    
    schema = pa.schema([('text', pa.string())])
    parse_options = paj.ParseOptions(explicit_schema=schema)
    table = paj.read_json(args.input_path, parse_options=parse_options)
    
    pq.write_table(table, args.save_path)

    print(f"[Success] Successfully converted {args.input_path} to {args.save_path}")

if __name__ == "__main__":
    main()

# python ./util/convert-json2parquet.py -i ./dataset/party-dataset-train-lower.jsonl -s ./dataset/party-dataset-train.parquet