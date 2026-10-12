import pyarrow.json as paj
import pyarrow.parquet as pq
import pyarrow as pa

def convert_jsonl_to_parquet_native(input_path, output_path):
    table = paj.read_json(input_path)
    
    schema = pa.schema([('text', pa.string())])
    parse_options = paj.ParseOptions(explicit_schema=schema)
    table = paj.read_json(input_path, parse_options=parse_options)
    
    pq.write_table(table, output_path)

    print(f"Successfully converted {input_path} to {output_path}")


convert_jsonl_to_parquet_native('party-dataset-train-lower.jsonl', 'party-dataset-train.parquet')