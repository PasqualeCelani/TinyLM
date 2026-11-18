import pyarrow.parquet as pq
from tqdm import tqdm
import json

def main():
    input_path = "./data/train-00000-of-00010.parquet"
    output_path = "../dataset/hf/party-dataset-train.jsonl"

    table = pq.read_table(input_path, columns=['prompt', 'completion'])

    with open(output_path, 'a', encoding='utf-8') as f:
        for i in tqdm(range(len(table)), desc="Counting..."):
            prompt = table["prompt"][i]
            completion = table["completion"][i]
            record = {"text" : "<||user||> " + str(prompt).replace("\"", "'") + " <||bot||> " +  str(completion).replace("\"", "'")}
            f.write('\n' if i == 0 else '' + json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"[Success] Successfully added {input_path} to {output_path}")

if __name__ == "__main__":
    main()