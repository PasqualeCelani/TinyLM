import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="normalize all text to lower case.")

    parser.add_argument("-i", "--input-path", help="Input path of the jsonl dataset.", type=str, required=True)
    parser.add_argument("-s", "--save-path", help="Output path for jsonl dataset file.", type=str, required=True)

    args = parser.parse_args()

    records = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: 
                doc = json.loads(line)
                doc["text"] = doc["text"].lower()
                records.append(doc)

    with open(args.save_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()

# python ./util/all_lowecase.py -i ./dataset/party-dataset-train.jsonl -s ./dataset/party-dataset-train-lower.jsonl