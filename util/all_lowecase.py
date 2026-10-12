import json

def main():
    records = []
    with open('party-dataset-train.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line: 
                doc = json.loads(line)
                doc["text"] = doc["text"].lower()
                records.append(doc)

    with open('party-dataset-train-lower.jsonl', 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()