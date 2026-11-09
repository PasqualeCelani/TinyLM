#!/bin/bash
set -e

dataset_name="${1:-PasqualeCelani/TinyLMPartyDataset}"
number_workers="${2:-2}"
merges="${3:-15000}"


mkdir -p dataset/tokenized

python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt

hf download "$dataset_name" --repo-type=dataset --local-dir .

python ./util/all_lowecase.py -i ./dataset/party-dataset-train.jsonl -s ./dataset/party-dataset-train-lower.jsonl
python ./util/all_lowecase.py -i ./dataset/party-dataset-validation.jsonl -s ./dataset/party-dataset-validation-lower.jsonl

python ./util/convert-json2parquet.py -i ./dataset/party-dataset-train-lower.jsonl -s ./dataset/party-dataset-train.parquet
python ./util/convert-json2parquet.py -i ./dataset/party-dataset-validation-lower.jsonl -s ./dataset/party-dataset-validation.parquet

PYTHONPATH=. python ./util/tokenize_dataset.py -nw $number_workers -m $merges -i ./dataset/party-dataset-train.parquet -s ./dataset/tokenized
PYTHONPATH=. python ./util/tokenize_dataset.py -nw $number_workers -m $merges -i ./dataset/party-dataset-validation.parquet -s ./dataset/tokenized -t ./tokenizer_train.json

PYTHONPATH=. python ./util/token_counter.py -nw $number_workers -t ./tokenizer_train.json -i ./dataset/tokenized/party-dataset-train_tokenized.parquet
PYTHONPATH=. python ./util/token_counter.py -nw $number_workers -t ./tokenizer_train.json -i ./dataset/tokenized/party-dataset-validation_tokenized.parquet

mv ./tokenizer_train.json ./dataset/tokenized