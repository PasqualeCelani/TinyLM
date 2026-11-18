# TinyLM

TinyLM is a small, from-scratch autoregressive language-model playground. It includes:

- A BPE tokenizer with special chat tokens.
- An `n-gram` model, a small Bengio-style neural LM [1], and a recurrent neural LM (`rnnlm`) implemented with [tinygrad](https://github.com/tinygrad/tinygrad).
- A terminal chat interface, configurable sampling, and scripts for preparing parquet datasets.

The tested dataset is a tiny "party host" conversation corpus.

<img src="https://github.com/PasqualeCelani/TinyLM/blob/main/showcase.png?raw=true" width="100%">

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 is recommended.

### Bootstrap a new instance

`bootstrap.sh` automates the whole dataset setup on a fresh machine: it downloads the dataset from Hugging Face, creates the virtualenv, installs dependencies, lowercases/converts/tokenizes the corpus, and prints token counts.

```bash
# Ensure the hf token is set if you are accessing a private repository
export HF_TOKEN="hf_your_actual_token_here"

bash bootstrap.sh [dataset_name] [number_workers] [merges]
```

Arguments are optional and default to `PasqualeCelani/TinyLMPartyDataset`, `2`, and `15000`. For example:

```bash
bash bootstrap.sh PasqualeCelani/TinyLMPartyDataset 4 20000
```

## Dataset format

The source data is JSONL: one JSON object per line with a single `text` field.

```json
{"text": "<||user||> hey! is this where the party is at? <||bot||> you just walked right into the middle of it! welcome in!"}
{"text": "<||user||> can you play some music? <||bot||> i'm warming up the virtual turntables right now! what genre are we feeling?"}
```

Turns use the special tokens:

```text
<||user||> ... <||bot||> ...
```

`<||start||>`, `<||end||>`, and `<||pad||>` are handled by the tokenizer/training code. The interactive prompt lowercases input, so lowercase the training text too. The example corpus is in `dataset/party-dataset-train.jsonl` and `dataset/party-dataset-validation.jsonl`.

## Convert JSONL to parquet

The training scripts read parquet files with one string column named `text`.

```bash
# Optional but recommended: lowercase the dataset
PYTHONPATH=. python util/all_lowecase.py \
  -i ./dataset/party-dataset-train.jsonl \
  -s ./dataset/party-dataset-train-lower.jsonl

# Convert JSONL -> parquet
PYTHONPATH=. python util/convert-json2parquet.py \
  -i ./dataset/party-dataset-train-lower.jsonl \
  -s ./dataset/party-dataset-train.parquet
```

Repeat for the validation split:

```bash
PYTHONPATH=. python util/all_lowecase.py \
  -i ./dataset/party-dataset-validation.jsonl \
  -s ./dataset/party-dataset-validation-lower.jsonl

PYTHONPATH=. python util/convert-json2parquet.py \
  -i ./dataset/party-dataset-validation-lower.jsonl \
  -s ./dataset/party-dataset-validation.parquet
```

## Tokenize the dataset

Train the BPE tokenizer on the training parquet and tokenize the corpus:

```bash
cd dataset/tokenized

PYTHONPATH=../.. python ../../util/tokenize_dataset.py \
  -nw 6 \
  -m 15000 \
  -i ../party-dataset-train.parquet \
  -s .
```

This produces:

- `dataset/tokenized/tokenizer_train.json` (vocabulary + merge rules)
- `dataset/tokenized/party-dataset-train_tokenized.parquet` (token IDs)

Then tokenize validation with the already-trained tokenizer:

```bash
PYTHONPATH=../.. python ../../util/tokenize_dataset.py \
  -i ../party-dataset-validation.parquet \
  -s . \
  -t ./tokenizer_train.json
```

Notes:

- `-m` is the number of BPE merges; larger values give a larger vocabulary.
- `-nw` is the number of worker processes used while counting words.
- Run the command from `dataset/tokenized` as shown because the tokenizer is always saved as `./tokenizer_train.json` in the current directory.

Check the token count:

```bash
PYTHONPATH=. python util/token_counter.py \
  -t ./dataset/tokenized/tokenizer_train.json \
  -i ./dataset/tokenized/party-dataset-train_tokenized.parquet
```

## Train

From the repository root:

```bash
python train.py
# or, with the smaller debug config
python train.py --debug
```

`train.py` reads `configs/base.json` (`configs/debug.json` with `--debug`), loads the tokenizer and tokenized parquet files, adds `<||start||>` / `<||end||>` delimiters, and trains the `rnnlm` model. The best checkpoint (by validation perplexity) is written to `./model.safetensors`.

To use it in the chat interface, copy it to the expected weights path:

```bash
mkdir -p weights
cp ./model.safetensors ./weights/model.safetensors
```

## Configuration

Edit `configs/base.json` (or `configs/debug.json` for quick runs, selected with `--debug`) to change paths, model size, and decoding behavior.

```json
{
  "tokenizer": {
    "concurent_workers": 6,
    "path": "./dataset/tokenized/tokenizer_train.json",
    "training_dataset_path": "./dataset/tokenized/party-dataset-train_tokenized.parquet",
    "validation_dataset_path": "./dataset/tokenized/party-dataset-validation_tokenized.parquet"
  },
  "model": {
    "n-gram": {
      "n": 5,
      "delta": 1e-8
    },
    "bengiolm": {
      "params": {
        "embeeding_dim": 64,
        "hiddend_dim": 128,
        "ctx_size": 8
      },
      "training": {
        "epochs": 3,
        "batch_size": 256,
        "lr": 0.3,
        "weight_decay": 1e-4,
        "eval_steps": 500,
        "eval_batch_size": 1024
      }
    },
    "rnnlm": {
      "params": {
        "embeeding_dim": 128,
        "hiddend_dim": 256
      },
      "training": {
        "epochs": 15,
        "batch_size": 32,
        "block_len": 30,
        "lr": 0.3,
        "weight_decay": 1e-4,
        "momentum": 0.8,
        "eval_steps": 300,
        "eval_batch_size": 32,
        "scheduling" : {
          "type" : "StepLR",
          "params" : {
              "step_size" : 5,
              "gamma" : 0.5
          }
        }
      }
    }
  },
  "decoding": {
    "name": "sample",
    "decoding_params": { "temperature": 0.6 },
    "decoding_max_token": 100
  }
}
```

Key settings:

- `concurent_workers`: tokenizer worker count.
- `embeeding_dim`, `hiddend_dim`, `ctx_size`: embedding size, hidden layer size, and context window (`ctx_size` is only used by `bengiolm`). These key spellings are intentional; keep them as-is.
- `block_len`: sequence length used by `rnnlm` for truncated backpropagation.
- `epochs`, `batch_size`, `lr`, `weight_decay`, `momentum`: optimization settings.
- `scheduling`: `rnnlm` learning-rate schedule (for example, `StepLR` with `step_size` and `gamma`. avaible schedulers in `./models/schedulers.py`).
- `eval_steps`: evaluate every N batches. Set to `0` to skip validation.
- `eval_batch_size`: batch size used for evaluation.
- `decoding.name`: `"greedy"` or `"sample"`.
- `decoding_params.temperature`: sampling temperature for `"sample"` (optionally `top_k`).
- `decoding_max_token`: maximum tokens generated per reply.

The model registry also contains an `n-gram` model (`n`, `delta`) and `bengiolm`, but `train.py` currently instantiates `rnnlm`.

## Interactive mode

```bash
python interactive.py
# or
python interactive.py --debug
```

Requires the tokenizer at the configured path and model weights at `./weights/model.safetensors`. Commands inside the chat:

- `:clear` clears the chat area.
- `:q` quits.

## Project layout

```text
configs/                 paths and hyperparameters (base and debug)
train.py                 training entry point
interactive.py           terminal chat interface
tokenizer.py             BPE tokenizer implementation
decoding.py              greedy / temperature sampling decoders
models/                  model registry, n-gram, Bengio LM, RNN LM
util/                    dataset conversion, tokenization, token counting
dataset/                 source JSONL and generated parquet files
weights/                 model weights used by interactive.py
```
## References
[1] Bengio, Yoshua, et al. "A neural probabilistic language model." Journal of machine learning research 3.Feb (2003): 1137-1155.

## License 
This project is released under the MIT License.
