import json

def load_config(path):
    with open(path) as f:
        config = json.load(f)
    return config