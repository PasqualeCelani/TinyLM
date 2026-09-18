import duckdb
import re
from tqdm import tqdm


def get_text_from_parquet(filepath):
    return duckdb.execute("""
        SELECT text
        FROM read_parquet(?)
        limit 10
    """, [filepath]).fetchall()

def split_ws_words(text):
    return re.split(r"\s+", text)

def word_counter(result):
    word_count = dict()
    for text in tqdm(result, desc="Counting words on corpus"):
        words = split_ws_words(text[0])
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
    return word_count

def create_corpus(word_count):
    vocab = {}
    id_to_token = []

    def get_id(token):
        idx = vocab.get(token)
        if idx is None:
            idx = len(id_to_token)
            vocab[token] = idx
            id_to_token.append(token)
        return idx
    
    corpus = []
    for word, freq in word_count.items():
        ids = [get_id(ch) for ch in word]
        corpus.append([ids, freq])
    
    return vocab, id_to_token, corpus


def find_most_frequent_pair(corpus):
    pair_counts = {}

    for ids, freq in corpus:
        if len(ids) < 2:
            continue
        prev = ids[0]
        for cur in ids[1:]:
            pair = (prev, cur)
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            prev = cur

    if not pair_counts:
        return None, None

    best_pair = None
    best_count = -1
    for pair, cnt in pair_counts.items():
        if cnt > best_count:
            best_count = cnt
            best_pair = pair
    
    return best_pair


def byte_pair_encoding(result, merges):
    word_count = word_counter(result)
    vocab, id_to_token, corpus = create_corpus(word_count)

    for _ in tqdm(range(merges), desc="Merging..."):
        left, right = find_most_frequent_pair(corpus)

        if left is None and right is None:
            break

        new_token = id_to_token[left] + id_to_token[right]
        new_id = len(id_to_token)

        vocab[new_token] = new_id
        id_to_token.append(new_token)

        for item in corpus:
            ids = item[0]
            new_ids = []
            i = 0
            n = len(ids)

            while i < n:
                if i + 1 < n and ids[i] == left and ids[i + 1] == right:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1

            item[0] = new_ids

    return set(id_to_token)



def main():
    result = get_text_from_parquet('/mnt/big-disk/Datasets/imdb/unsupervised-00000-of-00001.parquet')
    V = byte_pair_encoding(result, 1000)
    print(V)
    print(len(V))

if __name__ == "__main__":
    main()