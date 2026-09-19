import duckdb
import re
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


def get_text_from_parquet(filepath):
    return duckdb.execute("""
        SELECT text
        FROM read_parquet(?)
    """, [filepath]).fetchall()

def _count_shard(texts):
    local = Counter()
    for text in texts:
        for word in re.split(r"\s+", text[0]):
            local[word] += 1
    return local

def word_counter_parallel(result, n_workers=4, chunk_size=10000):
    shards = [
        result[i : i + chunk_size]
        for i in range(0, len(result), chunk_size)
    ]

    word_count = Counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for local in tqdm(
            pool.map(_count_shard, shards),
            total=len(shards),
            desc="Counting words on corpus",
        ):
            word_count.update(local)

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

def build_index(corpus):
    pair_to_words = {}
    pair_counts = {}

    for i, (ids, freq) in enumerate(corpus):
        for pair, mult in pairs_of(ids).items():
            pair_counts[pair] = pair_counts.get(pair, 0) + mult * freq
            pair_to_words.setdefault(pair, set()).add(i)

    return pair_to_words, pair_counts

def merge_ids(ids, left, right, new_id):
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and ids[i] == left and ids[i + 1] == right:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def pairs_of(ids):
    return Counter(zip(ids, ids[1:]))

def byte_pair_encoding(result, merges):
    word_count = word_counter_parallel(result, 6, 10000)
    vocab, id_to_token, corpus = create_corpus(word_count)
    pair_to_words, pair_counts = build_index(corpus)

    for _ in tqdm(range(merges), desc="Merging..."):
        if not pair_counts:
            break

        best_pair = None
        best_count = -1
        for pair, cnt in pair_counts.items():
            if cnt > best_count or (cnt == best_count and (best_pair is None or pair < best_pair)):
                best_pair, best_count = pair, cnt

        left, right = best_pair

        new_token = id_to_token[left] + id_to_token[right]
        new_id = len(id_to_token)
        vocab[new_token] = new_id
        id_to_token.append(new_token)

        affected = list(pair_to_words.get(best_pair, ()))

        for i in affected:
            ids, freq = corpus[i]

            old_counts = pairs_of(ids)
            new_ids = merge_ids(ids, left, right, new_id)
            new_counts = pairs_of(new_ids)
            corpus[i][0] = new_ids

            for pair in old_counts.keys() | new_counts.keys():
                delta = (new_counts.get(pair, 0) - old_counts.get(pair, 0)) * freq

                if delta:
                    v = pair_counts.get(pair, 0) + delta
                    if v > 0:
                        pair_counts[pair] = v
                    else:
                        pair_counts.pop(pair, None)

                if new_counts.get(pair, 0) > 0:
                    pair_to_words.setdefault(pair, set()).add(i)
                else:
                    s = pair_to_words.get(pair)
                    if s is not None:
                        s.discard(i)
                        if not s:
                            del pair_to_words[pair]

    return set(id_to_token)



def main():
    result = get_text_from_parquet('/mnt/big-disk/Datasets/imdb/unsupervised-00000-of-00001.parquet')
    V = byte_pair_encoding(result, 1000)
    print(V)
    print(len(V))

if __name__ == "__main__":
    main()