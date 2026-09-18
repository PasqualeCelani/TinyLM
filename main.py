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


def _find_most_frequent_pair_shard(corpus):
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
        return None

    best_pair = None
    best_count = -1
    for pair, cnt in pair_counts.items():
        if cnt > best_count:
            best_pair, best_count = pair, cnt

    return best_pair, best_count


def find_most_frequent_pair_parallel(pool, corpus, chunk_size=10000):
    shards = [
        corpus[i : i + chunk_size]
        for i in range(0, len(corpus), chunk_size)
    ]

    best_pair = None
    best_count = -1

    for result in pool.map(_find_most_frequent_pair_shard, shards):
        if result is None:
            continue

        pair, count = result

        if (
            best_pair is None
            or count > best_count
            or (count == best_count and pair < best_pair)
        ):
            best_pair, best_count = pair, count

    return best_pair

def _apply_merge_shard(args):
    shard, left, right, new_id = args
    out = []

    for ids, freq in shard:
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
        out.append((new_ids, freq))

    return out


def apply_merge_parallel(pool, corpus, left, right, new_id, chunk_size=10000):
    shards = [
        corpus[i : i + chunk_size]
        for i in range(0, len(corpus), chunk_size)
    ]

    args = [(shard, left, right, new_id) for shard in shards]

    new_corpus = []

    for new_shard in pool.map(_apply_merge_shard, args):
        new_corpus.extend(new_shard)

    return new_corpus

def byte_pair_encoding(result, merges):
    word_count = word_counter_parallel(result, 6, 10000)
    vocab, id_to_token, corpus = create_corpus(word_count)

    with ProcessPoolExecutor(max_workers=6) as pool:
        for _ in tqdm(range(merges), desc="Merging..."):
            frequnt_pair = find_most_frequent_pair_parallel(pool, corpus, 15000)

            if frequnt_pair is None: break
            left, right = frequnt_pair

            new_token = id_to_token[left] + id_to_token[right]
            new_id = len(id_to_token)

            vocab[new_token] = new_id
            id_to_token.append(new_token)

            corpus = apply_merge_parallel(pool, corpus, left, right, new_id,  15000)

    return set(id_to_token)



def main():
    result = get_text_from_parquet('/mnt/big-disk/Datasets/imdb/unsupervised-00000-of-00001.parquet')
    V = byte_pair_encoding(result, 1000)
    print(V)
    print(len(V))

if __name__ == "__main__":
    main()