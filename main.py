import duckdb
import re
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

class Tokenizer():
    def __init__(self, filepath):
        self.filepath = filepath
    
    def __get_text_from_parquet(self):
        try:
            rows =  duckdb.execute("""
                SELECT text
                FROM read_parquet(?)
            """, [self.filepath]).fetchall()
        except duckdb.Error as e:
            raise RuntimeError(f"Failed to read parquet at {self.filepath}: {e}") from e
        return rows

    def _count_shard(self, texts):
        local = Counter()
        for text in texts:
            for word in re.split(r"\s+", text[0]):
                local[word] += 1
        return local

    def __word_counter_parallel(self, result, n_workers=4, chunk_size=10000):
        shards = [
            result[i : i + chunk_size]
            for i in range(0, len(result), chunk_size)
        ]

        word_count = Counter()
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for local in tqdm(
                pool.map(self._count_shard, shards),
                total=len(shards),
                desc="Counting words on corpus",
            ):
                word_count.update(local)

        return word_count

    def __create_corpus(self, word_count):
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

    def __build_index(self, corpus):
        pair_to_words = {}
        pair_counts = {}

        for i, (ids, freq) in enumerate(corpus):
            for pair, mult in self.__pairs_of(ids).items():
                pair_counts[pair] = pair_counts.get(pair, 0) + mult * freq
                pair_to_words.setdefault(pair, set()).add(i)

        return pair_to_words, pair_counts

    def __merge_ids(self, ids, left, right, new_id):
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

    def __pairs_of(self, ids):
        return Counter(zip(ids, ids[1:]))

    def train(self, merges):
        result = self.__get_text_from_parquet()
        word_count = self.__word_counter_parallel(result, 6, 10000)
        vocab, id_to_token, corpus = self.__create_corpus(word_count)
        pair_to_words, pair_counts = self.__build_index(corpus)
        merge_rules = []  

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

            merge_rules.append((left, right, new_id))  

            affected = list(pair_to_words.get(best_pair, ()))

            for i in affected:
                ids, freq = corpus[i]

                old_counts = self.__pairs_of(ids)
                new_ids = self.__merge_ids(ids, left, right, new_id)
                new_counts = self.__pairs_of(new_ids)
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

        return set(id_to_token), merge_rules



def main():
    tokenizer = Tokenizer('/mnt/big-disk/Datasets/imdb/unsupervised-00000-of-00001.parquet')
    V, merge_rules = tokenizer.train(1000)
    print(V)
    print(len(V))

if __name__ == "__main__":
    main()