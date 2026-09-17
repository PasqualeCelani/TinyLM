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

def byte_pair_encoding(result, merges):
    word_count = dict()
    V = set()

    for text in tqdm(result, desc="Counting words on corpus"):
        words = split_ws_words(text[0])
        for word in words:
            V = V.union(set(word))
            word_count[word] = word_count.get(word, 0) + 1
    
    corpus = [[" ".join(list(word)), word_count[word]] for word in word_count.keys()]

    for _ in tqdm(range(merges), desc="Merging..."):
        pair_counter = dict()
        max_counter = -1
        max_left = None
        max_right = None

        for c in corpus:
            value = c[1]
            text = c[0].split(" ")
            for i in range(len(text)-1):
                left = text[i]
                right = text[i+1]
                pair_counter[(left, right)] = pair_counter.get((left, right), 0) + value
                if pair_counter[(left, right)]  > max_counter:
                    max_counter = pair_counter[(left, right)] 
                    max_left = left
                    max_right = right
        
        if max_left is None or max_right is None:
            continue

        V.add(max_left+max_right)
        corpus = [
            [c[0].replace(f"{max_left} {max_right}", max_left+max_right), c[1]] 
            for c in corpus
        ]

    return V



def main():
    result = get_text_from_parquet('/mnt/big-disk/Datasets/imdb/unsupervised-00000-of-00001.parquet')
    V = byte_pair_encoding(result, 100000)
    print(V)

if __name__ == "__main__":
    main()