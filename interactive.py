from tokenizer import Tokenizer
from collections import Counter
import sys, shutil

def pack_corpus(rows, V):
    corpus = []
    for row in rows:
        row.insert(0, V["<||start||>"])
        row.append(V["<||end||>"])
        corpus += row
    return corpus


def init_screen(bar_lines=2):
    rows = shutil.get_terminal_size().lines
    sys.stdout.write(f"\0337\033[1;{rows - bar_lines}r\0338")
    sys.stdout.flush()

def draw_hero():
    cols = shutil.get_terminal_size().columns
    print("="*cols)
    print("""
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄   ▄▄▄▄▄▄▄▄▄▄   ▄▄▄    ▄▄▄   ▄▄▄▄▄▄▄▄▄▄   ▄▄▄        
▒          █ ▒   █ ▒          █ ▒   █  █   █ ▒          █ ▒   █       
▀▀▀▀▀▀▀▀▀▀▀▀ ▀▀▀▀▀ ▀▀▀▀▀▀▀█   █ ▓   █  █   █ ▀▀▀▀▀▀▀█   █ ▀   █       
    ▄▄▄      █▀▀▀█ █▀▀▀█  ▒   █ █   ▀▄▄▀   █ █▀▀▀█  ▒   █ ▒   █       
   ▒   █     ▒   █ ▒   █  ▓   █ ▒          █ ▒   █  ▓   █ ▒   █       
   ▓   █     ▓   █ ▓   █  ▓   █ ▀▄▄▄▄▄▄▄   █ ▓   █  ▓   █ ▓   █       
   ░   ▄     ░   ▄ ░   ▄  █   █  ▄▄▄   █   █ ░   ▄  █   █ ░   █       
   ▒   █     ▒   █ ▒   █  ▀   █ ▒   █▄▄▀   █ ▒   █  ▀   █ ▒   ▀▀▀▀▀▀▀█
   ▓   █     ▓   █ ▓   █  █   █ ▓          █ ▓   █  █   █ ▓          █
   ▀▀▀▀▀     ▀▀▀▀▀  ▀▀▀   ▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀   ▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀▀
    """)
    print("="*cols)

def draw_footer():
    size = shutil.get_terminal_size()
    rows, cols = size.lines, size.columns
    sys.stdout.write(f"\0337\033[{rows - 2};1H\033[J")
    print("=" * cols)
    print("© Made by Celani software")
    print("=" * cols, end="")
    sys.stdout.write("\0338")
    sys.stdout.flush()

def train_n_gram(corpus, n=4):
    nominator =  Counter()
    denominator = Counter()

    for i in range(n-1, len(corpus)):
        denominator[tuple(corpus[i-(n-1):i])] += 1
        nominator[tuple(corpus[i-(n-1):i+1])] += 1

    ngram = {}
    for key, value in nominator.items():
        ngram[key] = value/denominator[key[:-1]] if denominator[key[:-1]] != 0 else 0
    
    return ngram

def get_next_token_id_n_gram(ngram, V, generated, n):
    next_id = None
    max_p = 0

    for w, w_id in V.items():
        p = ngram[tuple(generated[-(n-1):])+ (w_id,)] if tuple(generated[-(n-1):])+(w_id,) in ngram.keys() else 0
        if p > max_p:
            max_p = p
            next_id = w_id

    return next_id

def main():
    tokenizer = Tokenizer(n_workers=2)
    V, id_to_token, merge_rules = tokenizer.load_training_from_file("./dataset/tokenized/part-dataset-tokenizer_train.json")
    rows = tokenizer.get_text_from_parquet("./dataset/tokenized/party-dataset_tokenized.parquet")

    print("[INFO] packing corpus...")
    packed_corpus = pack_corpus(rows, V)
    print("[SUCCESS] corpus packed.")

    n = 4
    n_gram = train_n_gram(packed_corpus, n)
    
    init_screen(bar_lines=3)
    draw_hero()
    draw_footer()

    while True:

        user_input = input("User: ")

        if user_input == ":q": 
            sys.stdout.write("\033[r\033[3J\033[2J\033[H")
            sys.stdout.flush()
            break

        text = "<||user||> " + user_input + "<||bot||>"
        
        row = tokenizer.tokenize_text(text, merge_rules, V)[0]
        
        ids = [V["<||start||>"]] + row  
        generated = list(ids)
        decoding_max_token = 100
        print("\n")
        print("TinyLM: ", end=" ", flush=True)
        for _ in range(decoding_max_token):
            next_id = get_next_token_id_n_gram(n_gram, V, generated, n)

            if next_id is None:
                print("I'm sorry, but that's a bit beyond what I can do!", end=" ", flush=True)
                break

            if next_id == V["<||end||>"]:
                break

            print(id_to_token[next_id], end=" ", flush=True)
            generated.append(next_id) 

        print("\n")


if __name__ == "__main__":
    main()

