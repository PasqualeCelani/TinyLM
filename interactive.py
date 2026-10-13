from tokenizer import Tokenizer
import sys, shutil
from n_gram import NGram
import random

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



def main():
    tokenizer = Tokenizer(n_workers=2)
    V, id_to_token, merge_rules = tokenizer.load_training_from_file("./dataset/tokenized/tokenizer_train.json")
    rows = tokenizer.get_text_from_parquet("./dataset/tokenized/party-dataset-train_tokenized.parquet")

    packed_corpus = pack_corpus(rows, V)

    n_gram = NGram(5, V)
    n_gram.train(packed_corpus)
    
    init_screen(bar_lines=3)
    draw_hero()
    draw_footer()

    while True:

        user_input = input("User: ")

        if user_input == ":q": 
            sys.stdout.write("\033[r\033[3J\033[2J\033[H")
            sys.stdout.flush()
            break

        text = "<||user||> " + user_input.lower() + "<||bot||>"
        
        row = tokenizer.tokenize_text(text, merge_rules, V)[0]
        
        ids = [V["<||start||>"]] + row  
        generated = list(ids)
        decoding_max_token = 100
        print("\n")
        print("TinyLM: ", end=" ", flush=True)

        if user_input == "":
            insert_text_messages = [
                "Go ahead and type your message here!", 
                "Let your words flow here", 
                "Share your thoughts below!",
                "Please enter the required information below",
                "Pop your text in!"
            ]

            print(random.choice(insert_text_messages), end=" ", flush=True)
            print("\n")
            continue

        for _ in range(decoding_max_token):
            next_id = n_gram.get_next_token_id(generated)

            if next_id is None:
                fallback_response_messages = [
                    "I'm sorry, but that's a bit beyond what I can do!",
                    "Oops, you've stumped me! That's a little out of my wheelhouse right now.",
                    "I wish I could help with that, but it's just outside my current capabilities.",
                    "I'm sorry, but that's a bit beyond my expertise!",
                    "Hmm, I'm not quite sure how to answer that one!"
                ]
                print(random.choice(fallback_response_messages), end=" ", flush=True)
                break

            if next_id == V["<||end||>"] or next_id == V["<||bot||>"]:
                break

            print(id_to_token[next_id], end=" ", flush=True)
            generated.append(next_id) 

        print("\n")


if __name__ == "__main__":
    main()

