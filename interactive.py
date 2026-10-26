from tokenizer import Tokenizer
import sys, shutil, threading, time, termios
from models import build_model
import random
from decoding import build_decoder
from util.config import load_config



def init_screen(header_lines, footer_lines):
    rows = shutil.get_terminal_size().lines
    top = header_lines + 1     
    bottom = rows - footer_lines
    sys.stdout.write(f"\0337\033[{top};{bottom}r\0338")
    sys.stdout.flush()

def draw_hero():
    cols = shutil.get_terminal_size().columns
    print("="*cols)
    print("""
 ▄▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄  ▄▄▄▄▄▄▄      ▄▄▄▄▄▄▄▄▄  ▄▄▄▄       ▄▄▄▄▄       ▄▄▄▄▄ 
▐░░░░░░░░░░░░▌▐░░░░▌▐░░░░░░░▀▀▄  ▐░░░░▓░░░░▌▐░░░░▌     ▐░░░░░▌     ▐░░░░░▌
▐░░░░░░░░░░░░▌▐░░░░▌▐░░░░░░░░░░▌ ▐░░░░▓░░░░▌▐░░░░▌     ▐░░░░░░▌   ▐░░░░░░▌
 ▀▀▀▐▒▒▒▒▌▀▀▀ ▐▒▒▒▒▌▐▒▒▒▒▒█▒▒▒▒▒▌▐▒▒▒▒▓▒▒▒▒▌▐▒▒▒▒▌     ▐▒▒▒▒▒▒▒▌ ▐▒▒▒▒▒▒▒▌
    ▐▓▓▓▓▌    ▐▓▓▓▓▌▐▓▓▓▓▓█▓▓▓▓▓▌▐▓▓▓▓▓▓▓▓▓▌▐▓▓▓▓▌     ▐▓▓▓▓▓▓▓▓▀▓▓▓▓▓▓▓▓▌
    ▐░░░░▌    ▐░░░░▌▐░░░░░█░░░░░▌ ▀▀█░░░░░░▌▐░░░░▌▄▄▄▄ ▐░░░░▄░░░░░░░▄░░░░▌
    ▐░░░░▌    ▐░░░░▌▐░░░░░▓░░░░░▌ ▄▄▄▄▓░░░░▌▐░░░░█░░░░▌▐░░░░▌▐░░░░░▌▐░░░░▌
    ▐▒▒▒▒▌    ▐▒▒▒▒▌▐▒▒▒▒▒▓▒▒▒▒▒▌▐▒▒▒▒▓▒▒▒▒▌▐░▒▒▒▒▒▒▒▒▌▐▒▒▒▒▌ ▐▒▒▒▌ ▐▒▒▒▒▌
    ▐▓▓▓▓▌    ▐▓▓▓▓▌▐▓▓▓▓▓▓▓▓▓▓▓▌▐▓▓▓▓▓▓▓▓▓▌ ▀▀▒▒▓▓▓▓▓▌▐▓▓▓▓▌  ▐▓▌  ▐▓▓▓▓▌
     ▀▀▀▀      ▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀     ▀▀▀▀▀▀  ▀▀▀▀    ▀    ▀▀▀▀ 
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

def clear_region(top, bottom):
    height = bottom - top + 1
    sys.stdout.write(f"\033[{top};1H\033[{height}S\033[{top};1H")
    sys.stdout.flush()

def spin_until(done, prefix="TinyLM: thinking "):
    SPINNER_FRAMES = ["\\", "|", "/", "-"]
    i = 0
    while not done.is_set():
        sys.stdout.write("\r" + prefix + SPINNER_FRAMES[i % len(SPINNER_FRAMES)])
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(prefix) + 1) + "\r")
    sys.stdout.flush()

def lock_input():
    saved_termios = None
    try:
        fd = sys.stdin.fileno()
        saved_termios = termios.tcgetattr(fd)
        termios.tcflush(fd, termios.TCIFLUSH)
        locked = list(saved_termios)
        locked[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, locked)
    except termios.error:
        pass
    return saved_termios


def unlock_input(saved_termios):
    try:
        fd = sys.stdin.fileno()
        termios.tcflush(fd, termios.TCIFLUSH)
        if saved_termios is not None:
            termios.tcsetattr(fd, termios.TCSANOW, saved_termios)
    except termios.error:
        pass


def main():
    config = load_config("./config.json")

    tokenizer = Tokenizer(n_workers=config["tokenizer"]["concurent_workers"])
    V, id_to_token, merge_rules = tokenizer.load_training_from_file(config["tokenizer"]["path"])
    model_config = config["model"]["bengiolm"]

    model = build_model("bengiolm", V, model_config)
    model.load_model_weights("./weights/model.safetensors")

    decoder = build_decoder(config["decoding"])
    
    sys.stdout.write("\033[2J\033[H")
    draw_hero()
    init_screen(14, 3)                      
    draw_footer()      

    saved_termios = None                

    while True:
        user_input = input("User: ")

        if user_input == ":q": 
            sys.stdout.write("\033[r\033[3J\033[2J\033[H")
            sys.stdout.flush()
            break

        if user_input == ":clear":
            size = shutil.get_terminal_size()
            clear_region(14 + 1, size.lines - 3)
            continue

        text = "<||user||> " + user_input.lower() + " <||bot||>"
        
        row = tokenizer.tokenize_text(text, merge_rules, V)[0]
        
        ids = [V["<||start||>"]] + row  
        generated = list(ids)
        decoding_max_token = config["decoding"]["decoding_max_token"]
        print("\n")
        print("TinyLM: ", end="", flush=True)

        if user_input == "":
            insert_text_messages = [
                "Go ahead and type your message here!", 
                "Let your words flow here", 
                "Share your thoughts below!",
                "Please enter the required information below",
                "Pop your text in!"
            ]

            print(random.choice(insert_text_messages), end="", flush=True)
            print("\n")
            continue
        
        saved_termios = lock_input()
        first_token_done = threading.Event()
        spinner = threading.Thread(
            target=spin_until, args=(first_token_done,), daemon=True
        )
        spinner.start()

        is_first_token = True

        for _ in range(decoding_max_token):
            dist = model.next_token_distribution(generated)
            next_id = decoder(dist)

            if is_first_token:
                first_token_done.set()
                spinner.join()
                print("TinyLM: ", end="", flush=True)
                is_first_token = False

            if next_id is None or V["<||start||>"] == next_id:
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

            print(id_to_token[next_id], end="", flush=True)
            generated.append(next_id) 

        unlock_input(saved_termios)
        print("\n")


if __name__ == "__main__":
    main()

