import json
import os
from pathlib import Path
import random
from rich.console import Console, OverflowMethod


def select_quote(file=Path(os.path.dirname(__file__)) / "res/Quotes.json"):
    with open(file, "r") as f:
        data = json.load(f)
        n_quotes = len(data)
        idx_quote = random.randint(0, n_quotes - 1)
        quote_json = data[idx_quote]
    return quote_json


def print_quote():
    quote_json = select_quote()
    console = Console(width=70)
    console.rule(f"Book {quote_json['Book']}", style="red")
    text = f"{quote_json['Quote']}"
    console.print(text, justify="full")
    console.rule(style="red")

if __name__ == "__main__":
    print_quote()
