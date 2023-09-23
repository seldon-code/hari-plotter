import json
import os
from pathlib import Path
import random
from rich import print


def select_quote(file=Path(os.path.dirname(__file__)) / "res/Quotes.json"):
    with open(file, "r") as f:
        data = json.load(f)
        n_quotes = len(data)
        idx_quote = random.randint(0, n_quotes - 1)
        quote_json = data[idx_quote]
    return quote_json


def print_quote():
    quote_json = select_quote()
    text = f"{quote_json['Quote']} (Book {quote_json['Book']})"
    print(text)


if __name__ == "__main__":
    print_quote()
