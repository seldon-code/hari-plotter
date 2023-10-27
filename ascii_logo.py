import os
from pathlib import Path
from rich_pixels import Pixels
from PIL import Image
from rich.console import Console


def print_logo(text=False, width=60):
    console = Console()

    if not text:
        file = Path(os.path.dirname(__file__)) / "res/logo.png"
    else:
        file = Path(os.path.dirname(__file__)) / "res/logo_text.png"

    with Image.open(file) as image:
        new_size = [int(width), int(width / image.size[0] * image.size[1])]
        image = image.resize(new_size, Image.Resampling.BICUBIC)
        pixels = Pixels.from_image(image)

    console.print(pixels, overflow="crop")


if __name__ == "__main__":
    print_logo()
    print_logo(True)
