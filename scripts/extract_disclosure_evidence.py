"""Extract new disclosure PDFs using the already installed PDF environment."""

import argparse
from pathlib import Path

from pdfminer.high_level import extract_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    for path in sorted(args.directory.glob('*.pdf')):
        with path.with_suffix('.txt').open('x', encoding='utf-8', newline='\n') as stream:
            text = extract_text(path)
            stream.write(text)
        print(path.name, len(text), flush=True)


if __name__ == '__main__':
    main()
