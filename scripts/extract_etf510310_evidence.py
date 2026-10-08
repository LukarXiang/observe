"""Extract archived PDFs using the existing Windows PDF environment."""

from pathlib import Path

from pdfminer.high_level import extract_text


def main() -> None:
    directory = Path('data/staging/strategies-batch10/ETF证据/20261005-official')
    for path in sorted(directory.glob('*.pdf')):
        text = extract_text(path)
        with path.with_suffix('.txt').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
        print(path.name, len(text), flush=True)


if __name__ == '__main__':
    main()
