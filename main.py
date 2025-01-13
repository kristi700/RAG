import argparse

from utils import pdf_processor

# NOTE - add LLM choice
def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments for RAG.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Settings for RAG")
    parser.add_argument("pdf file", help="PDF file to use as context for the LLM")
    return parser.parse_known_args()

def main():
    args, modifiers = parse_args()


if __name__ == "__main__":
    main()