import argparse

from PyPDF2 import PdfReader, PdfWriter

def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments for RAG.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Settings for RAG")
    parser.add_argument("pdf_file", help="PDF file path to shorten.")
    parser.add_argument("output_name", help="Shortened files name.")
    parser.add_argument("start_page", type=int, help="Shortened files name.")
    parser.add_argument("end_page", type=int, help="Shortened files name.")
    return parser.parse_known_args()

def shorten_pdf(pdf_path: str, output_filename: str, start_page: int, end_page: int):
    """
    Saves start_page + page_count number of pages from the provided PDF.
    """
    assert start_page <= end_page, "Start page should be smaller than end page!"

    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    
    end_index = min(end_page, len(reader.pages))

    for page_num in range(start_page-1, end_index):
        writer.add_page(reader.pages[page_num])

    with open(f'{output_filename}.pdf', "wb") as output_pdf:
        writer.write(output_pdf)

    print(f"Extracted pages {start_page}-{end_index} from the PDF and saved to {output_filename}.")

def main():
    args, _ = parse_args()
    shorten_pdf(args.pdf_file, args.output_name, args.start_page, args.end_page)


if __name__ == "__main__":
    main()