import io
from math import ceil
from pathlib import Path
from typing import List, BinaryIO

import tiktoken
from dataclasses import dataclass
from html2text import html2text
import fitz
import pypdf


@dataclass
class Text:
    text: str
    page: str


def parse_pdf_fitz(file: BinaryIO, chunk_chars: int, overlap: int) -> List[Text]:
    file = fitz.open(stream=file, filetype="pdf")
    split = ""
    pages: List[str] = []
    texts: List[Text] = []
    for i in range(file.page_count):
        page = file.load_page(i)
        split += page.get_text("text", sort=True)
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            texts.append(Text(text=split[:chunk_chars], page=pg))
            split = split[chunk_chars - overlap:]
            pages = [str(i + 1)]
    if len(split) > overlap or len(texts) == 0:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(Text(text=split[:chunk_chars], page=pg))
    file.close()
    return texts


def parse_pdf(file: BinaryIO, chunk_chars: int, overlap: int) -> List[Text]:
    pdfReader = pypdf.PdfReader(file)
    split = ""
    pages: List[str] = []
    texts: List[Text] = []
    for i, page in enumerate(pdfReader.pages):
        split += page.extract_text()
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            texts.append(Text(text=split[:chunk_chars], page=pg))
            split = split[chunk_chars - overlap:]
            pages = [str(i + 1)]
    if len(split) > overlap or len(texts) == 0:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], page=pg)
        )
    return texts


def parse_txt(
    file: BinaryIO, chunk_chars: int, overlap: int, html: bool = False
) -> List[Text]:
    """Parse a document into chunks, based on tiktoken encoding.

    NOTE: We get some byte continuation errors.
    Currnetly ignored, but should explore more to make sure we
    don't miss anything.
    """
    try:
        f = io.TextIOWrapper(file)
        text = f.read()
    except UnicodeDecodeError:
        f = io.TextIOWrapper(file, encoding="utf-8", errors="ignore")
        text = f.read()
    if html:
        text = html2text(text)
    texts: list[Text] = []
    # we tokenize using tiktoken so cuts are in reasonable places
    # See https://github.com/openai/tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    encoded = enc.encode_ordinary(text)
    split = []
    # convert from characters to chunks
    char_count = len(text)  # e.g., 25,000
    token_count = len(encoded)  # e.g., 4,500
    chars_per_token = char_count / token_count  # e.g., 5.5
    chunk_tokens = chunk_chars / chars_per_token  # e.g., 3000 / 5.5 = 545
    overlap_tokens = overlap / chars_per_token  # e.g., 100 / 5.5 = 18
    chunk_count = ceil(token_count / chunk_tokens)  # e.g., 4500 / 545 = 9
    for i in range(chunk_count):
        split = encoded[
            max(int(i * chunk_tokens - overlap_tokens), 0) : int(
                (i + 1) * chunk_tokens + overlap_tokens
            )
        ]
        texts.append(Text(text=enc.decode(split), page=f"{i + 1}"))
    return texts


def parse_code_txt(file: BinaryIO, chunk_chars: int, overlap: int) -> List[Text]:
    """Parse a document into chunks, based on line numbers (for code)."""

    split = ""
    texts: List[Text] = []
    last_line = 0

    f = io.TextIOWrapper(file)
    for i, line in enumerate(f):
        split += line
        while len(split) > chunk_chars:
            texts.append(
                Text(text=split[:chunk_chars], page=f"{last_line}-{i}"))
            split = split[chunk_chars - overlap:]
            last_line = i
    if len(split) > overlap or len(texts) == 0:
        texts.append(Text(text=split[:chunk_chars], page=f"{last_line}-{i}"))
    return texts


def read_doc(
    file: BinaryIO,
    mime_type: str,
    chunk_chars: int = 3000,
    overlap: int = 100,
    force_pypdf: bool = False,
) -> List[Text]:
    """Parse a document into chunks."""
    if mime_type == "application/pdf":
        if force_pypdf:
            return parse_pdf(file, chunk_chars, overlap)
        try:
            return parse_pdf_fitz(file, chunk_chars, overlap)
        except ImportError:
            return parse_pdf(file, chunk_chars, overlap)
    elif mime_type == "text/plain":
        return parse_txt(file, chunk_chars, overlap)
    elif mime_type == "text/html":
        return parse_txt(file, chunk_chars, overlap, html=True)
    else:
        return parse_code_txt(file, chunk_chars, overlap)
