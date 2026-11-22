"""Wczytywanie tekstu z plików przesłanych przez użytkownika.

Obsługujemy:
- pliki tekstowe (.txt, .md, .csv, .log, .json) – bez dodatkowych bibliotek,
- PDF (PyMuPDF) i Word .docx (python-docx) – najczęstsze formaty umów
  i dokumentacji medycznej. W obu tekst z obrazów odczytuje OCR (engine/ocr.py).

Wszystko dzieje się lokalnie, w pamięci. Plik nie jest nigdzie zapisywany.
"""

import io
import unicodedata
from pathlib import Path

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json"}
# Znak „nowej strony” (form feed) – oddziela strony PDF w jednym tekście.
PAGE_BREAK = "\f"
SUPPORTED_EXTENSIONS = sorted(TEXT_EXTENSIONS | {".pdf", ".docx"})

# Kolejność ma znaczenie: UTF-8 jest najczęstsze, a cp1250 to stare polskie
# pliki z Windows (bez tego „ł” i „ś” zamieniłyby się w krzaki).
ENCODINGS = ["utf-8-sig", "utf-8", "cp1250", "iso-8859-2"]


class FileReadError(Exception):
    """Pliku nie da się odczytać (zły format, uszkodzony, pusty)."""


def decode_text(data: bytes) -> str:
    """Zamienia bajty na tekst, próbując kolejnych kodowań."""
    for encoding in ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise FileReadError("Nie rozpoznano kodowania pliku tekstowego.")


# Znaki, które w PDF udają zwykły myślnik „-”. Generatory PDF (Word, platformy
# podpisu) często zapisują myślnik jako SOFT HYPHEN (U+00AD). Wtedy NIP
# „526-025-02-74”, UUID czy data „2025-03-12” wyglądają tak samo na ekranie,
# ale regex ich nie znajduje – w testowej Karcie Podpisów dane by wyciekły.
HYPHEN_LOOKALIKES = {
    "­": "-",  # SOFT HYPHEN
    "‐": "-",  # HYPHEN
    "‑": "-",  # NON-BREAKING HYPHEN
    "−": "-",  # MINUS SIGN
    "​": "",   # ZERO WIDTH SPACE (niewidoczny, rozbija słowa)
    "﻿": "",   # BOM w środku tekstu
}
_HYPHEN_TABLE = str.maketrans(HYPHEN_LOOKALIKES)


def normalize_text(text: str) -> str:
    """Ujednolica tekst, zanim zobaczy go Regex i model.

    Bez tego dane można „przemycić” obok wzorców, np. PESEL z niewidoczną
    spacją w środku („9203150\\u200b2349”) wygląda normalnie, ale regex
    go nie widzi i dane trafiają do chmury.

    1. Podróbki myślnika -> „-” (SOFT HYPHEN z PDF itd.).
    2. NFKC: cyfry pełnej szerokości („９２０…”) -> zwykłe cyfry, twarda spacja
       -> spacja, ligatury („ﬁ”) -> „fi”.
    3. Usunięcie pozostałych znaków formatujących Unicode (kategoria Cf:
       zero-width joiner, znaczniki kierunku tekstu itp.).
    Znaki sterujące (nowa linia, tabulator, „\\f” = nowa strona) zostają.
    """
    text = unicodedata.normalize("NFKC", text.translate(_HYPHEN_TABLE))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


# Dawna nazwa – PDF używa tej samej, pełnej normalizacji.
normalize_pdf_text = normalize_text


def read_pdf_pages(data: bytes, ocr: bool = True) -> list[str]:
    """Zwraca tekst PDF jako listę stron (jedna pozycja = jedna strona).

    ocr=True: tekst rozpoznany na OBRAZACH strony (skan, pieczątka, zdjęcie
    dokumentu) jest doklejany na końcu tej strony – patrz engine/ocr.py.
    Strony bez obrazów nie są przepuszczane przez OCR.
    """
    # PyMuPDF – ta sama biblioteka, która potem zaczernia dane w PDF.
    # Dzięki temu tekst, w którym szukamy encji, jest identyczny z tym,
    # który redaktor znajdzie na stronie.
    import pymupdf

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
        # sort=True układa bloki tekstu w kolejności czytania (z góry na dół,
        # od lewej do prawej). Bez tego tabele, kolumny i pieczęcie w Kartach
        # Podpisów (np. Autenti) wychodzą w kolejności, w jakiej generator
        # zapisał je w pliku – często rozrzucone po całym tekście.
        pages = [normalize_pdf_text(page.get_text("text", sort=True)).strip()
                 for page in document]
    except (RuntimeError, ValueError) as error:
        raise FileReadError(f"Uszkodzony plik PDF: {error}") from error

    if ocr:
        from engine.ocr import ocr_pdf

        report = ocr_pdf(data)
        pages = [
            "\n".join(part for part in (page, normalize_pdf_text(extra).strip()) if part)
            for page, extra in zip(pages, report.texts)
        ]
        if not any(pages) and report.pending:
            raise FileReadError(
                "PDF to skan (same obrazy), a OCR jest niedostępny. Zainstaluj Tesseract "
                "z językiem polskim, żeby odczytać tekst z obrazów."
            )
    if not any(pages):
        raise FileReadError("PDF nie zawiera tekstu – ani w warstwie tekstowej, "
                            "ani na obrazach (pusty skan?).")
    return pages


def read_docx(data: bytes, ocr: bool = True) -> str:
    """Tekst dokumentu Word w kolejności czytania: akapity i tabele tak, jak
    stoją w pliku, a tekst rozpoznany na obrazach (ocr=True) zaraz po akapicie
    albo wierszu tabeli, w którym obraz jest wstawiony – model widzi go
    w kontekście („Skan dowodu:” + treść dowodu).
    """
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = Document(io.BytesIO(data))
    except Exception as error:  # python-docx rzuca różne wyjątki dla złych plików
        raise FileReadError(f"Uszkodzony plik .docx: {error}") from error

    image_texts: dict[str, str] = {}
    pending = 0
    if ocr:
        from engine.ocr import docx_image_parts, ocr_docx

        report = ocr_docx(data)
        image_texts, pending = report.image_texts, report.pending

    def image_lines(element) -> list[str]:
        if not image_texts:
            return []
        texts = (image_texts.get(str(part.partname), "")
                 for part in docx_image_parts(element, document.part))
        return [normalize_text(text).strip() for text in texts if text.strip()]

    lines = []
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            lines.append(Paragraph(child, document).text)
            lines.extend(image_lines(child))
        elif child.tag == qn("w:tbl"):
            # Umowy często mają dane stron w tabelach – nie możemy ich pominąć.
            for row in Table(child, document).rows:
                lines.append(" | ".join(cell.text for cell in row.cells))
                lines.extend(image_lines(row._tr))
    text = "\n".join(lines).strip()
    if not text and pending:
        raise FileReadError(
            "Dokument Word zawiera same obrazy, a OCR jest niedostępny. Zainstaluj Tesseract "
            "z językiem polskim, żeby odczytać tekst z obrazów."
        )
    return text


def read_uploaded_pages(filename: str, data: bytes) -> list[str]:
    """Zwraca tekst pliku jako listę stron.

    PDF ma prawdziwe strony. Pozostałe formaty nie mają stron, więc cały
    plik to jedna „strona”.
    """
    extension = Path(filename).suffix.lower()
    if extension in TEXT_EXTENSIONS:
        pages = [decode_text(data)]
    elif extension == ".pdf":
        pages = read_pdf_pages(data)
    elif extension == ".docx":
        pages = [read_docx(data)]
    else:
        raise FileReadError(
            f"Nieobsługiwany format „{extension}”. "
            f"Dozwolone: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if not any(page.strip() for page in pages):
        raise FileReadError("Plik jest pusty.")
    return pages


def join_pages(pages: list[str]) -> str:
    """Skleja strony w jeden tekst, oddzielając je znakiem końca strony.

    PAGE_BREAK („\\f”) to standardowy znak nowej strony. Dzięki niemu LLM
    dostaje każdą stronę osobno (patrz llm_masker.split_into_chunks),
    nawet gdy cały dokument jest jednym tekstem w polu edycji.
    """
    return PAGE_BREAK.join(pages)


def read_uploaded_file(filename: str, data: bytes) -> str:
    """Zwraca tekst z pliku (strony sklejone znakiem PAGE_BREAK)."""
    return join_pages(read_uploaded_pages(filename, data))
