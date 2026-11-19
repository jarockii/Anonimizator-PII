"""Testy wczytywania plików. Pliki tworzymy w pamięci, bez zapisu na dysk."""

import io

import pymupdf
import pytest
from docx import Document

from engine.file_reader import PAGE_BREAK, FileReadError, read_uploaded_file, read_uploaded_pages


def make_text_pdf(text: str) -> bytes:
    """Tworzy w pamięci jednostronicowy PDF z jedną linią tekstu."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    return document.tobytes()


def test_reads_utf8_text():
    assert read_uploaded_file("notatka.txt", "Zażółć gęślą jaźń".encode()) == "Zażółć gęślą jaźń"


def test_reads_old_windows_polish_encoding():
    data = "Pacjent: Łukasz Żółkiewski".encode("cp1250")
    assert read_uploaded_file("stary.log", data) == "Pacjent: Łukasz Żółkiewski"


def test_strips_utf8_bom():
    assert read_uploaded_file("excel.csv", "﻿imie;nazwisko".encode()) == "imie;nazwisko"


def test_extension_is_case_insensitive():
    assert read_uploaded_file("LOG.TXT", b"abc") == "abc"


def test_reads_docx_paragraphs_and_tables():
    document = Document()
    document.add_paragraph("Umowa nr 17/2025")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Zleceniodawca"
    table.rows[0].cells[1].text = "Anna Nowak"
    buffer = io.BytesIO()
    document.save(buffer)

    text = read_uploaded_file("umowa.docx", buffer.getvalue())
    assert "Umowa nr 17/2025" in text
    assert "Zleceniodawca | Anna Nowak" in text


def test_reads_pdf_text():
    text = read_uploaded_file("wynik.pdf", make_text_pdf("Jan Kowalski PESEL 44051401359"))
    assert "Jan Kowalski PESEL 44051401359" in text


def test_pdf_without_text_reports_scan():
    document = pymupdf.open()
    document.new_page()  # pusta strona = jak skan bez warstwy tekstowej
    with pytest.raises(FileReadError, match="skan"):
        read_uploaded_file("skan.pdf", document.tobytes())


def test_broken_files_raise_friendly_error():
    with pytest.raises(FileReadError):
        read_uploaded_file("zly.pdf", b"to nie jest PDF")
    with pytest.raises(FileReadError):
        read_uploaded_file("zly.docx", b"to nie jest docx")


def test_unsupported_extension():
    with pytest.raises(FileReadError, match="Nieobsługiwany"):
        read_uploaded_file("zdjecie.png", b"\x89PNG")


def test_empty_file():
    with pytest.raises(FileReadError, match="pusty"):
        read_uploaded_file("pusty.txt", b"   \n")


def test_pdf_soft_hyphens_are_normalized():
    # Prawdziwy przypadek: myślnik zapisany w PDF jako SOFT HYPHEN (U+00AD).
    # Bez normalizacji regex nie znajdzie NIP-u ani UUID z Karty Podpisów.
    pdf = make_text_pdf("NIP 526­025­02­74")
    assert "NIP 526-025-02-74" in read_uploaded_file("umowa.pdf", pdf)


def test_pdf_pages_are_returned_separately():
    document = pymupdf.open()
    for text in ["Strona pierwsza", "KARTA PODPISÓW"]:
        document.new_page().insert_text((72, 72), text)
    pages = read_uploaded_pages("umowa.pdf", document.tobytes())
    assert pages == ["Strona pierwsza", "KARTA PODPISÓW"]
    assert read_uploaded_file("umowa.pdf", document.tobytes()).count(PAGE_BREAK) == 1
