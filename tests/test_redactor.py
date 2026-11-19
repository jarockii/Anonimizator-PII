"""Testy zaczerniania PDF.

Sprawdzamy dwie rzeczy:
1. dane ZNIKNĘŁY z pliku (nie da się ich skopiować ani wyszukać),
2. paski są tam, gdzie trzeba – mierzymy kolor pikseli na wyrenderowanej stronie.
"""

import pymupdf

from engine.redactor import (
    page_count,
    redact_pdf,
    redact_text_to_pdf,
    render_preview,
    text_to_pdf,
)

TEXT = (
    "Pacjent: Łukasz Żółkiewski, PESEL 44051401359\n"
    "ul. Śląska 5, 31-147 Kraków\n"
    "\n"
    "Koszt leczenia: 4 850,00 zł netto"
)
SECRETS = ["Łukasz Żółkiewski", "44051401359", "ul. Śląska 5, 31-147 Kraków", "4 850,00 zł"]


def pdf_text(pdf_bytes: bytes) -> str:
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    return "".join(page.get_text() for page in document)


def darkness(pdf_bytes: bytes, rect: pymupdf.Rect) -> float:
    """Jaka część pikseli w prostokącie jest czarna (0.0 – 1.0)."""
    page = pymupdf.open(stream=pdf_bytes, filetype="pdf")[0]
    pixmap = page.get_pixmap(dpi=100, clip=rect, colorspace=pymupdf.csGRAY)
    samples = pixmap.samples
    return sum(1 for value in samples if value < 60) / len(samples)


def word_rect(pdf_bytes: bytes, word: str) -> pymupdf.Rect:
    page = pymupdf.open(stream=pdf_bytes, filetype="pdf")[0]
    return page.search_for(word)[0]


# --- PDF z tekstu ---------------------------------------------------------------


def test_text_to_pdf_keeps_polish_characters():
    assert "Łukasz Żółkiewski" in pdf_text(text_to_pdf(TEXT))


def test_long_text_spans_multiple_pages():
    long_text = "\n".join(f"Linia numer {i}" for i in range(200))
    assert page_count(text_to_pdf(long_text)) > 1


def test_text_redaction_never_writes_secrets_to_file():
    result = redact_text_to_pdf(TEXT, SECRETS)
    text = pdf_text(result.pdf)
    for secret in ["Żółkiewski", "44051401359", "Śląska", "4 850,00"]:
        assert secret not in text
    # Reszta tekstu zostaje i da się ją zaznaczyć.
    assert "Pacjent:" in text and "PESEL" in text and "netto" in text
    assert result.bars == 4
    assert result.missing == []


def test_text_redaction_draws_black_bars_but_not_on_neighbours():
    pdf = redact_text_to_pdf(TEXT, SECRETS).pdf
    # Pasek między „Pacjent:” a „, PESEL” – w tym miejscu było nazwisko.
    pacjent, pesel = word_rect(pdf, "Pacjent:"), word_rect(pdf, ", PESEL")
    bar_area = pymupdf.Rect(pacjent.x1 + 3, pacjent.y0 + 4, pesel.x0 - 3, pacjent.y1 - 4)
    assert darkness(pdf, bar_area) > 0.9
    # Słowo tuż za paskiem nie może być zamalowane.
    assert darkness(pdf, word_rect(pdf, "netto")) < 0.3


def test_text_redaction_value_across_lines():
    result = redact_text_to_pdf("Firma: Marek\nLewandowski sp.j.", ["Marek\nLewandowski"])
    assert "Lewandowski" not in pdf_text(result.pdf)
    assert result.bars == 2  # po jednym pasku na każdą linię


def test_text_redaction_escapes_html():
    result = redact_text_to_pdf("<b>Jan</b> & Anna", ["Anna"])
    assert "<b>Jan</b> &" in pdf_text(result.pdf)


def test_text_redaction_reports_missing():
    assert redact_text_to_pdf(TEXT, ["Anna Nowak"]).missing == ["Anna Nowak"]


# --- Wgrany PDF (spłaszczanie) -----------------------------------------------------


def make_pdf(*lines: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    for number, line in enumerate(lines):
        page.insert_text((72, 72 + 18 * number), line, fontsize=11)
    return document.tobytes()


def test_flattened_pdf_has_no_text_layer():
    original = make_pdf("Pacjent: Jan Kowalski, PESEL 44051401359")
    result = redact_pdf(original, ["Jan Kowalski", "44051401359"])
    assert pdf_text(result.pdf).strip() == ""  # nic do skopiowania
    assert result.bars == 2


def test_flattened_pdf_bars_cover_values_and_spare_neighbours():
    original = make_pdf("Kwota 18 500 PLN netto do zaplaty")
    result = redact_pdf(original, ["18 500 PLN"])
    assert darkness(result.pdf, word_rect(original, "18 500 PLN")) > 0.6
    # Przy wbudowanym apply_redactions() „netto” wchodziło pod pasek.
    assert darkness(result.pdf, word_rect(original, "netto")) < 0.3


def test_flattened_pdf_keeps_page_size():
    original = make_pdf("Jan Kowalski")
    source_rect = pymupdf.open(stream=original, filetype="pdf")[0].rect
    result_rect = pymupdf.open(stream=redact_pdf(original, ["Jan"]).pdf, filetype="pdf")[0].rect
    assert source_rect == result_rect


def test_pdf_value_broken_across_lines():
    original = make_pdf("Adres: ul. Marszalkowska 84/92,", "00-514 Warszawa. Dalszy tekst.")
    result = redact_pdf(original, ["ul. Marszalkowska 84/92, 00-514 Warszawa"])
    assert darkness(result.pdf, word_rect(original, "Marszalkowska")) > 0.6
    assert darkness(result.pdf, word_rect(original, "Warszawa")) > 0.6
    assert darkness(result.pdf, word_rect(original, "Dalszy")) < 0.3
    assert result.missing == []


def test_pdf_reports_missing_and_handles_empty_values():
    result = redact_pdf(make_pdf("Jan Kowalski"), ["Anna Nowak", "", "  "])
    assert result.missing == ["Anna Nowak"]
    assert result.bars == 0


def test_preview_is_png():
    assert render_preview(text_to_pdf(TEXT)).startswith(b"\x89PNG")


def test_flattened_pdf_finds_polish_uppercase_and_soft_hyphens():
    # search_for() w MuPDF ignoruje wielkość liter tylko dla ASCII,
    # a myślnik w pliku bywa SOFT HYPHEN-em.
    document = pymupdf.open()
    page = document.new_page()
    page.insert_font(fontname="dejavu", fontbuffer=pymupdf.Font("cjk").buffer)
    page.insert_text((72, 72), "PODPISAŁ: TOMASZ WIŚNIEWSKI", fontname="dejavu", fontsize=11)
    page.insert_text((72, 100), "NIP 526­025­02­74", fontname="dejavu", fontsize=11)
    original = document.tobytes()
    result = redact_pdf(original, ["Tomasz Wiśniewski", "526-025-02-74"])
    assert result.missing == []
    assert result.bars == 2
