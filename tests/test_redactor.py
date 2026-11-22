"""Testy zaczerniania PDF.

Sprawdzamy dwie rzeczy:
1. dane ZNIKNĘŁY z pliku (nie da się ich skopiować ani wyszukać),
2. paski są tam, gdzie trzeba – mierzymy kolor pikseli na wyrenderowanej stronie.
"""

import pymupdf

from engine.redactor import (
    page_words,
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


# --- Tekst obrócony (ukośny znak wodny) ------------------------------------------

def _rotated_pdf(color=(0, 0, 0), fill_opacity=1.0) -> bytes:
    """Tabela z ukośnym napisem „SPECIMEN” na środku (jak wyciąg bankowy)."""
    import math

    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    for row in range(20):
        page.insert_text((60, 250 + row * 16), f"Wiersz {row:02d}   Kwota 1{row:02d},00", fontsize=10)
    writer = pymupdf.TextWriter(page.rect, opacity=fill_opacity, color=color)
    writer.append((120, 560), "SPECIMEN", fontsize=80)
    writer.write_text(page, morph=(pymupdf.Point(120, 560), pymupdf.Matrix(-38)))
    return document.tobytes()


def _black_share(pdf: bytes, clip=None) -> float:
    import numpy as np

    page = pymupdf.open(stream=pdf, filetype="pdf")[0]
    pixmap = page.get_pixmap(dpi=50, clip=clip)
    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(-1, pixmap.n)[:, :3]
    return float((pixels.max(axis=1) < 40).mean())


def test_rotated_word_gets_rotated_bar_not_page_wide_rectangle():
    # Wcześniej pasek = prostokąt otaczający ukośne słowo -> zakryta cała tabela.
    pdf = _rotated_pdf()
    page = pymupdf.open(stream=pdf, filetype="pdf")[0]
    word = next(w for w in page_words(page) if w.key == "specimen")
    assert word.quad is not None and word.quad.faint_level is None
    result = redact_pdf(pdf, ["SPECIMEN"])
    assert result.bars == 1 and result.missing == []
    box = word.rect
    assert box.get_area() > 60_000                         # prostokąt otaczający jest ogromny
    assert _black_share(result.pdf, box) < 0.6               # a pasek zajmuje tylko pas napisu
    assert "SPECIMEN" not in pymupdf.open(stream=result.pdf, filetype="pdf")[0].get_text()


def test_faint_rotated_watermark_is_erased_and_table_stays_readable():
    pdf = _rotated_pdf(color=(0.55, 0.55, 0.55), fill_opacity=0.16)
    page = pymupdf.open(stream=pdf, filetype="pdf")[0]
    word = next(w for w in page_words(page) if w.key == "specimen")
    assert word.quad.faint_level is not None
    result = redact_pdf(pdf, ["SPECIMEN"])
    assert _black_share(result.pdf) < 0.01                   # żadnego czarnego pasa
    import numpy as np

    def levels(document_bytes):
        pixmap = pymupdf.open(stream=document_bytes, filetype="pdf")[0].get_pixmap(dpi=100)
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(-1, pixmap.n)[:, :3].min(axis=1)

    before, after = levels(pdf), levels(result.pdf)
    # Ta sama strona bez napisu, spłaszczona tak samo – jej jasnoszare piksele
    # to tylko wygładzone krawędzie liter tabeli.
    baseline = levels(redact_pdf(_rotated_pdf(color=(1, 1, 1), fill_opacity=0.0), []).pdf)
    gray = lambda pixels: int(((pixels > 190) & (pixels < 250)).sum())  # noqa: E731
    assert gray(before) > 2 * gray(baseline)                 # napis był wyraźny…
    assert gray(after) <= 1.02 * gray(baseline)              # …a po wybieleniu nic z niego nie zostało
    # …a ciemny tekst tabeli (także ten, który leżał na napisie) został.
    assert (after < 120).sum() > 0.9 * (before < 120).sum()
