"""Testy OCR obrazów w PDF.

Tesseract jest podmieniany (mock) – testy sprawdzają, KTÓRE obrazy trafiają
do OCR, jak wynik wraca do tekstu i czy dostaje czarne paski. Ostatni test
uruchamia prawdziwego Tesseracta, jeśli jest zainstalowany.
"""

import pymupdf
import pytest

from engine import ocr
from engine.file_reader import FileReadError, read_pdf_pages
from engine.redactor import redact_pdf

# Słowa, które „rozpozna” fałszywy Tesseract – współrzędne względem obrazu (0..1).
FAKE_WORDS = [
    (0.05, 0.10, 0.30, 0.40, "Jan", 0, 0),
    (0.35, 0.10, 0.90, 0.40, "Kowalski", 0, 0),
    (0.05, 0.55, 0.15, 0.85, "PESEL", 0, 1),
    (0.20, 0.55, 0.60, 0.85, "92031502349", 0, 1),
]


@pytest.fixture(autouse=True)
def fake_tesseract(monkeypatch):
    """Domyślnie: OCR „dostępny”, zwraca FAKE_WORDS i liczy wywołania."""
    calls = []

    def run(pixmap):
        calls.append((pixmap.width, pixmap.height))
        return FAKE_WORDS

    monkeypatch.setattr(ocr, "run_tesseract", run)
    monkeypatch.setattr(ocr, "ocr_available", lambda: True)
    ocr.clear_cache()
    yield calls
    ocr.clear_cache()


def white_image(width=400, height=160) -> pymupdf.Pixmap:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pixmap.set_rect(pixmap.irect, (255, 255, 255))
    return pixmap


def make_pdf(pages: list[dict]) -> bytes:
    """pages: [{"text": str, "image": Rect | None, "reuse": bool}]"""
    document = pymupdf.open()
    xref = 0
    for spec in pages:
        page = document.new_page(width=595, height=842)
        if spec.get("text"):
            page.insert_text((50, 60), spec["text"], fontsize=11)
        rect = spec.get("image")
        if rect is not None:
            if spec.get("reuse") and xref:
                page.insert_image(rect, xref=xref)
            else:
                xref = page.insert_image(rect, pixmap=white_image())
    return document.tobytes()


STAMP = pymupdf.Rect(100, 300, 400, 420)


def test_pages_without_images_are_never_ocrd(fake_tesseract):
    report = ocr.ocr_pdf(make_pdf([{"text": "Zwykła umowa"}, {"text": "Strona 2"}]))
    assert fake_tesseract == []
    assert report.images_total == 0
    assert report.texts == ["", ""]


def test_only_image_region_is_rendered(fake_tesseract):
    ocr.ocr_pdf(make_pdf([{"text": "Umowa", "image": STAMP}]))
    width, height = fake_tesseract[0]
    # Wycinek 300 x 120 pt w 300 dpi, a nie cała strona A4 (2480 x 3508 px).
    assert width == pytest.approx(300 * 300 / 72, abs=2)
    assert height == pytest.approx(120 * 300 / 72, abs=2)


def test_small_images_are_skipped(fake_tesseract):
    report = ocr.ocr_pdf(make_pdf([{"image": pymupdf.Rect(50, 50, 70, 60)}]))
    assert fake_tesseract == []
    assert report.skipped_small == 1


def test_image_with_text_layer_is_skipped(fake_tesseract):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(STAMP, pixmap=white_image())
    # Skan z warstwą OCR: tekst pokrywa obraz.
    for row in range(8):
        page.insert_text((105, 315 + row * 14), "tekst warstwy OCR " * 3, fontsize=11)
    report = ocr.ocr_pdf(document.tobytes())
    assert fake_tesseract == []
    assert report.skipped_text == 1


def test_full_page_scan_with_short_ocr_layer_is_skipped(fake_tesseract):
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.insert_image(page.rect, pixmap=white_image(), keep_proportion=False)
    page.insert_text((72, 100), "Marian Kowalski", fontsize=12)  # warstwa OCR skanera
    report = ocr.ocr_pdf(document.tobytes())
    assert fake_tesseract == []
    assert report.skipped_text == 1


def test_same_image_on_many_pages_is_ocrd_once(fake_tesseract):
    pdf = make_pdf([{"image": STAMP}, {"image": STAMP, "reuse": True},
                    {"image": STAMP, "reuse": True}])
    report = ocr.ocr_pdf(pdf)
    assert len(fake_tesseract) == 1
    assert report.images_ocr == 1 and report.images_reused == 2
    assert all("Kowalski" in text for text in report.texts)


def test_whole_file_result_is_cached(fake_tesseract):
    pdf = make_pdf([{"image": STAMP}])
    ocr.ocr_pdf(pdf)
    ocr.ocr_pdf(pdf)
    assert len(fake_tesseract) == 1


def test_words_are_placed_in_page_coordinates():
    report = ocr.ocr_pdf(make_pdf([{"image": STAMP}]))
    kowalski = next(word for word in report.words[0] if word.text == "Kowalski")
    assert kowalski.rect.x0 == pytest.approx(100 + 0.35 * 300)
    assert kowalski.rect.y1 == pytest.approx(300 + 0.40 * 120)
    assert report.texts[0] == "Jan Kowalski\nPESEL 92031502349"


def test_ocr_text_is_appended_to_page_text():
    pages = read_pdf_pages(make_pdf([{"text": "Umowa nr 5", "image": STAMP}]))
    assert pages[0].startswith("Umowa nr 5")
    assert "Jan Kowalski" in pages[0]


def test_scan_without_text_layer_is_now_readable():
    pages = read_pdf_pages(make_pdf([{"image": pymupdf.Rect(0, 0, 595, 842)}]))
    assert "PESEL 92031502349" in pages[0]


def test_scan_without_tesseract_gives_clear_error(monkeypatch, fake_tesseract):
    monkeypatch.setattr(ocr, "ocr_available", lambda: False)
    pdf = make_pdf([{"image": pymupdf.Rect(0, 0, 595, 842)}])
    with pytest.raises(FileReadError, match="Tesseract"):
        read_pdf_pages(pdf)
    assert fake_tesseract == []


def test_missing_tesseract_is_reported_and_not_cached(monkeypatch, fake_tesseract):
    monkeypatch.setattr(ocr, "ocr_available", lambda: False)
    pdf = make_pdf([{"text": "Umowa", "image": STAMP}])
    assert ocr.ocr_pdf(pdf).pending == 1
    monkeypatch.setattr(ocr, "ocr_available", lambda: True)
    assert ocr.ocr_pdf(pdf).images_ocr == 1  # po „instalacji” plik jest rozpoznany


def test_ocr_error_on_one_image_does_not_break_file(monkeypatch):
    def broken(pixmap):
        raise RuntimeError("uszkodzony obraz")

    monkeypatch.setattr(ocr, "run_tesseract", broken)
    report = ocr.ocr_pdf(make_pdf([{"text": "Umowa", "image": STAMP}]))
    assert report.errors and "uszkodzony" in report.errors[0]


def darkness(pdf_bytes: bytes, rect: pymupdf.Rect) -> float:
    page = pymupdf.open(stream=pdf_bytes, filetype="pdf")[0]
    pixmap = page.get_pixmap(dpi=100, clip=rect, colorspace=pymupdf.csGRAY)
    return sum(1 for value in pixmap.samples if value < 60) / len(pixmap.samples)


def test_ocr_words_get_black_bars():
    pdf = make_pdf([{"text": "Umowa", "image": STAMP}])
    result = redact_pdf(pdf, ["Jan Kowalski", "92031502349"], redact_images=False)
    assert result.missing == []
    name = pymupdf.Rect(100 + 0.05 * 300, 300 + 0.10 * 120, 100 + 0.90 * 300, 300 + 0.40 * 120)
    assert darkness(result.pdf, name) > 0.9
    # Słowo „PESEL” nie jest wartością – pod nim zostaje biel.
    label = pymupdf.Rect(100 + 0.06 * 300, 300 + 0.56 * 120, 100 + 0.14 * 300, 300 + 0.84 * 120)
    assert darkness(result.pdf, label) < 0.5


# --- Numery rozbite przez OCR (spacja w środku) -----------------------------------------


@pytest.mark.parametrize("text, category", [
    ("PESEL 561 10446533", "PESEL"),          # prawdziwy błąd OCR z przyklady/2_*.pdf
    ("PESEL 9203 1502349", "PESEL"),
    ("NIP 526 025 02 74", "NIP"),
    ("NIP 5260 250274", "NIP"),
    ("rachunek PL61 1090 10140000 0712 1981 2874", "IBAN"),
])
def test_numbers_split_by_ocr_are_masked(text, category):
    from engine.regex_masker import mask_with_regex
    from engine.vault import Vault
    masked, found = mask_with_regex(text, Vault())
    assert f"[{category}_1]" in masked
    assert not any(ch.isdigit() for ch in masked.split("[")[0] + masked.split("]")[-1])


@pytest.mark.parametrize("text", [
    "tel. +48 601 234 567",           # 11 cyfr z 3 separatorami – nie PESEL
    "tel. 600 100 200, 12 lat",       # osobne liczby
    "faktury 1 2 3 4 5 6 7 8 9 0 1",  # za dużo separatorów
    "kod 4405 1401 358",              # zła suma kontrolna
])
def test_separate_numbers_are_not_glued_into_pesel_or_nip(text):
    from engine.regex_masker import mask_with_regex
    from engine.vault import Vault
    masked, _ = mask_with_regex(text, Vault())
    assert "[PESEL_" not in masked and "[NIP_" not in masked


def test_split_pesel_from_ocr_gets_bars(monkeypatch):
    words = [(0.05, 0.2, 0.25, 0.6, "PESEL", 0, 0), (0.30, 0.2, 0.45, 0.6, "561", 0, 0),
             (0.47, 0.2, 0.90, 0.6, "10446533", 0, 0)]
    monkeypatch.setattr(ocr, "run_tesseract", lambda pixmap: words)
    from engine.file_reader import join_pages
    from engine.pipeline import anonymize
    from engine.vault import Vault
    pdf = make_pdf([{"image": pymupdf.Rect(0, 0, 595, 842)}])
    result = anonymize(join_pages(read_pdf_pages(pdf)), Vault(), use_llm=False)
    assert result.masked_text == "PESEL [PESEL_1]"
    redacted = redact_pdf(pdf, [e["oryginał"] for e in result.entities], redact_images=False)
    assert redacted.missing == [] and redacted.bars == 1


# --- Prawdziwy Tesseract --------------------------------------------------------------


def test_real_tesseract_reads_polish_text(monkeypatch):
    monkeypatch.undo()  # bez mocka
    ocr.clear_cache()
    if not ocr.ocr_available():
        pytest.skip("Tesseract z modelem „pol” nie jest zainstalowany")
    # Tekst zamieniony w obraz, tak jak w zeskanowanym dokumencie.
    source = pymupdf.open()
    page = source.new_page(width=400, height=120)
    page.insert_text((20, 60), "Pacjent: Łukasz Żółkiewski", fontsize=20, fontname="helv")
    image = page.get_pixmap(dpi=200)
    scan = pymupdf.open()
    scan.new_page(width=400, height=120).insert_image(pymupdf.Rect(0, 0, 400, 120), pixmap=image)
    report = ocr.ocr_pdf(scan.tobytes())
    assert report.images_ocr == 1
    assert "Pacjent" in report.texts[0]
