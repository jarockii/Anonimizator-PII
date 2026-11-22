"""Testy OCR obrazów w plikach Word (.docx).

Tesseract jest podmieniany (mock) – testy sprawdzają, które obrazy trafiają
do OCR i gdzie rozpoznany tekst ląduje w dokumencie. Ostatni test uruchamia
prawdziwego Tesseracta, jeśli jest zainstalowany.
"""

import io

import pymupdf
import pytest
from docx import Document
from docx.shared import Cm

from engine import ocr
from engine.file_reader import FileReadError, read_docx, read_uploaded_file
from engine.pipeline import anonymize
from engine.vault import Vault

FAKE_WORDS = [
    (0.05, 0.10, 0.30, 0.40, "Jan", 0, 0),
    (0.35, 0.10, 0.90, 0.40, "Kowalski", 0, 0),
    (0.05, 0.55, 0.25, 0.85, "PESEL", 0, 1),
    (0.30, 0.55, 0.90, 0.85, "92031502349", 0, 1),
]


@pytest.fixture(autouse=True)
def fake_tesseract(monkeypatch):
    """OCR „dostępny”, zwraca FAKE_WORDS i zapisuje rozmiar każdej pixmapy."""
    calls = []

    def run(pixmap):
        calls.append((pixmap.width, pixmap.height, pixmap.colorspace.n, pixmap.alpha))
        return FAKE_WORDS

    monkeypatch.setattr(ocr, "run_tesseract", run)
    monkeypatch.setattr(ocr, "ocr_available", lambda: True)
    ocr.clear_cache()
    yield calls
    ocr.clear_cache()


def image_png(width=400, height=160, gray=False, alpha=False, seed=0) -> bytes:
    colorspace = pymupdf.csGRAY if gray else pymupdf.csRGB
    pixmap = pymupdf.Pixmap(colorspace, pymupdf.IRect(0, 0, width, height), alpha)
    pixmap.clear_with(200 + seed)   # różny kolor = różny obraz w pakiecie .docx
    return pixmap.tobytes("png")


def save(document) -> bytes:
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_image_text_lands_right_after_its_paragraph(fake_tesseract):
    document = Document()
    document.add_paragraph("Skan dowodu w załączniku:")
    document.add_picture(io.BytesIO(image_png()), width=Cm(8))
    document.add_paragraph("Koniec pisma.")
    text = read_docx(save(document))
    assert text.splitlines() == [
        "Skan dowodu w załączniku:", "", "Jan Kowalski", "PESEL 92031502349", "Koniec pisma."]
    assert len(fake_tesseract) == 1


def test_tables_keep_their_position_and_images_in_cells_are_read():
    # Wcześniej tabele były doklejane na końcu – teraz stoją tam, gdzie w pliku.
    document = Document()
    document.add_paragraph("Strony umowy:")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Pieczątka"
    table.rows[0].cells[1].paragraphs[0].add_run().add_picture(io.BytesIO(image_png()), width=Cm(4))
    document.add_paragraph("§1 Przedmiot umowy")
    lines = read_docx(save(document)).splitlines()
    # Tekst z obrazu trafia do TEJ komórki, w której obraz stoi.
    assert lines == ["Strony umowy:", "Pieczątka | Jan Kowalski PESEL 92031502349",
                     "§1 Przedmiot umowy"]


def test_same_image_twice_is_ocr_once(fake_tesseract):
    document = Document()
    logo = image_png()
    document.add_picture(io.BytesIO(logo))
    document.add_paragraph("Treść")
    document.add_picture(io.BytesIO(logo))
    data = save(document)
    report = ocr.ocr_docx(data)
    assert (report.images_total, report.images_ocr, report.images_reused) == (2, 1, 1)
    assert len(fake_tesseract) == 1
    assert read_docx(data).count("92031502349") == 2   # tekst przy każdym wystąpieniu


def test_small_images_are_skipped(fake_tesseract):
    document = Document()
    document.add_paragraph("Punktor")
    document.add_picture(io.BytesIO(image_png(20, 10)))
    report = ocr.ocr_docx(save(document))
    assert report.skipped_small == 1 and report.images_ocr == 0 and fake_tesseract == []


@pytest.mark.parametrize("gray, alpha", [(True, False), (False, True), (True, True)])
def test_image_is_converted_to_rgb_without_alpha(fake_tesseract, gray, alpha):
    # W skali szarości pdfocr_tobytes() po cichu zwraca pusty tekst.
    document = Document()
    document.add_picture(io.BytesIO(image_png(gray=gray, alpha=alpha)))
    ocr.ocr_docx(save(document))
    (_, _, channels, has_alpha), = fake_tesseract
    assert channels == 3 and not has_alpha


def test_small_screenshot_is_upscaled_and_big_photo_is_capped(fake_tesseract):
    document = Document()
    document.add_picture(io.BytesIO(image_png(400, 100, seed=1)), width=Cm(5))
    document.add_picture(io.BytesIO(image_png(5000, 4000, seed=2)), width=Cm(5))
    ocr.ocr_docx(save(document))
    (small_w, *_), (big_w, big_h, *_) = fake_tesseract
    assert small_w == 400 * ocr.DOCX_OCR_MAX_UPSCALE      # 400 px -> ×3 (limit)
    assert big_w * big_h <= ocr.MAX_OCR_PIXELS * 1.01


def test_without_tesseract_text_is_read_and_image_is_pending(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: False)
    document = Document()
    document.add_paragraph("Jan Kowalski")
    document.add_picture(io.BytesIO(image_png()))
    data = save(document)
    assert read_docx(data) == "Jan Kowalski"
    assert ocr.ocr_docx(data).pending == 1


def test_only_images_without_tesseract_gives_clear_error(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: False)
    document = Document()
    document.add_picture(io.BytesIO(image_png()))
    with pytest.raises(FileReadError, match="OCR jest niedostępny"):
        read_uploaded_file("skan.docx", save(document))


def test_ocr_error_on_one_image_is_reported(monkeypatch):
    def broken(pixmap):
        raise RuntimeError("uszkodzony obraz")

    monkeypatch.setattr(ocr, "run_tesseract", broken)
    document = Document()
    document.add_paragraph("Tekst zostaje")
    document.add_picture(io.BytesIO(image_png()))
    data = save(document)
    assert read_docx(data) == "Tekst zostaje"
    assert ocr.ocr_docx(data).errors == ["obraz image1.png: uszkodzony obraz"]


def test_ocr_can_be_disabled(fake_tesseract):
    document = Document()
    document.add_picture(io.BytesIO(image_png()))
    document.add_paragraph("Tylko tekst")
    assert read_docx(save(document), ocr=False) == "Tylko tekst"
    assert fake_tesseract == []


def test_pesel_from_image_in_docx_is_masked():
    document = Document()
    document.add_paragraph("Załącznik:")
    document.add_picture(io.BytesIO(image_png()))
    result = anonymize(read_uploaded_file("wniosek.docx", save(document)), Vault(), use_llm=False)
    assert "92031502349" not in result.masked_text
    assert "[PESEL_1]" in result.masked_text


# --- Prawdziwy Tesseract --------------------------------------------------------------


def test_real_tesseract_reads_low_resolution_image_in_docx(monkeypatch):
    monkeypatch.undo()  # bez mocka
    ocr.clear_cache()
    if not ocr.ocr_available():
        pytest.skip("Tesseract z modelem „pol” nie jest zainstalowany")
    # Tekst 11 pt jako obraz 72 dpi (jak mały zrzut ekranu). Bez powiększenia
    # Tesseract czytał „92031802349” – PESEL ze złą sumą kontrolną.
    source = pymupdf.open()
    page = source.new_page(width=360, height=60)
    page.insert_text((10, 20), "Jan Kowalski", fontsize=11, fontname="helv")
    page.insert_text((10, 36), "PESEL 92031502349", fontsize=11, fontname="helv")
    document = Document()
    document.add_picture(io.BytesIO(page.get_pixmap(dpi=72).tobytes("png")))
    text = read_docx(save(document))
    assert "92031502349" in text and "Kowalski" in text
    ocr.clear_cache()
