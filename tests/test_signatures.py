"""Testy wykrywania podpisów: wektorowych (A) i odręcznych na skanach (B).

Skan budujemy jak biurowy skaner: obraz całej strony + niewidoczna warstwa
tekstowa, więc testy nie potrzebują Tesseracta.
"""

import math

import numpy as np
import pymupdf
import pytest

from engine import ocr, signatures
from engine.redactor import redact_pdf


@pytest.fixture(autouse=True)
def no_ocr(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: False)
    ocr.clear_cache()
    yield
    ocr.clear_cache()


def draw_signature(page, x=100, y=700, width=150):
    """Pismo odręczne: pętle liter (pióro wraca w lewo) + kreska w poprzek."""
    points = []
    steps = 120
    for i in range(steps):
        t = i / steps * 12 * math.pi
        points.append(pymupdf.Point(x + width * i / steps + 7 * math.cos(t),
                                    y + 9 * math.sin(t) + 3 * math.sin(t / 5)))
    shape = page.new_shape()
    shape.draw_polyline(points)
    shape.draw_bezier((x + 10, y + 15), (x + 50, y - 25), (x + 90, y + 30), (x + width, y - 5))
    shape.finish(color=(0, 0, 0.5), fill=None, width=1.2, closePath=False)
    shape.commit()


def draw_table(page):
    shape = page.new_shape()
    for row in range(6):
        shape.draw_line((50, 100 + row * 20), (400, 100 + row * 20))
    for col in range(5):
        shape.draw_line((50 + col * 87.5, 100), (50 + col * 87.5, 200))
    shape.finish(color=(0, 0, 0), width=0.5)
    shape.commit()


def draw_line_chart(page):
    points = [pymupdf.Point(60 + i * 8, 400 - 40 * abs(math.sin(i * 0.9))) for i in range(40)]
    shape = page.new_shape()
    shape.draw_polyline(points)
    shape.finish(color=(0.8, 0, 0), fill=None, width=1)
    shape.commit()


def draw_pie_chart(page):
    shape = page.new_shape()
    for start, sweep, color in ((0, 120, (1, 0, 0)), (120, 100, (0, 1, 0)), (220, 140, (0, 0, 1))):
        shape.draw_sector((300, 500), (360, 500), sweep, fullSector=True)
        shape.finish(color=(0, 0, 0), fill=color, closePath=True)
    shape.commit()


def page_with(*painters) -> pymupdf.Page:
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    for paint in painters:
        paint(page)
    return page


# --- A: podpisy wektorowe -----------------------------------------------------

def test_vector_signature_is_found():
    rects = signatures.find_vector_signatures(page_with(draw_signature))
    assert len(rects) == 1
    rect = rects[0]
    assert rect.contains(pymupdf.Rect(105, 695, 245, 705))
    assert rect.width < 200 and rect.height < 80


@pytest.mark.parametrize("painter", [draw_table, draw_line_chart, draw_pie_chart],
                         ids=["tabela", "wykres-liniowy", "wykres-kolowy"])
def test_tables_and_charts_are_not_signatures(painter):
    assert signatures.find_vector_signatures(page_with(painter)) == []


def test_underline_and_signature_line_are_not_signatures():
    def underline(page):
        page.draw_line((100, 720), (300, 720))            # linia „podpis: ______”
        page.draw_line((100, 740), (300, 740), dashes="[2] 0")
    assert signatures.find_vector_signatures(page_with(underline)) == []


def test_signature_next_to_table_is_found_alone():
    rects = signatures.find_vector_signatures(page_with(draw_table, draw_signature))
    assert len(rects) == 1 and rects[0].y0 > 600


def test_dense_hatching_of_single_lines_is_not_a_signature():
    import random
    rng = random.Random(1)

    def hatching(page):
        for _ in range(1500):
            x, y = rng.uniform(20, 570), rng.uniform(20, 820)
            page.draw_line((x, y), (x + rng.uniform(-6, 6), y + rng.uniform(-6, 6)))
    assert signatures.find_vector_signatures(page_with(hatching)) == []


def test_signature_made_of_separate_strokes_is_one_rect():
    def two_words(page):
        draw_signature(page, x=100, width=70)
        draw_signature(page, x=178, width=70)   # drugie słowo 8 pt dalej
    rects = signatures.find_vector_signatures(page_with(two_words))
    assert len(rects) == 1 and rects[0].width > 140


def signed_pdf(scan: bool = False) -> bytes:
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    if scan:
        background = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 595, 842), False)
        background.clear_with(255)
        page.insert_image(page.rect, pixmap=background)
    page.insert_text((100, 650), "Podpis Wynajmującego:", fontsize=11)
    if not scan:
        draw_signature(page)
    return document.tobytes()


def ink_share(pdf: bytes, clip: pymupdf.Rect) -> float:
    """Część pikseli w obszarze, które są (prawie) czarne."""
    page = pymupdf.open(stream=pdf, filetype="pdf")[0]
    pixmap = page.get_pixmap(dpi=72, clip=clip)
    pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(-1, pixmap.n)[:, :3]
    return float((pixels.max(axis=1) < 40).mean())


def test_vector_signature_gets_black_bar_in_flattened_pdf():
    result = redact_pdf(signed_pdf(), [])
    assert result.vector_signatures == 1
    assert ink_share(result.pdf, pymupdf.Rect(110, 692, 240, 708)) > 0.95
    assert ink_share(result.pdf, pymupdf.Rect(100, 600, 300, 640)) < 0.05   # reszta strony


def test_vector_signature_stays_when_image_redaction_is_off():
    result = redact_pdf(signed_pdf(), [], redact_images=False)
    assert result.vector_signatures == 0
    assert ink_share(result.pdf, pymupdf.Rect(110, 692, 240, 708)) < 0.5


def test_sample_pdfs_have_no_false_vector_signatures():
    from pathlib import Path
    for path in sorted((Path(__file__).parent.parent / "przyklady").glob("*.pdf")):
        for page in pymupdf.open(path):
            assert signatures.find_vector_signatures(page) == [], (path.name, page.number)


# --- B: podpisy na skanach (atrament przy słowie „podpis”) --------------------

def scanned_pdf(label: str = "Podpis Wynajmującego:", sign: bool = True,
                text_layer: bool = True, dotted_line: bool = True) -> bytes:
    """Skan jak z biurowego skanera: cała strona to obraz (200 dpi, szum),
    a tekst jest niewidoczną warstwą OCR nad nim."""
    source = pymupdf.open()
    page = source.new_page(width=595, height=842)
    page.insert_text((100, 300), "Ja, niżej podpisany Jan Kowalski, oświadczam", fontsize=11)
    page.insert_text((100, 660), label, fontsize=11)
    if dotted_line:
        page.insert_text((100, 720), "." * 60, fontsize=11)
    if sign:
        draw_signature(page, x=110, y=695)
    image = page.get_pixmap(dpi=200)
    scan = pymupdf.open()
    scanned = scan.new_page(width=595, height=842)
    scanned.insert_image(scanned.rect, pixmap=image)
    if text_layer:
        for word in page.get_text("words"):
            scanned.insert_text((word[0], word[3] - 2), word[4], fontsize=11, render_mode=3)
    return scan.tobytes()


def test_signature_on_scan_next_to_label_is_blacked_out():
    result = redact_pdf(scanned_pdf(), [])
    assert result.scan_signatures == 1 and result.vector_signatures == 0
    assert ink_share(result.pdf, pymupdf.Rect(120, 690, 240, 702)) > 0.95
    # Tekst skanu zostaje czytelny: „niżej podpisany…” i sama etykieta.
    assert ink_share(result.pdf, pymupdf.Rect(100, 288, 300, 302)) < 0.4
    assert ink_share(result.pdf, pymupdf.Rect(100, 650, 200, 662)) < 0.4


def test_scan_without_signature_gets_no_bar():
    # Sama etykieta i linia z kropek – nic do zaczernienia.
    result = redact_pdf(scanned_pdf(sign=False), [])
    assert result.scan_signatures == 0


def test_signature_word_in_prose_is_not_an_anchor():
    # „Ja, niżej podpisany” to treść pisma, nie miejsce na podpis.
    result = redact_pdf(scanned_pdf(label="Data: 12.03.2025"), [])
    assert result.scan_signatures == 0


@pytest.mark.parametrize("word, expected", [
    ("podpis", True), ("Podpis:", True), ("(podpis)", True), ("podpisy", True),
    ("signature", True), ("podpisany", False), ("podpisania", False), ("Podpisujący", False),
])
def test_anchor_words(word, expected):
    assert signatures._is_anchor(word) is expected


def test_scan_without_any_text_is_left_alone():
    # Bez warstwy tekstowej i bez OCR nie ma słowa „podpis” – nie zgadujemy.
    result = redact_pdf(scanned_pdf(text_layer=False), [])
    assert result.scan_signatures == 0


def test_heuristic_runs_only_on_scans():
    result = redact_pdf(signed_pdf(), [])
    assert result.scan_signatures == 0 and result.vector_signatures == 1


def test_scan_signatures_off_with_image_redaction_off():
    result = redact_pdf(scanned_pdf(), [], redact_images=False)
    assert result.scan_signatures == 0
    assert ink_share(result.pdf, pymupdf.Rect(120, 690, 240, 702)) < 0.5
