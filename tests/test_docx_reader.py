"""Odczyt CAŁEGO dokumentu Word: nagłówki, stopki, przypisy, pola tekstowe,
kontrolki treści, tabele zagnieżdżone, śledzenie zmian (engine/docx_reader.py).

Pliki składamy z surowego XML (tests/docx_raw.py) – python-docx nie umie
utworzyć pól tekstowych ani przypisów, a w prawdziwych pismach z Worda są.
"""

import io

import pymupdf
import pytest
from docx import Document

import docx_raw
from engine import ocr
from engine.file_reader import read_docx, read_uploaded_file
from engine.pipeline import anonymize
from engine.vault import Vault

HIDDEN_BEFORE = ["Kazimierz Wilk", "Oliwia Szpak", "Biuro Rachunkowe Dąb", "Bogumiła Czajka",
                 "Jan Bocian", "Witold Kret", "Haliny Sowy", "Teofilowi Gilowi"]


@pytest.fixture(autouse=True)
def no_ocr(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: False)
    ocr.clear_cache()
    yield
    ocr.clear_cache()


@pytest.mark.parametrize("value", HIDDEN_BEFORE)
def test_every_part_of_the_document_is_read_exactly_once(value):
    # Wcześniej ŻADNA z tych wartości nie trafiała do tekstu.
    assert read_docx(docx_raw.sample()).count(value) == 1


def test_reading_order_header_body_notes_footer():
    lines = read_docx(docx_raw.sample()).splitlines()
    assert lines[0].startswith("Kancelaria Notarialna")           # nagłówek
    assert lines[1] == "UMOWA NAJMU LOKALU"                      # treść
    assert lines.index("Koniec umowy.") < lines.index("Pełnomocnictwo dla Haliny Sowy z 3 marca 2025 r.")
    assert lines[-1].startswith("Sporządził:")                   # stopka


def test_inline_content_control_stays_inside_its_sentence():
    assert "Najemca: Oliwia Szpak, PESEL 85042311448" in read_docx(docx_raw.sample())


def test_textbox_is_not_doubled_by_vml_fallback():
    data = docx_raw.build(docx_raw.textbox("Pieczęć: Jadwiga Kos") + docx_raw.p("Treść"))
    assert read_docx(data).splitlines() == ["Pieczęć: Jadwiga Kos", "Treść"]


def test_nested_table_in_cell():
    inner = docx_raw.table([["Imię", "Lucjan"], ["Nazwisko", "Gajewski"]])
    data = docx_raw.build(docx_raw.table([["Poręczyciel", inner]]))
    assert read_docx(data) == "Poręczyciel | Imię | Lucjan Nazwisko | Gajewski"


def test_footnote_separators_are_not_text():
    data = docx_raw.build(docx_raw.p("Treść"), footnotes=["Przypis pierwszy."])
    assert read_docx(data).splitlines() == ["Treść", "Przypis pierwszy."]


def test_tracked_changes_keep_inserted_and_drop_deleted_text():
    paragraph = ('<w:p><w:r><w:t xml:space="preserve">Najemca: </w:t></w:r>'
                 '<w:del w:id="1" w:author="x"><w:r><w:delText>Adam Stary</w:delText></w:r></w:del>'
                 '<w:ins w:id="2" w:author="x"><w:r><w:t>Ewa Nowa</w:t></w:r></w:ins></w:p>')
    # Tekst usunięty nie jest widoczny i nie trafia do AI; wstawiony – tak.
    assert read_docx(docx_raw.build(paragraph)) == "Najemca: Ewa Nowa"


def test_merged_cells_are_not_repeated():
    # python-docx powtarzał tekst scalonej komórki w każdej kolumnie, którą zajmuje.
    document = Document()
    table = document.add_table(rows=1, cols=3)
    merged = table.cell(0, 0).merge(table.cell(0, 1))
    merged.text = "Wynajmujący: Marian Kos"
    table.cell(0, 2).text = "podpis"
    buffer = io.BytesIO()
    document.save(buffer)
    assert read_docx(buffer.getvalue()).count("Marian Kos") == 1


def test_header_and_footer_with_python_docx():
    document = Document()
    document.sections[0].header.paragraphs[0].text = "Sygn. akt I C 412/25, sędzia Irena Mak"
    document.sections[0].footer.paragraphs[0].text = "Strona 1 – sporządziła Ida Wróbel"
    document.add_paragraph("Treść pisma.")
    buffer = io.BytesIO()
    document.save(buffer)
    lines = read_docx(buffer.getvalue()).splitlines()
    assert lines == ["Sygn. akt I C 412/25, sędzia Irena Mak", "Treść pisma.",
                     "Strona 1 – sporządziła Ida Wróbel"]


def test_image_in_header_goes_through_ocr(monkeypatch):
    words = [(0.1, 0.1, 0.4, 0.4, "Notariusz", 0, 0), (0.45, 0.1, 0.9, 0.4, "Bocian", 0, 0)]
    monkeypatch.setattr(ocr, "ocr_available", lambda: True)
    monkeypatch.setattr(ocr, "run_tesseract", lambda pixmap: words)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 120), False)
    pixmap.clear_with(230)
    document = Document()
    document.sections[0].header.paragraphs[0].add_run().add_picture(io.BytesIO(pixmap.tobytes("png")))
    document.add_paragraph("Treść.")
    buffer = io.BytesIO()
    document.save(buffer)
    data = buffer.getvalue()
    assert read_docx(data).splitlines()[0] == "Notariusz Bocian"
    assert ocr.ocr_docx(data).images_ocr == 1


def test_footer_data_is_masked_in_pipeline():
    text = read_uploaded_file("umowa.docx", docx_raw.sample())
    result = anonymize(text, Vault(), use_llm=False)
    for value in ("512 300 400", "600 700 800", "85042311448"):
        assert value not in result.masked_text
