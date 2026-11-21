"""Testy zestawu pomiarowego eval/: logika oceny i poprawność oznaczeń."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from scoring import Annotation, Item, load_annotations, missing_fragments, score_document, visible  # noqa: E402

from engine import ocr  # noqa: E402
from engine.file_reader import read_uploaded_file  # noqa: E402


@pytest.mark.parametrize("fragment, text, expected", [
    ("Grzegorz", "Prezes Grzegorza Wróblewskiego", False),   # całe słowa, nie podciąg
    ("Grzegorza", "Prezes Grzegorza Wróblewskiego", True),
    ("Lis", "Pan Lisowski", False),
    ("Lis", "PESEL [PESEL_1], Wojciech LIS", True),          # bez wielkości liter
    ("Grzegorza Wróblewskiego", "Grzegorza\nWróblewskiego", True),  # złamanie linii
    ("w.lis@skrzynka-test.pl", "kontakt: [EMAIL_1]", False),
    ("(at) wp", "mietek.zuraw (at) wp kropka pl", True),     # fragment z nawiasem
])
def test_visible(fragment, text, expected):
    assert visible(fragment, text) is expected


def make_annotation() -> Annotation:
    return Annotation("test", Path("x.pdf"), "", [
        Item("OSOBA", ["Jan", "Kowalski"], ""),
        Item("PESEL", ["92031502349"], ""),
        Item("ZDROWIE", ["cukrzycę"], "diagnoza"),
    ], ["Pacjent", "netto"])


def test_score_counts_partial_name_as_leak():
    score = score_document(make_annotation(),
                           "Pacjent [OSOBA_1] Kowalski, PESEL [PESEL_1], [ZDROWIE_1], [KWOTA_1] netto")
    assert [i.fragments[0] for i in score.leaked] == ["Jan"]    # samo nazwisko też jest wyciekiem
    assert [i.category for i in score.hidden] == ["PESEL", "ZDROWIE"]
    assert score.over_masked == []


def test_score_over_masking_and_kept_categories():
    score = score_document(make_annotation(), "[STANOWISKO_1] [OSOBA_1], choruje na cukrzycę",
                           kept_categories={"ZDROWIE"})
    assert score.leaked == [] and [i.category for i in score.kept] == ["ZDROWIE"]
    assert score.over_masked == ["Pacjent", "netto"]


ANNOTATIONS = load_annotations(ROOT / "eval" / "oznaczenia", ROOT)


def test_item_labels_are_unique_per_document():
    # Wspólny opis („celownik”) zlewał dwie pozycje w jedną w liczniku wycieków.
    for annotation in ANNOTATIONS:
        labels = [item.label for item in annotation.items]
        assert len(labels) == len(set(labels)), annotation.name


def test_every_sample_pdf_is_annotated():
    annotated = {a.path.name for a in ANNOTATIONS}
    assert annotated == {p.name for p in (ROOT / "przyklady").glob("*.pdf")}


@pytest.mark.parametrize("annotation", ANNOTATIONS, ids=lambda a: a.name[:2])
def test_annotations_match_document_text(annotation):
    """Literówka w oznaczeniu = dana „zawsze ukryta” – zawyżony wynik pomiaru."""
    if not ocr.ocr_available():
        pytest.skip("część oznaczeń pochodzi z OCR – potrzebny Tesseract")
    text = read_uploaded_file(annotation.path.name, annotation.path.read_bytes())
    assert missing_fragments(annotation, text) == []
