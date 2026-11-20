"""Przełącznik „zostaw jawnie” per kategoria – np. diagnozy dla AI streszczającego
dokumentację medyczną, przy nadal zamaskowanych osobach i PESEL-ach."""

import json

import pytest

from engine import llm_masker
from engine.llm_masker import Entity, sensitive_category
from engine.pipeline import KEEPABLE_CATEGORIES, anonymize
from engine.redactor import redact_text_to_pdf
from engine.vault import Vault

TEXT = ("Pacjent Jan Kowalski, PESEL 92031502349, choruje na cukrzycę typu 2 "
        "i jest członkiem OPZZ. Koszt: 450 zł.")


@pytest.fixture
def fake_llm(monkeypatch):
    """Główne przejście znajduje osobę, wąskie – dwie dane szczególne."""
    monkeypatch.setattr(llm_masker, "_detect_in_chunk",
                        lambda *a, **k: [Entity(original="Jan Kowalski", tag="OSOBA")])

    def fake_post(url, json, timeout):
        content = {"fragments": ["choruje na cukrzycę typu 2", "jest członkiem OPZZ"]}

        class Reply:
            status_code, ok, text = 200, True, ""

            def json(self):
                return {"message": {"content": _dumps(content)}}
        return Reply()

    monkeypatch.setattr(llm_masker.requests, "post", fake_post)


def _dumps(value):
    return json.dumps(value, ensure_ascii=False)


@pytest.mark.parametrize("fragment, category", [
    ("choruje na cukrzycę typu 2", "ZDROWIE"),
    ("zaostrzenie przewlekłej obturacyjnej choroby płuc", "ZDROWIE"),
    ("migotanie przedsionków (I48)", "ZDROWIE"),
    ("leczy się psychiatrycznie z powodu depresji", "ZDROWIE"),
    ("po kardiowersji stabilny", "ZDROWIE"),      # prawdziwa odpowiedź qwen2.5:3b
    ("jest członkiem OPZZ", "WRAZLIWE"),
    ("Świadek Jehowy", "WRAZLIWE"),
    ("katolik po zawale serca", "WRAZLIWE"),     # mieszane -> bezpieczniejsza kategoria
    ("głosuje na prawicę", "WRAZLIWE"),          # brak podpowiedzi -> bezpieczniejsza
])
def test_health_is_split_from_other_sensitive_data(fragment, category):
    assert sensitive_category(fragment) == category


def test_by_default_diagnosis_is_masked(fake_llm):
    result = anonymize(TEXT, Vault())
    assert "cukrzycę" not in result.masked_text
    assert "[ZDROWIE_1]" in result.masked_text and result.kept_entities == []


def test_keep_health_leaves_diagnosis_but_masks_the_rest(fake_llm):
    vault = Vault()
    result = anonymize(TEXT, vault, keep_categories=["ZDROWIE"])
    assert result.masked_text == ("Pacjent [OSOBA_1], PESEL [PESEL_1], choruje na cukrzycę typu 2 "
                                  "i [WRAZLIWE_1]. Koszt: [KWOTA_1].")
    assert [e["kategoria"] for e in result.kept_entities] == ["ZDROWIE"]
    assert all(e["kategoria"] != "ZDROWIE" for e in result.entities)
    # De-anonimizacja odpowiedzi AI nadal działa.
    assert vault.restore(result.masked_text) == TEXT


def test_kept_values_get_no_black_bar(fake_llm):
    result = anonymize(TEXT, Vault(), keep_categories=["ZDROWIE"])
    redacted = redact_text_to_pdf(TEXT, [e["oryginał"] for e in result.entities])
    import pymupdf
    pdf_text = pymupdf.open(stream=redacted.pdf, filetype="pdf")[0].get_text()
    assert "cukrzycę" in pdf_text and "Kowalski" not in pdf_text and "OPZZ" not in pdf_text


def test_several_categories_at_once(fake_llm):
    result = anonymize(TEXT, Vault(), keep_categories=["ZDROWIE", "KWOTA"])
    assert "450 zł" in result.masked_text and "cukrzycę" in result.masked_text
    assert "Kowalski" not in result.masked_text


@pytest.mark.parametrize("category", ["OSOBA", "PESEL", "ADRES", "WRAZLIWE", "POSREDNI", "SECRET"])
def test_identifiers_can_never_be_kept(category):
    assert category not in KEEPABLE_CATEGORIES
    with pytest.raises(ValueError, match=category):
        anonymize("Jan Kowalski", Vault(), use_llm=False, keep_categories=[category])
