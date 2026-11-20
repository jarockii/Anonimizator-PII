"""Testy etapu NER. Model GLiNER2 jest podmieniany (mock) – testy sprawdzają
progi, mapowanie etykiet i miejsce NER w pipeline. Ostatni test uruchamia
prawdziwy model, jeśli jest zainstalowany i pobrany."""

import pytest

from engine import llm_masker, ner_masker
from engine.pipeline import anonymize
from engine.vault import Vault


class FakeModel:
    def __init__(self, entities: dict):
        self.entities = entities
        self.calls = []

    def extract_entities(self, text, labels, threshold, include_confidence):
        self.calls.append({"text": text, "labels": labels, "threshold": threshold})
        return {"entities": {
            label: [item for item in items if item["text"] in text]
            for label, items in self.entities.items()
        }}


@pytest.fixture
def fake_ner(monkeypatch):
    def install(entities: dict) -> FakeModel:
        model = FakeModel(entities)
        monkeypatch.setattr(ner_masker, "load_model", lambda: model)
        return model
    return install


def test_labels_are_mapped_to_our_categories(fake_ner):
    fake_ner({"person": [{"text": "Jan Kowalski", "confidence": 0.9}],
              "company": [{"text": "Optima Sp. z o.o.", "confidence": 0.8}],
              "username": [{"text": "tnowak", "confidence": 0.95}]})
    found = ner_masker.detect_entities("Jan Kowalski z Optima Sp. z o.o., login tnowak")
    assert {(e.original, e.tag) for e in found} == {
        ("Jan Kowalski", "OSOBA"), ("Optima Sp. z o.o.", "FIRMA"), ("tnowak", "LOGIN")}


def test_low_confidence_is_dropped_per_label(fake_ner, monkeypatch):
    monkeypatch.setitem(ner_masker.THRESHOLDS, "person", 0.7)
    fake_ner({"person": [{"text": "Pracownik", "confidence": 0.55},
                         {"text": "Anna Nowak", "confidence": 0.85}]})
    found = ner_masker.detect_entities("Pracownik Anna Nowak")
    assert [e.original for e in found] == ["Anna Nowak"]


def test_unknown_labels_are_ignored(fake_ner):
    fake_ner({"product": [{"text": "Toyota", "confidence": 0.99}]})
    assert ner_masker.detect_entities("Toyota") == []


def test_long_text_is_split_into_pages(fake_ner):
    model = fake_ner({})
    ner_masker.detect_entities("strona 1\fstrona 2\fstrona 3")
    assert [call["text"].strip() for call in model.calls] == ["strona 1", "strona 2", "strona 3"]


def test_pipeline_ner_runs_before_llm(fake_ner, monkeypatch):
    fake_ner({"person": [{"text": "Jan Kowalski", "confidence": 0.9}]})
    seen_by_llm = []

    def fake_llm(text, *args, **kwargs):
        seen_by_llm.append(text)
        return [llm_masker.Entity(original="Prezes Zarządu", tag="STANOWISKO")]

    monkeypatch.setattr(llm_masker, "_detect_in_chunk", fake_llm)
    result = anonymize("Prezes Zarządu Jan Kowalski, PESEL 92031502349", Vault(), use_ner=True)
    # LLM widzi tekst już po Regexie i NER.
    assert "Jan Kowalski" not in seen_by_llm[0] and "[OSOBA_1]" in seen_by_llm[0]
    assert result.masked_text == "[STANOWISKO_1] [OSOBA_1], PESEL [PESEL_1]"
    assert {e["źródło"] for e in result.entities} == {"regex", "ner", "llm"}


def test_ner_entity_outside_text_is_rejected(fake_ner):
    # NER wycina fragmenty z tekstu, ale i tak sprawdzamy (wspólny apply_entities).
    vault = Vault()
    entities = [ner_masker.NerEntity("Ktoś Zmyślony", "OSOBA", 0.9)]
    masked, applied = llm_masker.apply_entities("Tekst bez osób.", entities, vault, source="ner")
    assert masked == "Tekst bez osób." and applied == [] and len(vault) == 0


def test_missing_ner_does_not_block_when_llm_runs(monkeypatch):
    def unavailable():
        raise ner_masker.NerUnavailableError("brak gliner2")

    monkeypatch.setattr(ner_masker, "load_model", unavailable)
    monkeypatch.setattr(llm_masker, "_detect_in_chunk", lambda *a, **k: [])
    result = anonymize("Jan Kowalski", Vault(), use_ner=True)
    assert result.ner_error == "brak gliner2"
    assert not result.needs_review


def test_no_ner_and_no_llm_requires_review(monkeypatch):
    def unavailable():
        raise ner_masker.NerUnavailableError("brak gliner2")

    monkeypatch.setattr(ner_masker, "load_model", unavailable)
    assert anonymize("Jan Kowalski", Vault(), use_ner=True, use_llm=False).needs_review


def test_person_with_title_is_masked_exactly():
    # Prawdziwy wyciek (przyklady/2_*.pdf): NER zwrócił „lek. med. Andrzej
    # Kamiński”, dopasowanie form gubiło kropkę i osoba przepadała.
    text = "Leczenie prowadził\nlek. med. Andrzej Kamiński. Lekarz: lek. med. Andrzej Kamiński"
    entities = [ner_masker.NerEntity("lek. med. Andrzej Kamiński", "OSOBA", 0.86)]
    masked, _ = llm_masker.apply_entities(text, entities, Vault(), source="ner")
    assert "Kamiński" not in masked


@pytest.mark.parametrize("tag, value, ok", [
    ("KWOTA", "NIP", False), ("KWOTA", "netto", False), ("KWOTA", "30 dni", False),
    ("KWOTA", "09:12:44", False), ("KWOTA", "2024", False), ("KWOTA", "1200 zł", True),
    ("KWOTA", "15 tys. EUR", True), ("STANOWISKO", "J44.1", False), ("STANOWISKO", "§1", False),
    ("STANOWISKO", "Prezes Zarządu", True), ("OSOBA", "561 10446533", False),
    ("OSOBA", "tomasz nowak", True),  # nazwisko małymi literami (log) NIE jest odrzucane
])
def test_category_sanity_rules(tag, value, ok):
    assert llm_masker.is_plausible(tag, value) is ok


# --- Prawdziwy model -----------------------------------------------------------------


@pytest.mark.slow
def test_real_gliner2_finds_polish_person():
    if not ner_masker.ner_installed():
        pytest.skip("gliner2 nie jest zainstalowany")
    try:
        ner_masker.load_model()
    except ner_masker.NerUnavailableError as error:
        pytest.skip(str(error))
    found = ner_masker.detect_entities(
        "Umowę podpisała Małgorzata Szczepańska, zam. ul. Lipowa 7/3, 80-180 Gdańsk.")
    assert any(e.tag == "OSOBA" and "Szczepańska" in e.original for e in found)
