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


def fake_fixes_reply(fixes: list[dict]):
    import json as _json

    class Reply:
        status_code, ok, text = 200, True, ""

        def json(self):
            return {"message": {"content": _json.dumps({"fixes": fixes})}}
    return Reply()


def test_llm_fixes_category_of_unsure_ner_entity(fake_ner, monkeypatch):
    # Prawdziwy przypadek: NER oznaczył inicjały „M.Z.” jako FIRMA.
    fake_ner({"company": [{"text": "M.Z.", "confidence": 0.7}],
              "person": [{"text": "Jan Kowalski", "confidence": 0.99}]})
    sent = []

    def fake_post(url, json, timeout):
        sent.append(json["messages"][1]["content"])
        return fake_fixes_reply([{"id": 1, "tag": "OSOBA"}])

    monkeypatch.setattr(llm_masker.requests, "post", fake_post)
    monkeypatch.setattr(llm_masker, "_detect_in_chunk", lambda *a, **k: [])
    result = anonymize("Pracownik o inicjałach M.Z. i Jan Kowalski.", Vault(), use_ner=True,
                       verify_ner=True)
    # Na liście do sprawdzenia jest tylko niepewne „M.Z.” (pewne wykrycia nie idą do LLM).
    assert '"M.Z." (teraz: FIRMA)' in sent[0] and '"Jan Kowalski" (teraz' not in sent[0]
    assert result.ner_fixes == {"M.Z.": "OSOBA"}
    assert result.masked_text == "Pracownik o inicjałach [OSOBA_1] i [OSOBA_2]."
    assert {(e["oryginał"], e["źródło"]) for e in result.entities} == {
        ("M.Z.", "ner+llm"), ("Jan Kowalski", "ner")}


def test_llm_verification_never_unmasks(fake_ner, monkeypatch):
    # Nawet gdy LLM „zgubi” pozycję albo poda bzdurny numer – wykrycie zostaje.
    fake_ner({"person": [{"text": "Ewa Nowak", "confidence": 0.6}]})
    monkeypatch.setattr(llm_masker.requests, "post",
                        lambda *a, **k: fake_fixes_reply([{"id": 7, "tag": "INNE"}]))
    monkeypatch.setattr(llm_masker, "_detect_in_chunk", lambda *a, **k: [])
    result = anonymize("Podpis: Ewa Nowak", Vault(), use_ner=True, verify_ner=True)
    assert result.masked_text == "Podpis: [OSOBA_1]"


def test_verification_is_off_by_default(fake_ner, monkeypatch):
    # qwen2.5:3b psuł kategorie częściej, niż je poprawiał (pomiar na przyklady/5–8).
    fake_ner({"company": [{"text": "M.Z.", "confidence": 0.7}]})
    from engine import pipeline
    calls = []
    # Podmieniamy funkcję tam, gdzie pipeline jej używa (import po nazwie).
    monkeypatch.setattr(pipeline, "verify_ner_categories",
                        lambda *a, **k: calls.append(1) or {})
    monkeypatch.setattr(llm_masker, "_detect_in_chunk", lambda *a, **k: [])
    anonymize("inicjały M.Z.", Vault(), use_ner=True)
    assert calls == []
    anonymize("inicjały M.Z.", Vault(), use_ner=True, verify_ner=True)
    assert calls == [1]  # kontrola: z flagą funkcja JEST wywoływana


def test_sensitive_pass_runs_only_on_hinted_chunks(monkeypatch):
    import json as _json
    asked = []

    def fake_post(url, json, timeout):
        user = json["messages"][1]["content"]
        if "fragments" in _json.dumps(json["format"]):
            asked.append(user)
            content = {"fragments": ["leczy się psychiatrycznie"]}
        else:
            content = {"entities": []}

        class Reply:
            status_code, ok, text = 200, True, ""

            def json(self):
                return {"message": {"content": _json.dumps(content)}}
        return Reply()

    monkeypatch.setattr(llm_masker.requests, "post", fake_post)
    text = "Raport kwartalny bez danych.\f[OSOBA_1] od 2019 r. leczy się psychiatrycznie."
    result = anonymize(text, Vault())
    assert len(asked) == 1 and "psychiatrycznie" in asked[0]  # strona 1 nie poszła do LLM
    assert "[ZDROWIE_1]" in result.masked_text  # leczenie = dane o zdrowiu


def test_verification_error_keeps_ner_categories(fake_ner):
    # Ollama zablokowana (conftest) – kategorie z NER zostają, tekst zamaskowany.
    fake_ner({"company": [{"text": "M.Z.", "confidence": 0.7}]})
    result = anonymize("inicjały M.Z.", Vault(), use_ner=True)
    assert result.masked_text == "inicjały [FIRMA_1]"
    assert result.needs_review  # bo etap LLM (wykrywanie) też nie zadziałał


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


def test_surname_alone_is_masked_after_full_name():
    # Prawdziwy wyciek: „Nazwisko: PIOTROWSKI” na skanie dowodu (przyklady/4_*.pdf).
    vault = Vault()
    text = "DAMIAN PIOTROWSKI – pożyczkobiorca.\nNazwisko: PIOTROWSKI\nImiona: DAMIAN"
    masked, _ = llm_masker.apply_entities(
        text, [ner_masker.NerEntity("DAMIAN PIOTROWSKI", "OSOBA", 0.98)], vault, source="ner")
    masked, applied = vault.mask_surnames(masked)
    assert "PIOTROWSKI" not in masked
    assert applied[0]["źródło"] == "propagacja"
    assert vault.restore(masked) == text


def test_surname_propagation_skips_short_and_lowercase():
    vault = Vault()
    vault.tokenize("OSOBA", "Wojciech Lis")         # „Lis” za krótkie – zostaje
    vault.tokenize("OSOBA", "Anna Kowal")
    masked, _ = vault.mask_surnames("Lis biegł. Pan Kowal i kowal z kuźni.")
    assert masked == "Lis biegł. Pan [OSOBA_3] i kowal z kuźni."


@pytest.mark.parametrize("tag, value, ok", [
    ("KWOTA", "NIP", False), ("KWOTA", "netto", False), ("KWOTA", "30 dni", False),
    ("KWOTA", "09:12:44", False), ("KWOTA", "2024", False), ("KWOTA", "1200 zł", True),
    ("KWOTA", "15 tys. EUR", True), ("STANOWISKO", "J44.1", False), ("STANOWISKO", "§1", False),
    ("STANOWISKO", "Prezes Zarządu", True), ("OSOBA", "561 10446533", False),
    ("OSOBA", "tomasz nowak", True),  # nazwisko małymi literami (log) NIE jest odrzucane
    # Prawdziwe odpowiedzi qwen2.5:3b na przyklady/*.pdf:
    ("POSREDNI", "syn burmistrza Kutna", True), ("POSREDNI", "kancelarii brata prezesa", True),
    ("POSREDNI", "operator wózka widłowego nr 7", True), ("POSREDNI", "wózka", False),
    ("POSREDNI", "Data i adres IP", False), ("POSREDNI", "Urzędem Skarbowym w Toruniu", False),
    ("WRAZLIWE", "leczy się psychiatrycznie z powodu depresji", True),
    ("WRAZLIWE", "od 2019 r.", False),
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
