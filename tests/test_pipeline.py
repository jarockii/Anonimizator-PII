"""Testy pipeline'u. Nie wymagają działającej Ollamy – odpowiedzi LLM-a
podmieniamy (mock), dzięki czemu testy są szybkie i powtarzalne."""

import json

import pymupdf
import pytest
import requests

from engine import llm_masker
from engine.llm_masker import (
    Entity,
    OllamaConnectionError,
    OllamaError,
    apply_entities,
    detect_entities,
    find_person_forms,
    parse_entities,
    split_into_chunks,
)
from engine.pipeline import anonymize
from engine.redactor import find_value_rects, page_words, redact_pdf, remove_images
from engine.regex_masker import is_valid_iban, is_valid_nip, is_valid_pesel, mask_with_regex
from engine.vault import Vault, VaultTokenError


@pytest.fixture
def vault() -> Vault:
    return Vault()


# --- Walidatory -------------------------------------------------------------


def test_pesel_checksum():
    assert is_valid_pesel("44051401359")
    assert not is_valid_pesel("44051401358")  # zła cyfra kontrolna
    assert not is_valid_pesel("1234")


def test_nip_checksum():
    assert is_valid_nip("526-025-02-74")
    assert not is_valid_nip("526-025-02-75")


def test_iban_checksum():
    assert is_valid_iban("PL61 1090 1014 0000 0712 1981 2874")
    assert is_valid_iban("61109010140000071219812874")  # NRB bez „PL”
    assert not is_valid_iban("PL61 1090 1014 0000 0712 1981 2875")


# --- Etap 1: Regex ------------------------------------------------------------


def test_regex_masks_email(vault):
    masked, found = mask_with_regex("Pisz na jan.kowalski@poczta.pl dziś.", vault)
    assert masked == "Pisz na [EMAIL_1] dziś."
    assert found[0]["oryginał"] == "jan.kowalski@poczta.pl"


def test_regex_masks_valid_pesel_only(vault):
    masked, _ = mask_with_regex("PESEL 44051401359, zamówienie 12345678901.", vault)
    assert "[PESEL_1]" in masked
    # 11 cyfr bez poprawnej sumy kontrolnej to nie PESEL – zostaje.
    assert "12345678901" in masked


@pytest.mark.parametrize("nip", ["5260250274", "526-025-02-74", "PL 5260250274"])
def test_regex_masks_nip_formats(vault, nip):
    masked, _ = mask_with_regex(f"NIP: {nip}.", vault)
    assert masked == "NIP: [NIP_1]."


@pytest.mark.parametrize("iban", [
    "PL61 1090 1014 0000 0712 1981 2874",
    "PL61109010140000071219812874",
    "61 1090 1014 0000 0712 1981 2874",
])
def test_regex_masks_iban_formats(vault, iban):
    masked, _ = mask_with_regex(f"Konto: {iban}", vault)
    assert masked == "Konto: [IBAN_1]"


@pytest.mark.parametrize("phone", [
    "+48 601 234 567", "601-234-567", "601234567", "(22) 628 41 00", "0048601234567",
])
def test_regex_masks_phone_formats(vault, phone):
    masked, _ = mask_with_regex(f"Tel. {phone}, dzięki", vault)
    assert masked == "Tel. [TEL_1], dzięki"


@pytest.mark.parametrize("amount", [
    "18 500 zł", "4 850,00 zł", "12 999,99 PLN", "1200 zł", "3000 EUR", "99,99 zł",
    "1.500,00 €", "1 500 zł", "15 tys. zł", "2 mln PLN", "100 złotych",
    "$1,200.50", "PLN 1 000", "250 USD", "250\nzł",
])
def test_regex_masks_amount_formats(vault, amount):
    masked, found = mask_with_regex(f"Do zapłaty: {amount}. Dziękujemy", vault)
    assert masked == "Do zapłaty: [KWOTA_1]. Dziękujemy"
    assert found[0]["oryginał"] == amount
    assert vault.restore(masked) == f"Do zapłaty: {amount}. Dziękujemy"


@pytest.mark.parametrize("text", [
    "Kontrola za 3 miesiące, w drugim kwartale.",
    "Nr księgi głównej: 2025/00417, umowa 17/2025.",
    "retry_count=3 order_id=10000000001",
    "Zysk wzrósł o 15%, a VAT wynosi 23%.",
    "Zaplanowano 150 zabiegów w 12 salach.",
])
def test_regex_ignores_numbers_without_currency(vault, text):
    masked, found = mask_with_regex(text, vault)
    assert masked == text
    assert found == []


def test_large_amount_is_not_mistaken_for_phone(vault):
    masked, _ = mask_with_regex("Budżet: 150 000 000 zł, tel. 601 234 567", vault)
    assert masked == "Budżet: [KWOTA_1], tel. [TEL_1]"


def test_regex_does_not_mask_dates_and_short_numbers(vault):
    text = "O 09:41 przyjęto 3 osoby, umowa nr 17/2025, art. 6 ust. 1."
    masked, found = mask_with_regex(text, vault)
    assert masked == text
    assert found == []


def test_regex_masks_password_but_keeps_key(vault):
    masked, _ = mask_with_regex('{"user": "x", "password": "Zima2025!@#"}', vault)
    assert masked == '{"user": "x", "password": "[HASLO_1]"}'
    assert vault.get("[HASLO_1]") == "Zima2025!@#"


def test_regex_masks_bearer_token_and_url_password(vault):
    text = ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc "
            "db=postgres://admin:S3cr3t@10.0.0.12:5432/prod")
    masked, _ = mask_with_regex(text, vault)
    assert "eyJhbGciOiJIUzI1NiJ9" not in masked
    assert "S3cr3t" not in masked
    assert "10.0.0.12" not in masked
    assert "Bearer [SECRET_1]" in masked


def test_same_value_gets_same_token(vault):
    masked, _ = mask_with_regex("a@b.pl, potem znowu a@b.pl i c@d.pl", vault)
    assert masked == "[EMAIL_1], potem znowu [EMAIL_1] i [EMAIL_2]"


def test_empty_text(vault):
    assert mask_with_regex("", vault) == ("", [])


# --- Sejf / de-anonimizacja ---------------------------------------------------


def test_roundtrip_restores_original(vault):
    original = ("Jan, PESEL 44051401359, mail jan@x.pl, tel. 601 234 567, "
                "konto PL61 1090 1014 0000 0712 1981 2874.")
    masked, _ = mask_with_regex(original, vault)
    assert "44051401359" not in masked
    assert vault.restore(masked) == original


def test_restore_tolerates_ai_formatting(vault):
    vault.tokenize("OSOBA", "Jan Kowalski")
    response = "Szanowny [ OSOBA_1 ], \\[OSOBA_1\\] oraz [osoba_1]."
    assert vault.restore(response) == "Szanowny Jan Kowalski, Jan Kowalski oraz Jan Kowalski."


def test_restore_leaves_unknown_tokens(vault):
    vault.tokenize("OSOBA", "Jan Kowalski")
    response = "[OSOBA_1] i [OSOBA_7] oraz [link]"
    assert vault.restore(response) == "Jan Kowalski i [OSOBA_7] oraz [link]"
    assert vault.unknown_tokens(response) == ["[OSOBA_7]"]


def test_restore_does_not_chain_replacements(vault):
    # Oryginał wygląda jak token – nie może zostać podmieniony drugi raz.
    vault.tokenize("HASLO", "[EMAIL_1]")
    vault.tokenize("EMAIL", "a@b.pl")
    assert vault.restore("[HASLO_1]") == "[EMAIL_1]"


def test_vault_counters_per_category(vault):
    assert vault.tokenize("OSOBA", "A") == "[OSOBA_1]"
    assert vault.tokenize("FIRMA", "B") == "[FIRMA_1]"
    assert vault.tokenize("OSOBA", "C") == "[OSOBA_2]"
    vault.clear()
    assert len(vault) == 0
    assert vault.tokenize("OSOBA", "D") == "[OSOBA_1]"


# --- Etap 2: deterministyczna podmiana encji z LLM -----------------------------


def test_apply_entities_replaces_longest_first(vault):
    text = "Jan Kowalski spotkał Jana. Jan Kowalski wrócił."
    entities = [Entity(original="Jan", tag="OSOBA"),
                Entity(original="Jan Kowalski", tag="OSOBA")]
    masked, _ = apply_entities(text, entities, vault)
    assert masked == "[OSOBA_1] spotkał Jana. [OSOBA_1] wrócił."


def test_apply_entities_ignores_hallucinations(vault):
    text = "Umowę podpisała Anna Nowak."
    entities = [Entity(original="Piotr Zieliński", tag="OSOBA"),  # nie ma w tekście
                Entity(original="Anna Nowak", tag="OSOBA")]
    masked, applied = apply_entities(text, entities, vault)
    assert masked == "Umowę podpisała [OSOBA_1]."
    assert len(applied) == 1


def test_apply_entities_respects_word_boundaries(vault):
    masked, _ = apply_entities("Janusz i Jan", [Entity(original="Jan", tag="OSOBA")], vault)
    assert masked == "Janusz i [OSOBA_1]"


def test_apply_entities_skips_existing_tokens(vault):
    text = "Kontakt: [EMAIL_1]"
    masked, applied = apply_entities(text, [Entity(original="[EMAIL_1]", tag="OSOBA")], vault)
    assert masked == text
    assert applied == []


def test_apply_entities_never_breaks_existing_tokens(vault):
    # Prawdziwy przypadek z qwen2.5:3b: model zwrócił wnętrze tokenu „TEL_1”.
    vault.tokenize("TEL", "601 234 567")
    text = "Tel. [TEL_1], dzwonić po 16."
    masked, applied = apply_entities(text, [Entity(original="TEL_1", tag="ADRES")], vault)
    assert masked == text
    assert applied == []
    assert vault.restore(masked) == "Tel. 601 234 567, dzwonić po 16."


def test_apply_entities_finds_declined_person_forms(vault):
    # Prawdziwy przypadek z qwen2.5:3b: model zwraca formę podstawową,
    # a w tekście jest forma odmieniona.
    text = "Skierowany przez Piotra Wiśniewskiego. Stan Jana Kowalskiego dobry."
    entities = [Entity(original="Piotr Wiśniewski", tag="OSOBA"),
                Entity(original="Jan Kowalski", tag="OSOBA")]
    masked, _ = apply_entities(text, entities, vault)
    assert masked == "Skierowany przez [OSOBA_1]. Stan [OSOBA_2] dobry."
    assert vault.restore(masked) == text


def test_person_forms_single_word_is_exact_only():
    assert find_person_forms("Janusz i Janina", "Jan") == []
    assert find_person_forms("Pan Jan", "Jan") == ["Jan"]


def test_person_forms_do_not_match_other_people():
    text = "Anna Kowalska i Jan Kowalski"
    assert find_person_forms(text, "Jan Kowalski") == ["Jan Kowalski"]


def test_split_into_chunks_keeps_whole_lines():
    text = "\n".join(f"linia {i} " + "x" * 50 for i in range(20))
    chunks = split_into_chunks(text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert "".join(chunks).rstrip("\n") == text


# --- Komunikacja z Ollamą (mock) -----------------------------------------------


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.ok = status_code < 400
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


def fake_ollama_reply(entities: list[dict]) -> FakeResponse:
    return FakeResponse(payload={"message": {"content": json.dumps({"entities": entities})}})


def test_detect_entities_sends_json_schema(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return fake_ollama_reply([{"original": "Anna Nowak", "tag": "OSOBA"}])

    monkeypatch.setattr(llm_masker.requests, "post", fake_post)
    entities = detect_entities("Anna Nowak")
    assert entities[0].original == "Anna Nowak"
    assert captured["options"]["temperature"] == 0.0
    assert captured["format"]["required"] == ["entities"]


def test_request_has_loop_safeguards(monkeypatch):
    # Prawdziwy przypadek: krótki plik, model zapętlił się na 8800 tokenów.
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json, timeout=timeout)
        return fake_ollama_reply([])

    monkeypatch.setattr(llm_masker.requests, "post", fake_post)
    detect_entities("tekst")
    assert captured["format"]["properties"]["entities"]["maxItems"] == 20
    assert captured["options"]["num_predict"] == 600
    assert captured["timeout"] <= 60


def test_truncated_json_keeps_complete_entities():
    content = ('{"entities": [{"original": "Jan Kowalski", "tag": "OSOBA"}, '
               '{"original": "Anna \\"Ania\\" Nowak", "tag": "OSOBA"}, {"original": "Jan Kow')
    entities = parse_entities(content)
    assert [e.original for e in entities] == ["Jan Kowalski", 'Anna "Ania" Nowak']


def test_progress_callback_reports_every_chunk(monkeypatch):
    monkeypatch.setattr(llm_masker.requests, "post", lambda *a, **k: fake_ollama_reply([]))
    calls = []
    text = "\n".join("x" * 100 for _ in range(10))  # ok. 1000 znaków -> kilka fragmentów
    detect_entities(text, on_progress=lambda done, total: calls.append((done, total)))
    total = calls[-1][1]
    assert total > 1
    assert calls == [(n, total) for n in range(1, total + 1)]


def test_detect_entities_rejects_wrong_category(monkeypatch):
    monkeypatch.setattr(llm_masker.requests, "post",
                        lambda *a, **k: fake_ollama_reply([{"original": "X", "tag": "KOT"}]))
    with pytest.raises(OllamaError):
        detect_entities("X")


def test_detect_entities_connection_error(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("odmowa połączenia")

    monkeypatch.setattr(llm_masker.requests, "post", fake_post)
    with pytest.raises(OllamaConnectionError):
        detect_entities("tekst")


def test_detect_entities_missing_model(monkeypatch):
    monkeypatch.setattr(llm_masker.requests, "post",
                        lambda *a, **k: FakeResponse(404, {"error": "model not found"}))
    with pytest.raises(OllamaError, match="ollama pull"):
        detect_entities("tekst")


# --- Cały pipeline ----------------------------------------------------------------


def test_pipeline_full_roundtrip(monkeypatch, vault):
    text = ("Anna Nowak (anna@firma.pl) z TechNova Sp. z o.o. "
            "zapłaci 18 500 zł na PL61 1090 1014 0000 0712 1981 2874.")
    monkeypatch.setattr(llm_masker.requests, "post", lambda *a, **k: fake_ollama_reply([
        {"original": "Anna Nowak", "tag": "OSOBA"},
        {"original": "TechNova Sp. z o.o.", "tag": "FIRMA"},
        {"original": "18 500 zł", "tag": "KWOTA"},
    ]))
    result = anonymize(text, vault)
    assert result.masked_text == ("[OSOBA_1] ([EMAIL_1]) z [FIRMA_1] "
                                  "zapłaci [KWOTA_1] na [IBAN_1].")
    assert result.llm_error is None
    assert len(result.entities) == 5

    cloud_answer = "Przypomnienie dla [OSOBA_1]: przelew [KWOTA_1] na [IBAN_1]."
    assert vault.restore(cloud_answer) == (
        "Przypomnienie dla Anna Nowak: przelew 18 500 zł na "
        "PL61 1090 1014 0000 0712 1981 2874."
    )


def test_pipeline_falls_back_to_regex_when_ollama_down(monkeypatch, vault):
    def fake_post(*args, **kwargs):
        raise requests.ConnectionError()

    monkeypatch.setattr(llm_masker.requests, "post", fake_post)
    result = anonymize("Mail: a@b.pl, Jan Kowalski", vault)
    assert result.masked_text == "Mail: [EMAIL_1], Jan Kowalski"
    assert "ollama serve" in result.llm_error


# --- Polskie identyfikatory urzędowe, pojazdy, dane cyfrowe -------------------------


@pytest.mark.parametrize("text, expected", [
    ("KRS 0000436998, Sąd Rejonowy", "KRS [KRS_1], Sąd Rejonowy"),
    ("wpisana pod nr KRS: 0000123456.", "wpisana pod nr KRS: [KRS_1]."),
    ("Numer 0000436998 w rejestrze", "Numer [KRS_1] w rejestrze"),
])
def test_regex_masks_krs(vault, text, expected):
    assert mask_with_regex(text, vault)[0] == expected


def test_krs_does_not_steal_nip(vault):
    # 10 cyfr bez słowa „KRS” i bez „0000” na początku to NIP, nie KRS.
    assert mask_with_regex("NIP 5260250274", vault)[0] == "NIP [NIP_1]"


@pytest.mark.parametrize("text, expected", [
    ("REGON 123456785", "REGON [REGON_1]"),
    ("REGON: 12345678901235", "REGON: [REGON_1]"),
    ("identyfikator 12345678901235 w GUS", "identyfikator [REGON_1] w GUS"),
])
def test_regex_masks_regon(vault, text, expected):
    assert mask_with_regex(text, vault)[0] == expected


def test_regon_with_bad_checksum_is_ignored(vault):
    # Zła cyfra kontrolna: 9 cyfr nie jest REGON-em, więc łapie je telefon.
    assert mask_with_regex("REGON 123456784", vault)[0] == "REGON [TEL_1]"


def test_regex_masks_bdo(vault):
    assert mask_with_regex("Nr BDO: 000012345.", vault)[0] == "Nr BDO: [BDO_1]."


@pytest.mark.parametrize("text, expected", [
    ("Dowód osobisty ABA300000.", "Dowód osobisty [DOWOD_1]."),
    ("dowód ABC 523456", "dowód [DOWOD_1]"),
    ("Paszport ZS0000177", "Paszport [PASZPORT_1]"),
    ("VIN: 1HGCM82633A004352", "VIN: [VIN_1]"),
])
def test_regex_masks_identity_documents_and_vin(vault, text, expected):
    assert mask_with_regex(text, vault)[0] == expected


def test_identity_checksums_reject_product_codes(vault):
    # „ABC123456” ma złą cyfrę kontrolną – to raczej kod produktu niż dowód.
    text = "Kod produktu ABC123456, faktura FV1234567."
    assert mask_with_regex(text, vault)[0] == text


@pytest.mark.parametrize("plate", ["GD 1234A", "WI 98765", "KR1AB23", "WZ 12345"])
def test_regex_masks_license_plates(vault, plate):
    masked, _ = mask_with_regex(f"Pojazd o nr rej. {plate} przekazano.", vault)
    assert masked == "Pojazd o nr rej. [REJESTRACJA_1] przekazano."


@pytest.mark.parametrize("text", [
    "GET 2000 ms", "PLN 1000 netto", "ISO 9001", "art. 12345", "TechNova2025",
])
def test_plates_ignore_look_alikes(vault, text):
    masked, found = mask_with_regex(text, vault)
    assert not any(item["kategoria"] == "REJESTRACJA" for item in found)


@pytest.mark.parametrize("date_text", [
    "12.03.2025", "1.3.2025", "2025-03-14", "14/03/2025", "12 marca 2025",
])
def test_regex_masks_dates(vault, date_text):
    masked, _ = mask_with_regex(f"Podpisano dnia {date_text} r.", vault)
    assert masked == "Podpisano dnia [DATA_1] r."


@pytest.mark.parametrize("text", ["31.02.2025", "45.13.2025", "nr 17/2025", "wersja 1.2.3"])
def test_regex_ignores_impossible_dates(vault, text):
    masked, found = mask_with_regex(text, vault)
    assert not any(item["kategoria"] == "DATA" for item in found)


def test_regex_masks_ip_addresses(vault):
    masked, _ = mask_with_regex("Podpis z IP 31.175.12.10 i 192.168.0.1", vault)
    assert masked == "Podpis z IP [IP_1] i [IP_2]"


def test_regex_ignores_invalid_ip(vault):
    text = "Wersja 999.1.1.1"
    assert mask_with_regex(text, vault)[0] == text


def test_regex_masks_uuid_and_restores_category_with_underscore(vault):
    text = "ID dokumentu: 24150c17-720c-4d3b-974b-abb25301120e"
    masked, _ = mask_with_regex(text, vault)
    assert masked == "ID dokumentu: [DOC_ID_1]"
    # Kategoria z podkreślnikiem musi dać się przywrócić (TOKEN_BODY).
    assert vault.restore("Dokument [DOC_ID_1] podpisano.") == (
        "Dokument 24150c17-720c-4d3b-974b-abb25301120e podpisano.")


@pytest.mark.parametrize("secret", [
    "sk-live-9f8e7d6c5b4a3210",
    "AKIAIOSFODNN7EXAMPLE",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0bm93YWsifQ.abc123XYZ",
])
def test_regex_masks_api_keys(vault, secret):
    masked, _ = mask_with_regex(f"Klucz: {secret} w konfiguracji", vault)
    assert masked == "Klucz: [SECRET_1] w konfiguracji"


# --- Globalna propagacja bez względu na wielkość liter ---------------------------------


def test_same_company_in_different_case_gets_one_token(vault):
    text = ("Liveri Spółka z ograniczoną odpowiedzialnością zawiera umowę.\f"
            "KARTA PODPISÓW\nLIVERI SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ")
    entities = [Entity(original="Liveri Spółka z ograniczoną odpowiedzialnością", tag="FIRMA")]
    masked, _ = apply_entities(text, entities, vault)
    assert masked == "[FIRMA_1] zawiera umowę.\fKARTA PODPISÓW\n[FIRMA_1]"
    assert len(vault) == 1


def test_longest_entity_wins_before_shorter_one(vault):
    text = "Liveri Spółka z o.o. oraz marka LIVERI."
    entities = [Entity(original="Liveri", tag="FIRMA"),
                Entity(original="Liveri Spółka z o.o.", tag="FIRMA")]
    masked, _ = apply_entities(text, entities, vault)
    assert masked == "[FIRMA_1] oraz marka [FIRMA_2]."


def test_mask_known_propagates_to_pages_the_model_missed(monkeypatch, vault):
    # Model znajduje firmę tylko na stronie 1; strona 2 (Karta Podpisów)
    # zostaje zamaskowana przez propagację w sejfie.
    text = "Umowę zawiera Nordis Logistyka Sp. z o.o.\fPodpis: NORDIS LOGISTYKA SP. Z O.O."
    replies = iter([
        fake_ollama_reply([{"original": "Nordis Logistyka Sp. z o.o.", "tag": "FIRMA"}]),
        fake_ollama_reply([]),
    ])
    monkeypatch.setattr(llm_masker.requests, "post", lambda *a, **k: next(replies))
    result = anonymize(text, vault)
    assert result.masked_text == "Umowę zawiera [FIRMA_1]\fPodpis: [FIRMA_1]"


def test_propagation_also_covers_regex_values_in_other_case(vault):
    masked, _ = mask_with_regex("Kontakt: Jan.Kowalski@Firma.pl", vault)
    assert vault.mask_known(masked + " / JAN.KOWALSKI@FIRMA.PL") == "Kontakt: [EMAIL_1] / [EMAIL_1]"


def test_passwords_stay_case_sensitive(vault):
    first = vault.tokenize("HASLO", "Zima2025")
    second = vault.tokenize("HASLO", "zima2025")
    assert first != second
    assert vault.restore(second) == "zima2025"


def test_propagation_handles_line_breaks_inside_value(vault):
    vault.tokenize("FIRMA", "Nordis Logistyka")
    assert vault.mask_known("Firma: NORDIS\nLOGISTYKA.") == "Firma: [FIRMA_1]."


def test_chunks_never_cross_page_boundary():
    text = "strona pierwsza\fstrona druga"
    assert split_into_chunks(text) == ["strona pierwsza\n", "strona druga\n"]


def test_long_line_is_split_to_limit():
    chunks = split_into_chunks("słowo " * 500, max_chars=400)
    assert all(len(chunk) <= 401 for chunk in chunks)
    assert len(chunks) > 5


# --- Adresy -----------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("przy ul. Grunwaldzkiej 82, 80-244 Gdańsk, KRS", "przy [ADRES_1], KRS"),
    ("zam. ul. Kwiatowa 3/12, 81-301 Gdynia.", "zam. [ADRES_1]."),
    ("al. Jana Pawła II 15, 00-828 Warszawa", "[ADRES_1]"),
    ("UL. DŁUGA 15/3 31-147 KRAKÓW", "[ADRES_1]"),
    ("kod 80-244 Gdańsk", "kod [ADRES_1]"),
])
def test_regex_masks_addresses(vault, text, expected):
    assert mask_with_regex(text, vault)[0] == expected


@pytest.mark.parametrize("text", ["mieszka na ul. Polnej", "art. 12 ust. 3 pkt 4", "pl. 5 osób"])
def test_address_look_alikes_are_ignored(vault, text):
    assert mask_with_regex(text, vault)[0] == text


# --- Zamazywanie obrazów w PDF (podpisy, pieczątki, kody QR) -------------------------

SIGNATURE_RECT = pymupdf.Rect(72, 100, 222, 160)


def make_signed_pdf() -> bytes:
    """PDF z tekstem i „podpisem odręcznym” – czerwonym obrazkiem PNG."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 80), "Podpis Pracownika: netto do zaplaty", fontsize=11)
    signature = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 24), False)
    signature.set_rect(signature.irect, (200, 0, 0))  # czerwony prostokąt
    page.insert_image(SIGNATURE_RECT, pixmap=signature)
    return document.tobytes()


def red_pixel_share(pdf_bytes: bytes, rect: pymupdf.Rect) -> float:
    page = pymupdf.open(stream=pdf_bytes, filetype="pdf")[0]
    pixmap = page.get_pixmap(dpi=50, clip=rect)
    samples = pixmap.samples
    pixels = [samples[i:i + 3] for i in range(0, len(samples), pixmap.n)]
    return sum(1 for r, g, b in pixels if r > 150 and g < 80) / len(pixels)


def test_remove_images_leaves_no_images_on_page():
    page = pymupdf.open(stream=make_signed_pdf(), filetype="pdf")[0]
    assert len(page.get_images()) == 1
    assert remove_images(page) == 1
    assert len(page.get_images()) == 0
    # Tekst obok NIE może się przesunąć ani zniknąć (tekstu nie ruszamy).
    assert "netto do zaplaty" in page.get_text()


def test_redact_pdf_cuts_signature_out_of_the_file():
    original = make_signed_pdf()
    with_images = redact_pdf(original, [], redact_images=True)
    without = redact_pdf(original, [], redact_images=False)
    assert with_images.images == 1
    # W miejscu podpisu jest czarny prostokąt zamiast czerwonego obrazka.
    assert red_pixel_share(with_images.pdf, SIGNATURE_RECT) == 0
    assert red_pixel_share(without.pdf, SIGNATURE_RECT) > 0.9
    # Spłaszczona strona to dokładnie jeden obraz (cała strona), a nie
    # oryginalny obrazek podpisu dołączony osobno.
    page = pymupdf.open(stream=with_images.pdf, filetype="pdf")[0]
    assert len(page.get_images()) == 1
    # Jedyny obraz pokrywa całą stronę (z dokładnością do zaokrągleń).
    assert page.get_image_info()[0]["bbox"] == pytest.approx(tuple(page.rect), abs=0.5)


# --- Token sejfu (cały sejf jako jeden ciąg znaków) -----------------------------------


def test_vault_token_roundtrip_restores_cloud_answer(vault):
    masked, _ = mask_with_regex("PESEL 44051401359, mail jan@x.pl, ID 24150c17-720c-4d3b-974b-abb25301120e", vault)
    vault.tokenize("FIRMA", "Spółka Żółć Sp. z o.o.")
    token = vault.to_token()
    # Nowa sesja (np. po zamknięciu karty): sejf odtwarzany tylko z tokenu.
    restored_vault = Vault.from_token(token)
    answer = "Klient [FIRMA_1] ([EMAIL_1], [PESEL_1]) podpisał [DOC_ID_1]."
    assert restored_vault.restore(answer) == vault.restore(answer)
    assert restored_vault.items() == vault.items()


def test_vault_token_is_one_line_and_hides_plaintext(vault):
    vault.tokenize("OSOBA", "Jan Kowalski")
    token = vault.to_token()
    assert token.startswith("PIIVAULT1.")
    assert "\n" not in token and " " not in token
    assert "Kowalski" not in token  # dane są skompresowane, nie czytelne wprost


def test_vault_token_continues_numbering_and_reuses_tokens(vault):
    vault.tokenize("OSOBA", "Jan Kowalski")
    restored = Vault.from_token(vault.to_token())
    assert restored.tokenize("OSOBA", "JAN KOWALSKI") == "[OSOBA_1]"  # ta sama osoba
    assert restored.tokenize("OSOBA", "Anna Nowak") == "[OSOBA_2]"    # numeracja dalej


def test_vault_token_tolerates_line_breaks_from_copying(vault):
    vault.tokenize("EMAIL", "a@b.pl")
    token = vault.to_token()
    broken = token[:20] + "\n  " + token[20:]
    assert Vault.from_token(broken).restore("[EMAIL_1]") == "a@b.pl"


@pytest.mark.parametrize("bad", ["", "abc", "PIIVAULT1.xyz", "PIIVAULT2.abc.00000000"])
def test_invalid_vault_token_is_rejected(bad):
    with pytest.raises(VaultTokenError):
        Vault.from_token(bad)


def test_tampered_vault_token_is_rejected(vault):
    vault.tokenize("OSOBA", "Jan Kowalski")
    prefix, data, checksum = vault.to_token().split(".")
    tampered = f"{prefix}.{data[:-3]}AAA.{checksum}"
    with pytest.raises(VaultTokenError, match="ucięty albo zmieniony"):
        Vault.from_token(tampered)


def test_empty_vault_token_roundtrip(vault):
    assert len(Vault.from_token(vault.to_token())) == 0


def make_scanned_pdf(with_signature: bool) -> bytes:
    """Zeskanowane pismo: biały skan na całą stronę + tekst OCR (+ podpis-obrazek)."""
    document = pymupdf.open()
    page = document.new_page()
    scan = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 170), False)
    scan.set_rect(scan.irect, (250, 250, 250))  # prawie biała kartka
    page.insert_image(page.rect, pixmap=scan)
    page.insert_text((72, 80), "Oskarzyciel prywatny: Marian Kowalski", fontsize=11)
    if with_signature:
        signature = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 24), False)
        signature.set_rect(signature.irect, (200, 0, 0))
        page.insert_image(SIGNATURE_RECT, pixmap=signature)
    return document.tobytes()


def dark_share(pdf_bytes: bytes, rect: pymupdf.Rect) -> float:
    page = pymupdf.open(stream=pdf_bytes, filetype="pdf")[0]
    samples = page.get_pixmap(dpi=40, clip=rect, colorspace=pymupdf.csGRAY).samples
    return sum(1 for value in samples if value < 60) / len(samples)


def test_scanned_page_is_not_blacked_out():
    # Prawdziwy przypadek: „305 wzorów pism procesowych” – każda strona to skan
    # A4 (2480×3508 px) z warstwą OCR. Wcześniej cała strona wychodziła czarna.
    result = redact_pdf(make_scanned_pdf(with_signature=False), ["Marian Kowalski"])
    page_rect = pymupdf.open(stream=result.pdf, filetype="pdf")[0].rect
    assert dark_share(result.pdf, page_rect) < 0.05   # strona NIE jest czarna
    assert result.background_pages == 1
    assert result.images == 0
    assert result.bars == 1                           # tekst OCR nadal zaczerniony


def test_signature_on_scan_is_removed_but_scan_stays():
    page = pymupdf.open(stream=make_scanned_pdf(with_signature=True), filetype="pdf")[0]
    assert len(page.get_images()) == 2
    assert remove_images(page) == 1
    # Skan zostaje (tryb PIXELS), a podpis jest wycięty.
    assert len(page.get_images()) == 1
    result = redact_pdf(make_scanned_pdf(with_signature=True), [])
    assert red_pixel_share(result.pdf, SIGNATURE_RECT) == 0
    assert dark_share(result.pdf, pymupdf.Rect(300, 400, 500, 700)) == 0  # reszta strony jasna


def test_value_from_other_page_does_not_spray_word_bars():
    # Prawdziwy przypadek („305 wzorów pism”): adres ze strony 3 rozbity na
    # słowa zaczerniał na stronie 1 każde „ul.”, „Długa” i „Kraków”.
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 80), "Kowalski mieszka przy ul. Polnej 5 w Krakow.", fontsize=11)
    result = redact_pdf(document.tobytes(), ["ul. Dluga 11, 31-500 Krakow"], redact_images=False)
    assert result.bars == 0
    assert result.missing == ["ul. Dluga 11, 31-500 Krakow"]


def test_short_name_is_not_found_inside_longer_word():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 80), "Rozmawialem z Janem, a potem przyszedl Jan.", fontsize=11)
    words = page_words(page)
    rects = find_value_rects(words, "Jan")
    assert len(rects) == 1                      # tylko samodzielne „Jan.”
    assert rects[0].x0 > page.search_for("Janem")[0].x1


def test_address_with_word_hyphenated_across_lines(vault):
    # Prawdziwy przypadek ze skanu: „przy ul. Niezna-\nnej 3/4” – nazwa ulicy
    # przeniesiona do następnej linii. Wcześniej adres wyciekał.
    masked, _ = mask_with_regex("zamieszkałego przy ul. Niezna-\nnej 3/4, o to, że", vault)
    assert masked == "zamieszkałego przy [ADRES_1], o to, że"


def test_bars_are_visible_over_form_fields():
    # Prawdziwy przypadek: Karta Podpisów Autenti – dane podpisujących leżą
    # pod polami podpisu (widgety) z białym tłem, które zasłaniały paski.
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 100), "Jakub Klim jakub.klim3@gmail.com", fontsize=11)
    widget = pymupdf.Widget()
    widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
    widget.field_name = "podpis"
    widget.rect = pymupdf.Rect(60, 80, 400, 120)
    widget.fill_color = (1, 1, 1)  # białe tło pola
    page.add_widget(widget)
    original = document.tobytes()
    result = redact_pdf(original, ["Jakub Klim", "jakub.klim3@gmail.com"])
    assert result.missing == []
    source = pymupdf.open(stream=original, filetype="pdf")[0]
    for value in ["Jakub Klim", "jakub.klim3@gmail.com"]:
        assert dark_share(result.pdf, source.search_for(value)[0]) > 0.6


def test_amount_glued_to_unit_is_found_in_pdf():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 80), "Kara 0.80 zl/km wedlug GPS.", fontsize=11)
    result = redact_pdf(document.tobytes(), ["0.80 zl"], redact_images=False)
    assert result.missing == []
    assert result.bars == 1
