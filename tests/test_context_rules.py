"""Reguły kontekstowe (engine/context_rules.py) i numery kredytów/polis.

Przykłady są celowo INNE niż w zestawie kontrolnym (eval/oznaczenia_kontrolne) –
reguły mają działać na klasę przypadków, a nie na konkretne słowa z tamtych plików.
"""

import pytest

from engine.llm_masker import described_in_context
from engine.regex_masker import mask_with_regex
from engine.vault import Vault


def mask(text: str) -> str:
    return mask_with_regex(text, Vault())[0]


@pytest.mark.parametrize("text, expected", [
    ("Wynagrodzenie – BUD-MAX\nRZESZÓW SP. Z O.O., NIP", "Wynagrodzenie – [FIRMA_1], NIP"),
    ("Prezes Zarządu Nordis Logistyka Sp. z o.o. podpisał", "Prezes Zarządu [FIRMA_1] podpisał"),
    ("Fakturę wystawiła Optiprint S.A. w terminie", "Fakturę wystawiła [FIRMA_1] w terminie"),
    ("umowa z Kowalczyk i Wspólnicy spółka komandytowa", "umowa z [FIRMA_1]"),
    ("Dostawca: P&K Serwis sp.j.", "Dostawca: [FIRMA_1]"),
    # Tabela PDF: nazwa złamana w komórce, liczby z sąsiednich kolumn pomiędzy.
    ("Zapłata – ELEKTRO-SERWIS        2       5\n      KIELCE S.A.  ok",
     "Zapłata – [FIRMA_1]  ok"),
])
def test_company_before_legal_form(text, expected):
    assert mask(text) == expected


@pytest.mark.parametrize("text", [
    "Spółka z o.o. ma zarząd.",            # sama forma prawna, bez nazwy
    "Umowa sp. z o.o. – wzór",            # słowo-funkcja przed formą
])
def test_legal_form_alone_is_not_a_company(text):
    assert "[FIRMA" not in mask(text)


@pytest.mark.parametrize("text, expected", [
    ("spłata pożyczki gotówkowej nr PG/2023/118204.", "spłata pożyczki gotówkowej nr [DOC_ID_1]."),
    ("szkoda z polisy nr PZ-4471-2025", "szkoda z polisy nr [DOC_ID_1]"),
    ("numer klienta nr 004829135", "numer klienta nr [DOC_ID_1]"),
])
def test_loan_and_policy_numbers(text, expected):
    assert mask(text) == expected


@pytest.mark.parametrize("text", [
    "Wyciąg z rachunku nr 3/2026",          # nie słowo kluczowe kredytu/polisy
    "umowa nr 17/2025 z dnia",
    "kredyt nr 12",                         # za krótki
])
def test_ordinary_numbers_are_not_doc_ids(text):
    assert "[DOC_ID" not in mask(text)


@pytest.mark.parametrize("text, names", [
    ("Bartek i Gosia przyszli", ["Bartek", "Gosia"]),
    ("a Zbyszka nie było, daj Kasi znać", ["Zbyszka", "Kasi"]),
    ("Krzysiu, oddzwoń", ["Krzysiu"]),
    ("Pozdrów Antosia i Olę", ["Antosia", "Olę"]),
])
def test_name_diminutives(text, names):
    masked = mask(text)
    assert all(name not in masked for name in names) and "[OSOBA_" in masked


@pytest.mark.parametrize("text", [
    "Kuba to wyspa na Karaibach.",           # dwuznaczne – pominięte
    "bartek",                                 # małą literą to nie imię
    "Bartkowiak podpisał umowę.",             # dłuższe słowo, nie zdrobnienie
])
def test_diminutives_do_not_hit_other_words(text):
    assert "[OSOBA_" not in mask(text)


@pytest.mark.parametrize("text, expected", [
    ("To jedyna kobieta w dziale.", "To [POSREDNI_1] w dziale."),
    ("Jest jedynym tłumaczem w gminie.", "Jest [POSREDNI_1] w gminie."),
])
def test_unique_description(text, expected):
    assert mask(text) == expected


@pytest.mark.parametrize("text", [
    "Jedyny sposób to czekać.", "To jedyna szansa.", "Jedynie we wtorek.", "jedyny raz",
])
def test_unique_non_person_phrases_stay(text):
    assert mask(text) == text


@pytest.mark.parametrize("text", [
    "orzeczenie o umiarkowanym stopniu niepełnosprawności",
    "Stopień niepełnosprawności: lekki",
    "symbol przyczyny 10-N",
])
def test_disability_degree_and_symbol(text):
    assert "[ZDROWIE_" in mask(text)


def test_disability_symbol_needs_exact_code():
    assert "[ZDROWIE_" not in mask("pokój 05-B, faktura 10-X")


def test_llm_description_accepted_with_lead_in_same_sentence():
    text = "Pani z trójki, ta co prowadzi zielony warzywniak, zgłosiła awarię. Dostawa wózka w piątek."
    assert described_in_context(text, "zielony warzywniak")
    assert not described_in_context(text, "Dostawa wózka")      # inne zdanie
    assert not described_in_context(text, "warzywniak")          # jedno słowo to za mało


@pytest.mark.parametrize("text, expected", [
    ("Pani z dwójki, ta co sprząta na trzecim piętrze, zgłosiła awarię.",
     "Pani z dwójki, ta co [POSREDNI_1], zgłosiła awarię."),
    ("to ten co sprzedaje miód na targu", "to ten co [POSREDNI_1]"),
])
def test_colloquial_description_of_a_person(text, expected):
    assert mask(text) == expected


@pytest.mark.parametrize("text", [
    "Umowa ta, która została zawarta w dniu podpisania.",   # urzędowe „która”
    "To jest to co zawsze.",                                  # „to co” – nie osoba
])
def test_formal_relative_clause_is_not_a_description(text):
    assert "[POSREDNI" not in mask(text)


def test_fuzzy_span_finds_phrase_the_model_reworded():
    from engine.llm_masker import fuzzy_span
    text = "Rozpoznanie: ostre zapalenie trzustki (K85.0), dr [OSOBA_1]."
    assert fuzzy_span(text, "choruje na ostrym zapaleniu trzustki") == "ostre zapalenie trzustki"
    assert fuzzy_span(text, "lek. [OSOBA_1] leczył") is None          # token to nie słowo
    assert fuzzy_span(text, "złamanie kości udowej") is None


def test_fuzzy_span_single_sensitive_word_around_tokens():
    from engine.llm_masker import fuzzy_span
    text = "Poszkodowana jest [STANOWISKO_1] „Solidarności” od lat, mieszka w Radomiu."
    assert fuzzy_span(text, "jest [OSOBA_1] Solidarność") == "Solidarności"
    assert fuzzy_span(text, "mieszka [OSOBA_2] Radomiu") is None    # zwykłe słowo – nie samo
