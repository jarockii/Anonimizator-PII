"""Reguły ze „ścieżki 1”: dane, które w przyklady/5–8 przeszły przez NER i LLM.

Każdy przypadek pozytywny pochodzi z prawdziwego wycieku w pomiarze, każdy
negatywny pilnuje, żeby reguła nie maskowała zwykłego tekstu.
"""

import pytest

from engine.numerals import find_spelled_numbers
from engine.regex_masker import mask_with_regex
from engine.vault import Vault


def mask(text: str) -> str:
    vault = Vault()
    masked, _ = mask_with_regex(text, vault)
    assert vault.restore(masked) == text  # każda reguła jest odwracalna co do znaku
    return masked


@pytest.mark.parametrize("text, expected", [
    # E-mail „zamaskowany” przez autora
    ("mietek.zuraw (at) wp kropka pl", "[EMAIL_1]"),
    ("pisz: jan[małpa]onet.pl!", "pisz: [EMAIL_1]!"),
    ("jan at firma dot com", "[EMAIL_1]"),
    # Profile i loginy
    ("linkedin.com/in/zbigniew-lukaszewicz-0b12a |", "[PROFIL_1] |"),
    ("https://www.facebook.com/profile.php?id=1000123", "[PROFIL_1]"),
    ("skype: zbych.lukaszewicz", "skype: [LOGIN_1]"),
    ('{"username": "tnowak", "remember": true}', '{"username": "[LOGIN_1]", "remember": true}'),
    # Zagraniczny IBAN, księga wieczysta, działka, paszport po słowie kluczowym
    ("IBAN DE89 3704 0044 0532 0130 00.", "IBAN [IBAN_1]."),
    ("księgę wieczystą nr SW1K/00054321/7.", "księgę wieczystą nr [KW_1]."),
    ("działkę gruntu nr 345/7, obręb 0012", "działkę gruntu nr [DZIALKA_1], obręb 0012"),
    ("paszportem FE\n482913, zamieszkały", "paszportem [PASZPORT_1], zamieszkały"),
    ("paszport ukraiński nr FE 482913", "paszport ukraiński nr [PASZPORT_1]"),
    # Daty po angielsku
    ("(born May 14th, 1961)", "(born [DATA_1])"),
    ("On 3 Mar 2025, at 09:14", "On [DATA_1], at 09:14"),
    ("the 3rd of March 2025", "the [DATA_1]"),
    # Adresy z odmienionym słowem kluczowym, lokalem, adres wiejski
    ("przy Placu Chrobrego 7 m. 4.", "przy [ADRES_1]."),
    ("na ulicy Długiej 5 lok. 2", "na [ADRES_1]"),
    ("zamieszkały: Wola\nZaręczycka 112, 37-415 Wola Zaręczycka,", "zamieszkały: [ADRES_1],"),
    ("mieszka na Długiej 5 m. 12, koło biedronki", "mieszka na [ADRES_1], koło biedronki"),
    ("sklep przy Polnej 7", "sklep przy [ADRES_1]"),
])
def test_new_rules_mask(text, expected):
    assert mask(text) == expected


@pytest.mark.parametrize("text", [
    "art. 157 Kodeksu cywilnego",        # „Kodeksu 157” to nie adres
    "you may 14 times",                  # „may” małą literą to nie miesiąc
    "Login failed for user",             # „user” bez dwukropka
    "spotkanie at 10:00 w sali",         # „at” bez domeny
    "obręb 0012 Stary Bielsk",           # sam obręb zostaje
    "Strona 1, 00-950 dodatkowo",        # kod pocztowy bez miasta wielką literą
    "jest na Stronie 5 raportu",         # „na” + rzeczownik, nie ulica (-ej)
])
def test_new_rules_leave_normal_text(text):
    assert mask(text) == text


# --- Liczby zapisane słownie -----------------------------------------------------------


@pytest.mark.parametrize("text, category", [
    ("sześćset dwa, trzysta czterdzieści jeden, dwieście dwanaście", "TEL"),
    ("zero zero czterdzieści osiem, sześćset jeden, dwieście trzydzieści cztery, "
     "pięćset sześćdziesiąt siedem", "TEL"),
    ("czterystu dwunastu tysięcy złotych", "KWOTA"),
    ("tysiąc dwieście złotych i pięćdziesiąt groszy", "KWOTA"),
    ("dziewięć dwa zero trzy jeden pięć zero dwa trzy cztery dziewięć", "PESEL"),
    ("dwudziestego czwartego lutego roku dwa tysiące dwudziestego piątego", "DATA"),
])
def test_spelled_numbers_are_classified(text, category):
    found = find_spelled_numbers(text)
    assert [item.category for item in found] == [category]
    assert (found[0].start, found[0].end) == (0, len(text))


def test_spelled_values():
    assert find_spelled_numbers("sześćset dwa, trzysta czterdzieści jeden, dwieście dwanaście")[0].digits == "602341212"
    assert find_spelled_numbers("czterystu dwunastu tysięcy złotych")[0].digits == "412000"


@pytest.mark.parametrize("text", [
    "jeden z pracowników", "trzy miesiące", "drugie piętro", "sto osób",
    "dwa tysiące dwudziestego piątego roku",  # sam rok to nie data osoby
    "dziewięć dwa zero trzy jeden pięć zero dwa trzy cztery osiem",  # zła suma PESEL
])
def test_ordinary_numerals_are_left_alone(text):
    assert find_spelled_numbers(text) == []


def test_spelled_numbers_restore_exactly():
    text = "Cena: czterystu dwunastu tysięcy złotych. Tel. sześćset dwa, trzysta czterdzieści jeden, dwieście dwanaście."
    assert mask(text) == "Cena: [KWOTA_1]. Tel. [TEL_1]."
