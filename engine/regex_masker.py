"""Etap 1: deterministyczne maskowanie danych o stałym formacie (Regex).

Dlaczego Regex przed LLM-em?
- E-mail, PESEL czy IBAN mają ścisły format – regex jest tu w 100%
  powtarzalny, działa w mikrosekundach i nie halucynuje.
- Mały model (3B) nie musi już szukać tych danych, więc może skupić się
  na trudniejszych rzeczach (imiona, firmy, adresy).
- Sumy kontrolne (PESEL, NIP, IBAN) odsiewają przypadkowe ciągi cyfr,
  np. numer zamówienia, który ma akurat 11 cyfr.
"""

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable

from engine.vault import Vault

# --- Walidatory sum kontrolnych -------------------------------------------


def is_valid_pesel(value: str) -> bool:
    """PESEL: 11 cyfr, ostatnia to suma kontrolna z wagami 1-3-7-9."""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 11:
        return False
    weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    total = sum(int(d) * w for d, w in zip(digits, weights))
    return (10 - total % 10) % 10 == int(digits[10])


def is_valid_nip(value: str) -> bool:
    """NIP: 10 cyfr, ostatnia = suma ważona (mod 11) pierwszych dziewięciu."""
    digits = re.sub(r"\D", "", value)
    if len(digits) != 10:
        return False
    weights = [6, 5, 7, 2, 3, 4, 5, 6, 7]
    checksum = sum(int(d) * w for d, w in zip(digits, weights)) % 11
    return checksum != 10 and checksum == int(digits[9])


def is_valid_iban(value: str) -> bool:
    """IBAN: standardowy test mod 97 (ISO 13616).

    Polski NRB (26 cyfr bez „PL”) traktujemy jak IBAN z prefiksem PL.
    """
    compact = re.sub(r"\s", "", value).upper()
    if compact.isdigit():
        compact = "PL" + compact
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
        return False
    # Przenosimy 4 pierwsze znaki na koniec, litery zamieniamy na liczby
    # (A=10, B=11, ... – int(znak, 36) robi to za nas).
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def is_valid_regon(value: str) -> bool:
    """REGON: 9 lub 14 cyfr, ostatnia = suma ważona mod 11 (10 liczy się jako 0)."""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 9:
        weights = [8, 9, 2, 3, 4, 5, 6, 7]
    elif len(digits) == 14:
        weights = [2, 4, 8, 5, 0, 9, 7, 3, 6, 1, 2, 4, 8]
    else:
        return False
    checksum = sum(int(d) * w for d, w in zip(digits, weights)) % 11
    return checksum % 10 == int(digits[-1])


def _weighted_mod10(value: str, weights: list[int]) -> bool:
    # int(znak, 36) zamienia litery na liczby: A=10, B=11, ... Z=35.
    return sum(int(ch, 36) * w for ch, w in zip(value, weights)) % 10 == 0


def is_valid_id_card(value: str) -> bool:
    """Dowód osobisty ABC523456: 4. znak to cyfra kontrolna (wagi 7-3-1-9-7-3-1-7-3).

    Sprawdzone na wzorze MSWiA „ABA300000”. Bez sumy kontrolnej każdy kod
    produktu w stylu „ABC123456” byłby uznany za dowód.
    """
    return _weighted_mod10(value, [7, 3, 1, 9, 7, 3, 1, 7, 3])


# Prefiksy polskich dokumentów księgowych (faktura VAT, przyjęcie/wydanie
# z magazynu, kasa...). „FV1234567” ma kształt paszportu i – jak każdy kod –
# w 1 na 10 przypadków przechodzi sumę kontrolną. Wykryte w testach.
ACCOUNTING_PREFIXES = {"FV", "FA", "FK", "FZ", "PZ", "WZ", "RW", "MM", "ZK", "ZM", "KP", "KW"}


def is_valid_passport(value: str) -> bool:
    """Paszport ZS0000177: 3. znak to cyfra kontrolna (wagi 7-3-9-1-7-3-1-7-3)."""
    if value[:2] in ACCOUNTING_PREFIXES:
        return False
    return _weighted_mod10(value, [7, 3, 9, 1, 7, 3, 1, 7, 3])


def is_valid_date(value: str) -> bool:
    """Odrzuca „daty” typu 31.02.2025 albo 45.13.2025."""
    numbers = [int(n) for n in re.findall(r"\d+", value)]
    if len(numbers) != 3:
        return True  # data słowna („12 marca 2025”) – sprawdzona przez regex
    year, month, day = (numbers if numbers[0] > 31 else numbers[::-1])
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


# Pierwsze litery polskich tablic to kody województw (W – mazowieckie,
# G – pomorskie, K – małopolskie...). Słowa, które wyglądają jak wyróżnik
# tablicy, ale nim nie są – np. „GET 2000” w logu HTTP albo „PLN 1000”.
PLATE_STOPWORDS = {
    "PL", "PLN", "NIP", "KRS", "BDO", "EUR", "USD", "GBP", "CHF", "VAT", "NR",
    "UL", "TEL", "FAX", "ID", "GET", "POST", "PUT", "DEL", "PDF", "OK", "ZL",
    "REGON", "PESEL", "WWW", "CEO", "RODO", "GDPR", "BOK", "SMS", "KW", "DZ",
}


def is_valid_plate(value: str) -> bool:
    prefix = value.split()[0] if " " in value else re.match(r"[A-Z]+", value).group(0)
    return prefix not in PLATE_STOPWORDS


# --- Kwoty pieniędzy --------------------------------------------------------
# Liczba: „18 500”, „4 850,00”, „1.500,00”, „1,200.50”, „1200”, „99,99”.
# Grupy tysięcy oddzielone spacją, twardą spacją, kropką lub przecinkiem.
_NUMBER = r"\d{1,3}(?:[  .,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"
# Opcjonalny mnożnik: „15 tys. zł”, „2 mln PLN”.
_MULTIPLIER = r"(?:\s?(?:tys\.?|tysięcy|mln|milion(?:ów|y)?|mld))?"
# Waluty. Dłuższe warianty przed krótszymi („złotych” przed „zł”).
_CURRENCY = r"(?:złotych|złote|złoty|zł|zl|PLN|EUR|euro|€|USD|\$|GBP|£|CHF)"

# Dwa szyki: „18 500 zł” (waluta po liczbie) i „$1,200” / „PLN 500” (przed).
# \s między liczbą a walutą obejmuje też nową linię – w PDF „250” i „zł”
# bywają w osobnych liniach, a model 3B takie kwoty gubił.
AMOUNT_PATTERN = re.compile(
    rf"(?<![\w.,])(?:{_NUMBER}){_MULTIPLIER}\s?{_CURRENCY}(?!\w)"
    rf"|(?<!\w){_CURRENCY}\s?(?:{_NUMBER}){_MULTIPLIER}(?![\w,.]\d)"
)


# --- Definicje wzorców ------------------------------------------------------


@dataclass(frozen=True)
class PiiPattern:
    category: str                    # nazwa kategorii -> token [KATEGORIA_n]
    regex: re.Pattern
    validator: Callable[[str], bool] | None = None
    # Która grupa regexa to wrażliwa wartość. 0 = całe dopasowanie.
    # Dla haseł maskujemy tylko wartość, a słowo „password=” zostawiamy,
    # żeby zewnętrzne AI nadal rozumiało kontekst logu.
    value_group: int = 0


# Adres: „ul. Grunwaldzkiej 82”, „al. Jana Pawła II 15/3”, „os. Lecha 12A”.
# Słowo kluczowe bez względu na wielkość liter (UL. na Karcie Podpisów),
# potem 1–4 słowa nazwy i numer budynku z opcjonalnym numerem lokalu.
_UPPER = "A-ZĄĆĘŁŃÓŚŹŻ"
_STREET = (
    r"(?<!\w)(?i:ul\.|ulica|al\.|aleja|pl\.|plac|os\.|osiedle)\s+"
    # „-\s+małe_litery” = wyraz przeniesiony do nowej linii („Niezna-\nnej”).
    rf"(?:[{_UPPER}0-9][\w.'-]*(?:-\s+[a-ząćęłńóśźż]+)?\s+){{1,4}}?"
    r"\d+[A-Za-z]?(?:\s?/\s?\d+[A-Za-z]?)?(?![\w/])"
)
# Kod pocztowy z miastem: „80-244 Gdańsk”, „31-147 Kraków”, „00-514 WARSZAWA”.
_POSTAL_CITY = (
    rf"(?<![\d-])\d{{2}}-\d{{3}}(?![\d-])\s+[{_UPPER}][\w]+"
    rf"(?:[ -][{_UPPER}][\w]+)?"
)

_MONTHS = (r"stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|"
           r"września|października|listopada|grudnia")

# KOLEJNOŚĆ MA ZNACZENIE: wzorce są stosowane po kolei, a już podmienione
# fragmenty stają się tokenami, których kolejne wzorce nie dotkną.
# Ogólna zasada: najpierw rzeczy długie i jednoznaczne (klucze, UUID, IBAN),
# potem krótkie ciągi cyfr, które łatwo pomylić (telefon na samym końcu).
PATTERNS: list[PiiPattern] = [
    # --- Sekrety -----------------------------------------------------------
    # Tokeny JWT (eyJ... – zakodowany nagłówek {"alg":...}), klucze OpenAI
    # (sk-...) i AWS (AKIA...). Rozpoznawalne po samym kształcie.
    PiiPattern(
        "SECRET",
        re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}"),
    ),
    PiiPattern("SECRET", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    PiiPattern("SECRET", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # Token Bearer przed hasłami – inaczej „token: Bearer abc” zamaskowałoby
    # słowo „Bearer”, a prawdziwy sekret by wyciekł.
    PiiPattern(
        "SECRET",
        re.compile(r"\bBearer\s+(?!\[)([A-Za-z0-9._~+/=-]{8,})"),
        value_group=1,
    ),
    # Hasło po słowie kluczowym: password=..., "hasło": "...", api_key=...
    PiiPattern(
        "HASLO",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|hasło|haslo|secret|token|api[_-]?key)"
            r"\b[\"']?\s*[:=]\s*[\"']?(?!Bearer\b)(?!\[)([^\s\"',;}]+)"
        ),
        value_group=1,
    ),
    # Hasło zaszyte w adresie URL: postgres://admin:S3cr3t@host
    PiiPattern("HASLO", re.compile(r"://[^:/\s@]+:([^@/\s\[]+)@"), value_group=1),
    PiiPattern(
        "EMAIL",
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ),
    # --- Identyfikatory dokumentów i podpisów ------------------------------------
    # UUID, np. identyfikator dokumentu w Autenti. Przed datami i liczbami,
    # bo jego fragmenty („2025”, „1234”) mogłyby zostać pocięte innymi wzorcami.
    PiiPattern(
        "DOC_ID",
        re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                   r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    ),
    # --- Finanse ------------------------------------------------------------------
    PiiPattern(
        "IBAN",
        # PL + 26 cyfr albo sam NRB (26 cyfr), z opcjonalnymi spacjami co 4.
        re.compile(r"\b(?:PL ?)?\d{2}(?: ?\d{4}){6}\b"),
        validator=is_valid_iban,
    ),
    # --- Rejestry firmowe --------------------------------------------------
    # KRS nie ma sumy kontrolnej, a 10 cyfr ma też NIP. Dlatego KRS łapiemy
    # tylko po słowie „KRS” albo po typowym początku „0000” (numery KRS są
    # dopełniane zerami; NIP nigdy tak się nie zaczyna).
    PiiPattern("KRS", re.compile(r"\bKRS\b\W{0,6}(\d{10})(?!\d)"), value_group=1),
    PiiPattern("KRS", re.compile(r"(?<!\d)0000\d{6}(?!\d)")),
    # REGON 9-cyfrowy wygląda jak numer telefonu, więc bez słowa „REGON”
    # akceptujemy tylko wersję 14-cyfrową (z poprawną sumą kontrolną).
    PiiPattern(
        "REGON",
        re.compile(r"\bREGON\b\W{0,6}(\d{14}|\d{9})(?!\d)"),
        validator=is_valid_regon,
        value_group=1,
    ),
    PiiPattern("REGON", re.compile(r"(?<!\d)\d{14}(?!\d)"), validator=is_valid_regon),
    # BDO (rejestr odpadowy) – 9 cyfr bez sumy kontrolnej, więc tylko po słowie BDO.
    PiiPattern("BDO", re.compile(r"\bBDO\b\W{0,6}(\d{9})(?!\d)"), value_group=1),
    PiiPattern(
        "PESEL",
        re.compile(r"(?<!\d)\d{11}(?!\d)"),
        validator=is_valid_pesel,
    ),
    PiiPattern(
        "NIP",
        re.compile(
            r"(?<!\d)(?:PL ?)?"
            r"(?:\d{3}-\d{3}-\d{2}-\d{2}|\d{3}-\d{2}-\d{2}-\d{3}|\d{10})(?!\d)"
        ),
        validator=is_valid_nip,
    ),
    # --- Dokumenty tożsamości i pojazdy ---------------------------------------------
    # VIN: 17 znaków bez I, O, Q (żeby nie myliły się z 1 i 0).
    # Wymagamy co najmniej jednej litery i jednej cyfry.
    PiiPattern(
        "VIN",
        re.compile(r"(?<![A-Za-z0-9])(?=[A-HJ-NPR-Z0-9]*\d)(?=[A-HJ-NPR-Z0-9]*[A-HJ-NPR-Z])"
                   r"[A-HJ-NPR-Z0-9]{17}(?![A-Za-z0-9])"),
    ),
    # --- Daty -----------------------------------------------------------------------
    PiiPattern(
        "DATA",
        re.compile(
            r"(?<![\d.])(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])\.(?:19|20)\d{2}(?!\d)"
            r"|(?<![\d/])(?:0?[1-9]|[12]\d|3[01])/(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}(?!\d)"
            r"|(?<![\d-])(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d)"
            rf"|(?<!\d)(?:0?[1-9]|[12]\d|3[01]) (?:{_MONTHS}) (?:19|20)\d{{2}}(?!\d)"
        ),
        validator=is_valid_date,
    ),
    # --- Adresy -------------------------------------------------------------------
    # Ulica z numerem (opcjonalnie z kodem i miastem) oraz sam kod z miastem.
    # Model 3B regularnie gubił adresy w umowach – a ich format jest stały.
    PiiPattern("ADRES", re.compile(rf"{_STREET}(?:,?\s+{_POSTAL_CITY})?")),
    PiiPattern("ADRES", re.compile(_POSTAL_CITY)),
    # Kwoty PRZED telefonem: inaczej „150 000 000 zł” wyglądałoby
    # dla regexa telefonu jak numer 150-000-000.
    PiiPattern("KWOTA", AMOUNT_PATTERN),
    # Dowód i paszport PO kwotach: „PLN 123456” nie może zostać dowodem.
    PiiPattern(
        "DOWOD",
        re.compile(r"(?<![A-Za-z0-9])[A-Z]{3} ?\d{6}(?![A-Za-z0-9])"),
        validator=lambda v: is_valid_id_card(v.replace(" ", "")),
    ),
    PiiPattern(
        "PASZPORT",
        re.compile(r"(?<![A-Za-z0-9])[A-Z]{2} ?\d{7}(?![A-Za-z0-9])"),
        validator=lambda v: is_valid_passport(v.replace(" ", "")),
    ),
    PiiPattern(
        "IP",
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
                   r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    ),
    # Tablice rejestracyjne: GD 1234A, WI 98765, KR1AB23. Pierwsza litera to
    # kod województwa, potem 4–5 znaków z co najmniej jedną cyfrą.
    PiiPattern(
        "REJESTRACJA",
        re.compile(r"(?<![\w-])[BCDEFGKLNOPRSTWZ][A-Z]{1,2} ?"
                   r"(?=[A-Z0-9]{0,4}\d)[A-Z0-9]{4,5}(?![\w-])"),
        validator=is_valid_plate,
    ),
    PiiPattern(
        "TEL",
        # Komórka: (+48) 600 100 200 / 600-100-200 / 600100200
        # Stacjonarny: (22) 123 45 67
        re.compile(
            r"(?<![\w+])(?:(?:\+|00)48[ -]?)?"
            r"(?:\d{3}[ -]?\d{3}[ -]?\d{3}|\(?\d{2}\)?[ -]?\d{3}[ -]?\d{2}[ -]?\d{2})"
            r"(?!\w)"
        ),
    ),
]


def _make_replacer(pattern: PiiPattern, vault: Vault, found: list[dict]):
    """Tworzy funkcję dla re.sub, która podmienia dopasowanie na token."""

    def replace(match: re.Match) -> str:
        value = match.group(pattern.value_group)
        # Walidator odrzuca „fałszywe alarmy” – zwracamy tekst bez zmian.
        if pattern.validator and not pattern.validator(value):
            return match.group(0)

        token = vault.tokenize(pattern.category, value)
        found.append({
            "token": token,
            "kategoria": pattern.category,
            "oryginał": value,
            "źródło": "regex",
        })

        if pattern.value_group == 0:
            return token
        # Podmieniamy tylko grupę z wartością, resztę dopasowania zostawiamy.
        whole = match.group(0)
        start = match.start(pattern.value_group) - match.start()
        end = match.end(pattern.value_group) - match.start()
        return whole[:start] + token + whole[end:]

    return replace


def mask_with_regex(text: str, vault: Vault) -> tuple[str, list[dict]]:
    """Maskuje dane o stałym formacie. Zwraca (tekst_z_tokenami, wykryte)."""
    found: list[dict] = []
    for pattern in PATTERNS:
        text = pattern.regex.sub(_make_replacer(pattern, vault, found), text)
    return text, found
