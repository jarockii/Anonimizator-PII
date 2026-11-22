"""Etap 1: deterministyczne maskowanie danych o stałym formacie (Regex).

Dlaczego Regex przed LLM-em?
- E-mail, PESEL czy IBAN mają ścisły format – regex jest tu w 100%
  powtarzalny, działa w mikrosekundach i nie halucynuje.
- Mały model (3B) nie musi już szukać tych danych, więc może skupić się
  na trudniejszych rzeczach (imiona, firmy, adresy).
- Sumy kontrolne (PESEL, NIP, IBAN) odsiewają przypadkowe ciągi cyfr,
  np. numer zamówienia, który ma akurat 11 cyfr.
"""

import ipaddress
import re
from dataclasses import dataclass
from datetime import date
from typing import Callable

from engine.context_rules import find_context_matches
from engine.numerals import find_spelled_numbers
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


def _with_separators(validator: Callable[[str], bool], max_separators: int):
    """Walidator, który dopuszcza najwyżej max_separators spacji/myślników
    w numerze (nie licząc spacji po prefiksie kraju „PL ”)."""

    def check(value: str) -> bool:
        number = re.sub(r"^PL ?", "", value)
        return len(re.findall(r"[ -]", number)) <= max_separators and validator(value)

    return check


def is_valid_card(value: str) -> bool:
    """Numer karty płatniczej: 13–19 cyfr i suma kontrolna Luhna."""
    digits = re.sub(r"\D", "", value)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for position, char in enumerate(reversed(digits)):
        number = int(char)
        if position % 2:  # co druga cyfra od końca jest podwajana
            number = number * 2 - 9 if number > 4 else number * 2
        total += number
    return total % 10 == 0


def is_valid_ipv6(value: str) -> bool:
    """Prawdziwy adres IPv6 (odrzuca godziny „10:41:07” i samo „::”)."""
    if not re.search(r"[0-9A-Fa-f]", value):
        return False
    try:
        ipaddress.IPv6Address(value)
    except ValueError:
        return False
    return True


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
# Numer budynku z opcjonalnym lokalem: 15/3, 12A, „7 m. 4”, „5 lok. 2”.
_BUILDING = r"\d+[A-Za-z]?(?:\s?/\s?\d+[A-Za-z]?|\s+(?:m\.|lok\.)\s?\d+[A-Za-z]?)?(?![\w/])"
_STREET = (
    # Słowo kluczowe także odmienione: „przy Placu Chrobrego”, „na ulicy”, „w alei”.
    r"(?<!\w)(?i:ul\.|ulica|ulicy|ulicę|al\.|aleja|alei|aleję|pl\.|plac|placu|placem"
    r"|os\.|osiedle|osiedlu|osiedla|rondo|ronda|rondzie)\s+"
    # „-\s+małe_litery” = wyraz przeniesiony do nowej linii („Niezna-\nnej”).
    rf"(?:[{_UPPER}0-9][\w.'-]*(?:-\s+[a-ząćęłńóśźż]+)?\s+){{1,4}}?"
    + _BUILDING
)
# Adres potoczny bez „ul.”: „mieszka na Długiej 5 m. 12”, „przy Polnej 7”.
# Nazwa ulicy w miejscowniku przymiotnikowym (-ej) + numer: „na Stronie 5”
# czy „przy Kasie 3” tego nie spełniają.
_COLLOQUIAL_STREET = rf"(?<!\w)(?:na|przy)\s+([{_UPPER}][a-ząćęłńóśźż]+ej\s+{_BUILDING})"
# Adres wiejski bez „ul.”: „Wola Zaręczycka 112, 37-415 Wola Zaręczycka”.
# Wymagamy przecinka i kodu pocztowego po numerze – bez nich „Kodeks 157”
# też wyglądałby jak adres.
_VILLAGE = rf"(?<![\w/])(?:[{_UPPER}][\w-]*\s+){{1,3}}{_BUILDING},\s*"
# Kod pocztowy z miastem: „80-244 Gdańsk”, „31-147 Kraków”, „00-514 WARSZAWA”.
_POSTAL_CITY = (
    rf"(?<![\d-])\d{{2}}-\d{{3}}(?![\d-])\s+[{_UPPER}][\w]+"
    rf"(?:[ -][{_UPPER}][\w]+)?"
)

_MONTHS = (r"stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|"
           r"września|października|listopada|grudnia")
# Miesiące po angielsku – z wielkiej litery, żeby „may” (czasownik) nie był datą.
_EN_MONTHS = (r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
              r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?")

# „Zamaskowane” @ i kropka w e-mailach: (at) [at] {at} (małpa) [małpa] ` at ` ` małpa `.
_OBF_AT = r"(?:\((?:at|małpa)\)|\[(?:at|małpa)\]|\{at\}|\s(?:at|małpa)\s)"
_OBF_DOT = r"(?:\.|\((?:dot|kropka)\)|\[(?:dot|kropka)\]|\s(?:dot|kropka)\s)"

# KOLEJNOŚĆ MA ZNACZENIE: wzorce są stosowane po kolei, a już podmienione
# fragmenty stają się tokenami, których kolejne wzorce nie dotkną.
# Ogólna zasada: najpierw rzeczy długie i jednoznaczne (klucze, UUID, IBAN),
# potem krótkie ciągi cyfr, które łatwo pomylić (telefon na samym końcu).
PATTERNS: list[PiiPattern] = [
    # --- Sekrety -----------------------------------------------------------
    # Klucz prywatny PEM – cały blok, razem z liniami BEGIN/END. Pierwszy,
    # bo jego treść (base64) pocięłyby inne wzorce i część klucza by wyciekła.
    PiiPattern(
        "SECRET",
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?"
                   r"-----END [A-Z0-9 ]*PRIVATE KEY-----"),
    ),
    # Tokeny GitHuba (ghp_, gho_, ghu_, ghs_, ghr_, github_pat_), Slacka
    # (xoxb-, xoxp-…), Google API (AIza…) i Stripe (sk_live_, rk_test_…).
    # Przed NIP-em i telefonem – inaczej cyfry w środku tokenu stawały się
    # [NIP_1], a reszta sekretu wyciekała.
    PiiPattern("SECRET", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{40,})")),
    PiiPattern("SECRET", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    PiiPattern("SECRET", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    PiiPattern("SECRET", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}")),
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
    # E-mail „zamaskowany” przez autora: jan.kowalski (at) wp kropka pl,
    # jan[małpa]onet.pl, jan at firma dot com. Ludzie tak piszą na forach
    # i w czatach – NER widzi tylko „jan.kowalski”, a domena zostaje.
    PiiPattern("EMAIL", re.compile(
        rf"(?i)(?<![\w.+-])[\w.+-]+\s*{_OBF_AT}\s*[\w-]+"
        rf"(?:\s*{_OBF_DOT}\s*[\w-]+)*\s*{_OBF_DOT}\s*[a-z]{{2,}}\b"
    )),
    # Adres profilu z nazwiskiem w ścieżce: linkedin.com/in/jan-kowalski-0b12a.
    # NER nie widzi w tym osoby, a to jednoznaczny identyfikator.
    PiiPattern("PROFIL", re.compile(
        r"(?i)\b(?:https?://)?(?:[a-z]{2,3}\.)?"
        r"(?:linkedin\.com/(?:in|pub)|facebook\.com|fb\.com|instagram\.com|twitter\.com|x\.com"
        r"|github\.com|gitlab\.com|tiktok\.com|youtube\.com|t\.me|goldenline\.pl)"
        r"/@?[\w.%-]+(?:\?id=\d+)?/?"
    )),
    # Login po słowie kluczowym: skype: zbych.lukaszewicz, "username": "tnowak",
    # login=jkowal. Maskujemy tylko wartość – słowo kluczowe zostaje (kontekst).
    PiiPattern(
        "LOGIN",
        re.compile(
            r"(?i)\b(?:skype|telegram|signal|discord|gg|teams|login|user(?:name)?|użytkownik)"
            r"[\"']?\s*[:=]\s*[\"']?@?(?!\[)([\w.-]{3,})"
        ),
        value_group=1,
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
        # PL + 26 cyfr albo sam NRB (26 cyfr). Spacje w dowolnych miejscach,
        # nie tylko co 4 – OCR skanu potrafi je przesunąć („1090 10147879”).
        re.compile(r"\b(?:PL ?)?\d{2}(?: ?\d){24}\b"),
        validator=_with_separators(is_valid_iban, 8),
    ),
    # Zagraniczny IBAN (DE89 3704 0044 0532 0130 00, GB…, FR…). Ta sama suma
    # mod 97. Wcześniej NER oznaczał niemiecki IBAN jako… FIRMA.
    PiiPattern(
        "IBAN",
        re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"),
        validator=is_valid_iban,
    ),
    # --- Nieruchomości i dokumenty po słowie kluczowym --------------------------------
    # Księga wieczysta: kod sądu (4 znaki, 3. to cyfra) / 8 cyfr / cyfra kontrolna,
    # np. SW1K/00054321/7. Format jest na tyle charakterystyczny, że nie wymagamy
    # sumy kontrolnej – lepiej zamaskować KW z literówką niż ją przepuścić.
    PiiPattern("KW", re.compile(r"\b[A-Z]{2}\d[A-Z]\s?/\s?\d{8}\s?/\s?\d\b")),
    # Numer działki ewidencyjnej po słowie „działka” (345/7, 1024).
    PiiPattern(
        "DZIALKA",
        re.compile(r"(?i)\bdziałk\w*\s+(?:gruntu\s+)?(?:ewidencyjn\w+\s+)?"
                   r"(?:nr|numer|o\s+numerze)\.?\s*(\d+(?:/\d+)?)(?![\d/])"),
        value_group=1,
    ),
    # Paszport po słowie kluczowym – dowolny kraj (ukraiński FE 482913 ma
    # 6 cyfr i nie pasuje do wzorca polskiego). Spacja może być złamaniem linii.
    PiiPattern(
        "PASZPORT",
        re.compile(r"(?i)\bpaszport\w*\s+(?:\w+\s+)?(?:nr\.?\s*|numer\s+|seria\s+i\s+numer\s+)?"
                   r"(?-i:([A-Z]{1,3}\s?\d{5,9}))\b"),
        value_group=1,
    ),
    # Numer kredytu, pożyczki, polisy, leasingu, klienta po słowie kluczowym:
    # „kredytu hipotecznego nr KH/2019/00731”, „polisy nr PZ-4471-2025”. Taki
    # numer wskazuje jedną osobę w banku/u ubezpieczyciela. Wymagamy cyfry
    # i ≥ 6 znaków, a słowo kluczowe musi stać najwyżej 2 słowa przed „nr”.
    PiiPattern(
        "DOC_ID",
        re.compile(r"(?i)\b(?:kredyt\w*|pożyczk\w*|polis\w*|leasing\w*|klient\w*|ubezpieczeni\w*"
                   r"|szkod\w*|hipote\w*)(?:\s+[^\W\d_]+){0,2}?\s+(?:nr|numer)\.?\s*"
                   r"(?-i:((?=[A-Z0-9/.-]*\d)[A-Z0-9][A-Z0-9/.-]{4,}[A-Z0-9]))(?![\w/-])"),
        value_group=1,
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
    # Karta płatnicza: 13–19 cyfr (opcjonalnie w grupach), suma Luhna.
    # (?![ -]?\d) – nie wolno uciąć dłuższego ciągu cyfr (np. rachunku).
    # Po REGON-ie 14-cyfrowym, przed PESEL-em, NIP-em i telefonem.
    PiiPattern(
        "KARTA",
        re.compile(r"(?<![\d-])[2-6]\d{3}(?:[ -]?\d){9,15}(?![ -]?\d)"),
        validator=is_valid_card,
    ),
    PiiPattern(
        "PESEL",
        # Pojedyncze myślniki lub spacje w środku też: „9203150-2349” (SOFT
        # HYPHEN wstawiony, żeby ominąć wzorzec) albo „561 10446533” (OCR
        # skanu wstawił spację). O wyniku decyduje suma kontrolna, a limit
        # separatorów nie pozwala skleić kilku osobnych liczb w jeden PESEL.
        re.compile(r"(?<![\d-])\d(?:[ -]?\d){10}(?![\d-])"),
        validator=_with_separators(is_valid_pesel, 2),
    ),
    PiiPattern(
        "NIP",
        # 526-025-02-74, 526-02-50-274, 5260250274, „526 025 02 74” i to samo
        # z przesuniętą przez OCR spacją. Najwyżej 3 separatory.
        re.compile(r"(?<![\d-])(?:PL ?)?\d(?:[ -]?\d){9}(?![\d-])"),
        validator=_with_separators(is_valid_nip, 3),
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
            # Po angielsku: May 14th, 1961 / March 3, 2025 / 14 May 1961 / 3rd of Mar 2025
            rf"|\b(?:{_EN_MONTHS})\.?\s+(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}\b"
            rf"|\b(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_EN_MONTHS})\.?,?\s+"
            rf"(?:19|20)\d{{2}}\b"
        ),
        validator=is_valid_date,
    ),
    # --- Adresy -------------------------------------------------------------------
    # Ulica z numerem (opcjonalnie z kodem i miastem) oraz sam kod z miastem.
    # Model 3B regularnie gubił adresy w umowach – a ich format jest stały.
    PiiPattern("ADRES", re.compile(rf"{_STREET}(?:,?\s+{_POSTAL_CITY})?")),
    PiiPattern("ADRES", re.compile(rf"{_VILLAGE}{_POSTAL_CITY}")),
    # „na Długiej 5 m. 12” – maskujemy samą nazwę z numerem, „na” zostaje.
    PiiPattern("ADRES", re.compile(_COLLOQUIAL_STREET), value_group=1),
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
    # IPv6 (pełny i skrócony zapis, np. 2001:db8::1). Kształt łapie też
    # godziny („10:41:07”), dlatego decyduje walidator z modułu ipaddress.
    PiiPattern(
        "IP",
        re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])"),
        validator=is_valid_ipv6,
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


def mask_spelled_numbers(text: str, vault: Vault, found: list[dict]) -> str:
    """Telefony, kwoty, PESEL-e i daty zapisane SŁOWNIE (engine/numerals.py).

    W sejfie ląduje oryginalny zapis słowny, więc de-anonimizacja przywraca
    dokładnie te same słowa.
    """
    pieces, position = [], 0
    for item in find_spelled_numbers(text):
        value = text[item.start:item.end]
        token = vault.tokenize(item.category, value)
        found.append({"token": token, "kategoria": item.category,
                      "oryginał": value, "źródło": "regex"})
        pieces += [text[position:item.start], token]
        position = item.end
    pieces.append(text[position:])
    return "".join(pieces)


def mask_context_rules(text: str, vault: Vault, found: list[dict]) -> str:
    """Firmy przed formą prawną, zdrobnienia imion, „jedyny X”, niepełnosprawność
    (engine/context_rules.py). Po wzorcach formatu – ich tokenów nie ruszamy."""
    pieces, position = [], 0
    for item in find_context_matches(text):
        value = text[item.start:item.end]
        token = vault.tokenize(item.category, value)
        found.append({"token": token, "kategoria": item.category,
                      "oryginał": value, "źródło": "regex"})
        pieces += [text[position:item.start], token]
        position = item.end
    pieces.append(text[position:])
    return "".join(pieces)


def mask_with_regex(text: str, vault: Vault) -> tuple[str, list[dict]]:
    """Maskuje dane o stałym formacie. Zwraca (tekst_z_tokenami, wykryte)."""
    found: list[dict] = []
    for pattern in PATTERNS:
        text = pattern.regex.sub(_make_replacer(pattern, vault, found), text)
    text = mask_spelled_numbers(text, vault, found)
    text = mask_context_rules(text, vault, found)
    return text, found
