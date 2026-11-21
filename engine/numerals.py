"""Liczby zapisane SŁOWNIE: telefony, kwoty, PESEL-e i daty.

Regex widzi tylko cyfry, a w czatach, aktach notarialnych i transkrypcjach
rozmów dane bywają zapisane słowami:
- „sześćset dwa, trzysta czterdzieści jeden, dwieście dwanaście” (telefon),
- „za cenę czterystu dwunastu tysięcy złotych” (kwota),
- „dnia dwudziestego czwartego lutego roku dwa tysiące dwudziestego piątego” (data).

Parser zamienia ciąg liczebników na liczby (mianownik, dopełniacz i liczebniki
porządkowe), a potem decyduje, czym jest całość. Maskujemy ORYGINALNE słowa –
de-anonimizacja przywraca tekst co do litery. Pojedyncze liczebniki bez
kontekstu („jeden z pracowników”, „trzy miesiące”) zostają nietknięte.
"""

import re
from dataclasses import dataclass

# Wartość i „rząd” słowa: 3 = setki, 2 = dziesiątki, 1 = jedności/nastki.
# Nastki zajmują też miejsce jedności, dlatego mają rząd 1.
_WORDS: dict[str, tuple[int, int]] = {}


def _add(forms: str, value: int, order: int) -> None:
    for form in forms.split():
        _WORDS[form] = (value, order)


for _v, _forms in enumerate([
    "zero", "jeden jedna jedno jednego jednej jednemu pierwszy pierwszego pierwsza pierwszej",
    "dwa dwie dwóch dwu dwoma drugi drugiego druga drugiej",
    "trzy trzech trzema trzeci trzeciego trzecia trzeciej",
    "cztery czterech czterema czwarty czwartego czwarta czwartej",
    "pięć pięciu pięcioma piąty piątego piąta piątej",
    "sześć sześciu sześcioma szósty szóstego szósta szóstej",
    "siedem siedmiu siedmioma siódmy siódmego siódma siódmej",
    "osiem ośmiu ośmioma ósmy ósmego ósma ósmej",
    "dziewięć dziewięciu dziewięcioma dziewiąty dziewiątego dziewiąta dziewiątej",
]):
    _add(_forms, _v, 1)
for _v, _stem in [(10, "dziesię"), (11, "jedenaś"), (12, "dwanaś"), (13, "trzynaś"),
                  (14, "czternaś"), (15, "piętnaś"), (16, "szesnaś"), (17, "siedemnaś"),
                  (18, "osiemnaś"), (19, "dziewiętnaś")]:
    base = {10: "dziesięć dziesięciu dziesiąty dziesiątego dziesiąta dziesiątej"}.get(_v)
    if base is None:
        stem = _stem[:-1]  # „jedenaś” -> „jedena”
        base = (f"{stem}ście {stem}stu {stem}sty {stem}stego {stem}sta {stem}stej")
    _add(base, _v, 1)
# Nieregularne: dwanaście -> dwUnastu, dwUnasty (nie „dwanastu”).
_add("dwunastu dwunasty dwunastego dwunasta dwunastej", 12, 1)
_add("dwadzieścia dwudziestu dwudziesty dwudziestego dwudziesta dwudziestej", 20, 2)
_add("trzydzieści trzydziestu trzydziesty trzydziestego trzydziesta trzydziestej", 30, 2)
_add("czterdzieści czterdziestu czterdziesty czterdziestego", 40, 2)
for _v, _w in [(50, "pięćdziesiąt pięćdziesięciu pięćdziesiąty pięćdziesiątego"),
               (60, "sześćdziesiąt sześćdziesięciu sześćdziesiąty sześćdziesiątego"),
               (70, "siedemdziesiąt siedemdziesięciu siedemdziesiąty siedemdziesiątego"),
               (80, "osiemdziesiąt osiemdziesięciu osiemdziesiąty osiemdziesiątego"),
               (90, "dziewięćdziesiąt dziewięćdziesięciu dziewięćdziesiąty dziewięćdziesiątego")]:
    _add(_w, _v, 2)
for _v, _w in [(100, "sto stu setny setnego"), (200, "dwieście dwustu dwusetny dwusetnego"),
               (300, "trzysta trzystu"), (400, "czterysta czterystu"),
               (500, "pięćset pięciuset"), (600, "sześćset sześciuset"),
               (700, "siedemset siedmiuset"), (800, "osiemset ośmiuset"),
               (900, "dziewięćset dziewięciuset")]:
    _add(_w, _v, 3)

# Mnożniki: „czterystu dwunastu TYSIĘCY”, „dwa MILIONY”.
_MULTIPLIERS = {}
for _form in "tysiąc tysiące tysięcy tysiąca tysiącem tysięczny tysięcznego".split():
    _MULTIPLIERS[_form] = 1000
for _form in "milion miliony milionów miliona milionem".split():
    _MULTIPLIERS[_form] = 1_000_000

_CURRENCY = re.compile(
    r"\s*(?:złotych|złote|złoty|złotego|zł|pln|euro|eur|dolarów|dolary|usd|franków|chf|funtów|gbp)"
    # „… złotych i pięćdziesięciu groszy” – najwyżej 4 słowa przed „groszy”.
    r"(?:\s+(?:i\s+)?(?:[a-ząćęłńóśźż]+\s+){0,4}groszy)?\b",
    re.IGNORECASE,
)
_MONTHS = ("stycznia lutego marca kwietnia maja czerwca lipca sierpnia września "
           "października listopada grudnia").split()
_WORD = re.compile(r"[a-ząćęłńóśźż]+", re.IGNORECASE)
# Między liczebnikami jednego ciągu może stać tylko spacja, przecinek lub myślnik.
_GAP = re.compile(r"[\s,–-]*")


@dataclass
class SpelledNumber:
    start: int
    end: int
    category: str      # TEL, KWOTA, PESEL, DATA
    digits: str        # do testów i debugowania: „602341212”


def _groups(words: list[str]) -> list[int]:
    """Zamienia ciąg liczebników na liczby. Nowa liczba zaczyna się, gdy słowo
    ma rząd >= rzędu poprzedniego („sześćset dwa | trzysta …”)."""
    numbers: list[int] = []
    total, current, last_order = 0, 0, 99
    started = False
    # Czy od ostatniego mnożnika padł jakiś liczebnik – NIE „current > 0”,
    # bo „zero” ma wartość 0, a „zero trzy” to dwie cyfry (PESEL dyktowany
    # cyfra po cyfrze), nie 3.
    pending = False
    for word in words:
        key = word.casefold()
        if key in _MULTIPLIERS:
            total += max(current, 1) * _MULTIPLIERS[key]
            current, last_order, started, pending = 0, 99, True, False
            continue
        value, order = _WORDS[key]
        if pending and order >= last_order:
            numbers.append(total + current)
            total, current = 0, 0
        current += value
        last_order = order
        started = pending = True
    if started:
        numbers.append(total + current)
    return numbers


def _is_pesel(digits: str) -> bool:
    if len(digits) != 11:
        return False
    weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    return (10 - sum(int(d) * w for d, w in zip(digits, weights)) % 10) % 10 == int(digits[10])


def find_spelled_numbers(text: str) -> list[SpelledNumber]:
    """Znajduje telefony, kwoty, PESEL-e i daty zapisane słownie."""
    found: list[SpelledNumber] = []
    words = list(_WORD.finditer(text))
    i = 0
    while i < len(words):
        if words[i].group().casefold() not in _WORDS and words[i].group().casefold() not in _MULTIPLIERS:
            i += 1
            continue
        # Najdłuższy ciąg liczebników przedzielonych tylko spacją/przecinkiem.
        j = i
        while (j + 1 < len(words)
               and (words[j + 1].group().casefold() in _WORDS
                    or words[j + 1].group().casefold() in _MULTIPLIERS)
               and _GAP.fullmatch(text[words[j].end():words[j + 1].start()])):
            j += 1
        run = [w.group() for w in words[i:j + 1]]
        start, end = words[i].start(), words[j].end()
        numbers = _groups(run)
        digits = "".join(str(n) for n in numbers)
        category = None

        # Data: liczebnik porządkowy dnia + miesiąc + (roku) + rok słownie.
        nxt = words[j + 1] if j + 1 < len(words) else None
        if (len(numbers) == 1 and 1 <= numbers[0] <= 31 and nxt is not None
                and nxt.group().casefold() in _MONTHS
                and text[end:nxt.start()].strip() == ""):
            k = j + 2
            if k < len(words) and words[k].group().casefold() in ("roku", "rok"):
                k += 1
            year_end = nxt.end()
            m = k
            while m < len(words) and (words[m].group().casefold() in _WORDS
                                      or words[m].group().casefold() in _MULTIPLIERS):
                m += 1
            if m > k:
                year = sum(_groups([w.group() for w in words[k:m]]))
                if 1900 <= year <= 2100:
                    year_end = words[m - 1].end()
            found.append(SpelledNumber(start, year_end, "DATA", str(numbers[0])))
            i = max(m, j + 2)
            continue

        currency = _CURRENCY.match(text, end)
        # „zero zero czterdzieści osiem, sześćset jeden…” = 0048 601…
        phone = digits[2:] if digits.startswith("00") else digits
        if currency and currency.group().strip():
            category, end = "KWOTA", currency.end()
        elif len(phone) == 9 or (len(phone) == 11 and phone.startswith("48")):
            category = "TEL"
        elif _is_pesel(digits):
            category = "PESEL"
        if category:
            found.append(SpelledNumber(start, end, category, digits))
        i = j + 1
    return found
