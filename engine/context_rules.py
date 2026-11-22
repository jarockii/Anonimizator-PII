"""Reguły kontekstowe: dane bez stałego formatu, które jednak mają stały KSZTAŁT.

Regex formatu (engine/regex_masker.py) łapie numery z sumami kontrolnymi.
NER i LLM łapią osoby i opisy, ale nie zawsze powtarzalnie. Tu są przypadki,
które w zestawie kontrolnym (eval/oznaczenia_kontrolne) wyciekały w KAŻDYM
przebiegu, a da się je opisać prostą regułą:

- firma przed formą prawną: „HYDRO-TECH PSZCZYNA SP. Z O.O.” (LLM zwracał
  tylko „PSZCZYNA SP. Z O.O.”, a „HYDRO-TECH” zostawało),
- zdrobnienia imion: „Zdzichu”, „Wiesia”, „Bartka” (NER ich nie zna),
- unikalna cecha: „jedyna kobieta”, „jedynej apteki” – identyfikacja pośrednia,
- stopień i symbol niepełnosprawności: „znacznym stopniu niepełnosprawności”,
  „05-R” (dane o zdrowiu w urzędowym, powtarzalnym zapisie).

Reguły są ogólne (dowolna firma, dowolne zdrobnienie z listy), a nie pod
konkretne słowa z dokumentów kontrolnych – te pozostają testem uczciwości.
"""

import re
from dataclasses import dataclass


@dataclass
class ContextMatch:
    start: int
    end: int
    category: str


# --- Firma przed formą prawną ------------------------------------------------------

_UPPER = "A-ZĄĆĘŁŃÓŚŹŻ"
_LOWER = "a-ząćęłńóśźż"
# Słowo nazwy firmy: zaczyna się wielką literą („Nordis”, „HYDRO-TECH”, „P&K”).
_NAME_WORD = rf"[{_UPPER}][{_UPPER}{_LOWER}0-9&'.-]*"
_LEGAL_FORM = (
    r"(?i:sp\.\s?z\s?o\.\s?o\.|spółk[aąięi]\s+z\s+o\.\s?o\.|spółk[aąięi]\s+z\s+ograniczoną\s+"
    r"odpowiedzialnością|s\.\s?a\.|spółk[aąięi]\s+akcyjn\w+|sp\.\s?k\.|sp\.\s?j\.|s\.\s?c\."
    r"|spółk[aąięi]\s+komandytow\w+|spółk[aąięi]\s+jawn\w+|gmbh|ltd\.?|inc\.?|llc)"
)
# Nazwa = 1–5 słów z wielkiej litery tuż przed formą prawną (także przez złamanie linii).
# Odstęp między słowami nazwy. Zwykle spacje/złamanie linii, ale w tabeli PDF
# czytanej wierszami nazwa złamana w komórce przeplata się z liczbami z sąsiednich
# kolumn: „HYDRO-TECH        7      11\n      PSZCZYNA SP. Z O.O.” – wtedy liczby
# przed końcem linii też należą do odstępu (trafiają do tokenu firmy).
_GAP = r"(?:(?:[ \t]+\d[\d.,]*)+[ \t]*\n\s*|\s+)"
# Między słowami może stać spójnik: „Kowalczyk i Wspólnicy sp.k.”, „Smith & Co Ltd”.
_COMPANY = re.compile(
    rf"(?<![\w&'.-])((?:{_NAME_WORD}{_GAP}(?:(?:i|oraz|and|&)\s+(?={_NAME_WORD}))?){{1,5}})"
    rf"{_LEGAL_FORM}(?![\w])")
# Słowa z wielkiej litery, które stoją przed nazwą, ale nią nie są
# („Prezes Zarządu Nordis Sp. z o.o.”, „Faktura Optiprint S.A.”).
_NOT_NAME = re.compile(
    r"(?i)^(?:prezes\w*|zarząd\w*|wiceprezes\w*|dyrektor\w*|członek|członk\w*|spółk\w*|firm\w*"
    r"|pracodawc\w*|pracowni\w*|najemc\w*|wynajmując\w*|zleceniodawc\w*|zleceniobiorc\w*"
    r"|dostawc\w*|odbiorc\w*|klient\w*|wykonawc\w*|zamawiając\w*|umow\w*|faktur\w*|przelew\w*"
    r"|wynagrodzeni\w*|sprzedawc\w*|nabywc\w*|kupując\w*|sprzedając\w*|pełnomocni\w*"
    r"|reprezentowan\w*|w|z|ze|i|oraz|dla|od|do|na)$"
)


def _companies(text: str) -> list[ContextMatch]:
    found = []
    for match in _COMPANY.finditer(text):
        start = match.start(1)
        # Odcinamy z przodu słowa-funkcje („Prezes Zarządu …”).
        for word in re.finditer(r"\S+", match.group(1)):
            if not _NOT_NAME.match(word.group(0).strip(":,;")):
                start = match.start(1) + word.start()
                break
        else:
            continue   # same słowa-funkcje + forma prawna – to nie nazwa
        found.append(ContextMatch(start, match.end(), "FIRMA"))
    return found


# --- Zdrobnienia imion ------------------------------------------------------------

# Mianowniki (i wołacze) zdrobnień. Pomijamy dwuznaczne („Kuba” = państwo,
# „Halka” = opera, „Maja” = miesiąc w dopełniaczu… „maja” jest małą literą,
# ale na początku zdania pisze się wielką).
_DIMINUTIVES = """
Zdzichu Zdzisiek Zdziś Krzysiu Krzysiek Krzyś Józek Józiu Józio Franek Franio Staszek Stasiu Stasio
Wojtek Wojtuś Bartek Bartuś Maciek Maciuś Tomek Tomuś Grzesiek Grześ Mietek Miecio Heniek Henio
Rysiek Rysio Zbyszek Zbychu Zbysio Zenek Zenio Wiesiek Wiesio Antek Antoś Jasiu Jasio Jaś Kazik
Kaziu Władek Władzio Mirek Jurek Darek Arek Piotrek Piotruś Pawełek Michałek Kubuś Szymek Szymuś
Romek Sławek Radek Andrzejek Jacuś Bogdanek Tadzio Tadek Edzio Gienek Gienio Felek Lolek
Wiesia Gosia Kasia Basia Asia Ania Ola Ula Ela Hania Zosia Jadzia Krysia Halinka Danusia Grażynka
Bożenka Madzia Iwonka Ewka Ewunia Marysia Zuzia Julka Natka Stasia Józia Celinka Tereska
Irenka Renia Bogusia Lusia Ludka Wandzia Helenka Agusia Dorotka Beatka Aneczka Kasieńka Gośka Baśka
"""


def _diminutive_forms(word: str) -> set[str]:
    """Mianownik + najczęstsze formy (dopełniacz, celownik, biernik, narzędnik)."""
    forms = {word}
    if word.endswith("ek"):                    # Bartek -> Bartka, Bartkowi, Bartkiem
        stem = word[:-2] + "k"
        forms |= {stem + "a", stem + "owi", stem + "iem"}
    elif word.endswith("ś"):                   # Zdziś -> Zdzisia, Zdzisiowi, Zdzisiem
        stem = word[:-1] + "si"
        forms |= {stem + "a", stem + "owi", stem + "em"}
    elif word.endswith(("u", "o")):            # Zdzichu -> Zdzicha, Franio -> Frania
        forms |= {word[:-1] + "a", word[:-1] + "owi", word[:-1] + "em"}
    elif word.endswith("ia"):                  # Kasia -> Kasi, Kasię, Kasią
        forms |= {word[:-1], word[:-1] + "ę", word[:-1] + "ą"}
    elif word.endswith("a"):                   # Ola -> Oli, Olę, Olą; Grażynka -> Grażynki
        stem = word[:-1]
        forms |= {stem + ("i" if stem[-1] in "kgl" else "y"), stem + "ę", stem + "ą"}
        if stem[-1] == "l":                    # Ola -> Oli (nie „Oly”)
            forms.add(stem + "i")
    return forms


_DIMINUTIVE_FORMS = sorted({form for word in _DIMINUTIVES.split() for form in _diminutive_forms(word)},
                           key=len, reverse=True)
_DIMINUTIVE = re.compile(r"(?<![\w-])(?:" + "|".join(map(re.escape, _DIMINUTIVE_FORMS)) + r")(?![\w-])")


def _diminutives(text: str) -> list[ContextMatch]:
    return [ContextMatch(m.start(), m.end(), "OSOBA") for m in _DIMINUTIVE.finditer(text)]


# --- „jedyny X” – identyfikacja pośrednia -------------------------------------------

# „jedyna kobieta”, „jedynym fizjoterapeutą”, „jedynej apteki” wskazują jedną osobę
# (albo jej najbliższych). Nie: „jedyny sposób”, „jedyna szansa”, „jedynie”.
_NOT_UNIQUE_PERSON = (
    "sposób|sposobem|sposobu|możliwość|możliwości|rozwiązanie|rozwiązaniem|wyjście|wyjściem|szansa"
    "|szansę|szansą|raz|warunek|warunkiem|przypadek|przypadku|problem|problemem|powód|powodem|cel"
    "|celem|dzień|dnia|dokument|dokumentem|plik|termin|terminem|kopia|kopię|egzemplarz|źródło"
    "|źródłem|wyjątek|wyjątkiem|metoda|metodą|opcja|opcją|droga|drogą|nadzieja|nadzieją|rzecz|rzeczą"
)
_UNIQUE = re.compile(
    r"(?i)(?<![\w-])jedyn(?:y|a|e|ego|ej|emu|ą|ym|ych|i)\s+"
    rf"(?!(?:{_NOT_UNIQUE_PERSON})(?![\w]))[^\W\d_]{{3,}}"
)


def _unique_descriptions(text: str) -> list[ContextMatch]:
    return [ContextMatch(m.start(), m.end(), "POSREDNI") for m in _UNIQUE.finditer(text)]


# Potoczny opis osoby: „pan z czwórki, ten co jeździ czerwoną taksówką”,
# „ta co sprząta na trzecim piętrze”. Maskujemy sam opis (do końca członu
# zdania), zostawiając „ten co”. Tylko potoczne „co” – urzędowe „ta, która”
# częściej dotyczy umowy albo strony niż człowieka.
_COLLOQUIAL_DESCRIPTION = re.compile(
    r"(?i)(?<![\w-])(?:ten|ta|tamten|tamta),?\s+co\s+([^,.;:!?\n\[\]]{6,70}?)(?=\s*[,.;:!?\n]|$)")


def _colloquial_descriptions(text: str) -> list[ContextMatch]:
    return [ContextMatch(m.start(1), m.end(1), "POSREDNI")
            for m in _COLLOQUIAL_DESCRIPTION.finditer(text)]


# --- Niepełnosprawność (zapis urzędowy) ----------------------------------------------

_DISABILITY = re.compile(
    r"(?i)(?<!\w)(?:(?:lekki\w*|umiarkowan\w*|znaczn\w*)\s+stop\w*\s+niepełnosprawnoś\w*"
    r"|stop\w*\s+niepełnosprawnoś\w*\s*[:–-]?\s*(?:lekki|umiarkowany|znaczny)\w*)"
    # Symbole przyczyny niepełnosprawności z orzeczenia: 01-U … 12-C.
    r"|(?<![\w-])(?:01-U|02-P|03-L|04-O|05-R|06-E|07-S|08-T|09-M|10-N|11-I|12-C)(?![\w-])"
)


def _disability(text: str) -> list[ContextMatch]:
    return [ContextMatch(m.start(), m.end(), "ZDROWIE") for m in _DISABILITY.finditer(text)]


def find_context_matches(text: str) -> list[ContextMatch]:
    """Wszystkie dopasowania, bez nakładania się (dłuższe i wcześniejsze wygrywają)."""
    candidates = (_companies(text) + _disability(text) + _unique_descriptions(text)
                  + _colloquial_descriptions(text) + _diminutives(text))
    candidates.sort(key=lambda m: (m.start, -(m.end - m.start)))
    chosen: list[ContextMatch] = []
    for match in candidates:
        if chosen and match.start < chosen[-1].end:
            continue
        chosen.append(match)
    return chosen
