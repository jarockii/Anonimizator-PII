"""Sejf (vault) – słownik mapowań {token: oryginalna_wartość}.

To jedyne miejsce, w którym przechowujemy prawdziwe dane. Do chmury wysyłamy
tylko tekst z tokenami, a sejf zostaje lokalnie (w pamięci sesji Streamlit).
"""

import base64
import json
import re
import zlib

# Treść tokenu: KATEGORIA_numer. Kategoria może mieć podkreślniki
# (DOC_ID_1), więc „[A-Za-z]+_\d+” by nie wystarczyło.
TOKEN_BODY = r"[A-Za-z]+(?:_[A-Za-z]+)*_\d+"

# Token dokładnie w naszym formacie, np. [FIRMA_1] – do omijania tokenów
# podczas podmiany (żeby „FIRMA” w środku [FIRMA_1] nie została ruszona).
EXACT_TOKEN = rf"\[{TOKEN_BODY}\]"

# Rozpoznaje tokeny typu [OSOBA_1] w odpowiedzi z zewnętrznego AI.
# Jesteśmy tolerancyjni, bo ChatGPT potrafi „przerobić” token:
#   - dodać spacje:            [ OSOBA_1 ]
#   - zescapować nawiasy (MD): \[OSOBA_1\]
#   - zmienić wielkość liter:  [osoba_1]
TOKEN_PATTERN = re.compile(rf"\\?\[\s*({TOKEN_BODY})\s*\\?\]")

# Kategorie, w których wielkość liter MA znaczenie. „Zima2025” i „zima2025”
# to dwa różne hasła – sklejenie ich w jeden token przywróciłoby złe hasło.
CASE_SENSITIVE_CATEGORIES = {"HASLO", "SECRET"}

# Krótszych wartości nie propagujemy po całym dokumencie – „Jan” albo „5”
# zamaskowałoby przypadkowe fragmenty w innych miejscach.
MIN_PROPAGATION_LENGTH = 3


# Prefiks tokenu sejfu: rozpoznajemy po nim nasz format i jego wersję.
VAULT_TOKEN_PREFIX = "PIIVAULT1"


class VaultTokenError(ValueError):
    """Token sejfu jest niepoprawny, ucięty albo zmieniony."""


def _category_of(token: str) -> str:
    """„[DOC_ID_3]” -> „DOC_ID”."""
    return token[1:-1].rsplit("_", 1)[0]


def _normalize(category: str, value: str) -> str:
    """Klucz porównania: „LIVERI  SPÓŁKA” i „Liveri Spółka” to ta sama encja.

    casefold() to „mocniejsze” lower(): poprawnie porównuje też np. niemieckie ß.
    Zbijamy też wielokrotne spacje i złamania linii (częste w tekście z PDF).
    """
    if category in CASE_SENSITIVE_CATEGORIES:
        return value
    return " ".join(value.split()).casefold()


def value_pattern(value: str, ignore_case: bool) -> re.Pattern:
    """Wzorzec znajdujący wartość w tekście z pominięciem istniejących tokenów.

    Alternatywa 1 – istniejący token – jest dopasowywana tylko po to, żeby go
    „przeskoczyć”. Alternatywa 2 to sama wartość, przy czym dowolny odstęp
    (spacja, wiele spacji, nowa linia) pasuje do dowolnego odstępu.
    (?<!\\w) i (?!\\w) to granice słowa, działające też dla polskich liter.
    """
    words = r"\s+".join(re.escape(word) for word in value.split())
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(rf"({EXACT_TOKEN})|(?<!\w){words}(?!\w)", flags)


class Vault:
    """Przechowuje dwukierunkowe mapowanie token <-> oryginał."""

    def __init__(self) -> None:
        # Kierunek do de-anonimizacji: "[OSOBA_1]" -> "Jan Kowalski"
        self._token_to_value: dict[str, str] = {}
        # Kierunek odwrotny, żeby ta sama wartość zawsze dostała ten sam
        # token. Kluczem jest (kategoria, znormalizowana wartość).
        self._value_to_token: dict[tuple[str, str], str] = {}
        # Osobny licznik dla każdej kategorii: OSOBA_1, OSOBA_2, EMAIL_1...
        self._counters: dict[str, int] = {}

    def tokenize(self, category: str, value: str) -> str:
        """Zwraca token dla wartości; tworzy nowy tylko gdy to nowa wartość.

        „Liveri Sp. z o.o.” i „LIVERI SP. Z O.O.” dostają ten sam [FIRMA_1],
        więc zewnętrzne AI wie, że chodzi o tę samą firmę. Do de-anonimizacji
        zapamiętujemy formę, która pojawiła się pierwsza.
        """
        category = category.upper()
        key = (category, _normalize(category, value))
        if key in self._value_to_token:
            return self._value_to_token[key]

        self._counters[category] = self._counters.get(category, 0) + 1
        token = f"[{category}_{self._counters[category]}]"
        self._token_to_value[token] = value
        self._value_to_token[key] = token
        return token

    def replace_everywhere(self, text: str, value: str, token: str) -> tuple[str, int]:
        """Podmienia KAŻDE wystąpienie wartości w tekście na token.

        Zwraca (nowy_tekst, liczba_podmian). Bez względu na wielkość liter
        (poza hasłami), więc nazwa firmy z Karty Podpisów pisana WIELKIMI
        literami też zostanie zamaskowana.
        """
        category = _category_of(token)
        pattern = value_pattern(value, ignore_case=category not in CASE_SENSITIVE_CATEGORIES)
        count = 0

        def replace(match: re.Match) -> str:
            nonlocal count
            if match.group(1):  # istniejący token – zostawiamy
                return match.group(1)
            count += 1
            return token

        return pattern.sub(replace, text), count

    def mask_known(self, text: str) -> str:
        """Globalna propagacja: maskuje w tekście wszystko, co sejf już zna.

        Przykład: model 3B znalazł „Liveri Spółka z o.o.” na stronie 1, ale
        przegapił „LIVERI SPÓŁKA Z O.O.” na ostatniej stronie (Karta Podpisów).
        Ten krok to naprawia – bez ponownego pytania modelu.

        Najdłuższe wartości najpierw: inaczej „Liveri” podmienione wcześniej
        rozbiłoby dłuższe „Liveri Spółka z o.o.” na „[FIRMA_2] Spółka z o.o.”.
        """
        for key, token in sorted(self._value_to_token.items(),
                                 key=lambda item: len(item[0][1]), reverse=True):
            value = self._token_to_value[token]
            if len(value.strip()) >= MIN_PROPAGATION_LENGTH:
                text, _ = self.replace_everywhere(text, value, token)
        return text

    def restore(self, text: str) -> str:
        """Podmienia wszystkie znane tokeny w tekście na oryginalne dane.

        Nieznane tokeny zostawiamy bez zmian – lepiej pokazać użytkownikowi
        „[OSOBA_9]” niż po cichu coś zgadywać.
        """

        def replace(match: re.Match) -> str:
            token = f"[{match.group(1).upper()}]"
            return self._token_to_value.get(token, match.group(0))

        return TOKEN_PATTERN.sub(replace, text)

    def unknown_tokens(self, text: str) -> list[str]:
        """Lista tokenów z tekstu, których nie ma w sejfie (np. wymyślone przez AI)."""
        unknown = []
        for match in TOKEN_PATTERN.finditer(text):
            token = f"[{match.group(1).upper()}]"
            if token not in self._token_to_value and token not in unknown:
                unknown.append(token)
        return unknown

    # --- Token sejfu: cały sejf jako jeden ciąg znaków ---------------------------

    def to_token(self) -> str:
        """Pakuje cały sejf w jeden ciąg: PIIVAULT1.<dane>.<suma kontrolna>.

        Kroki: słownik -> JSON -> kompresja zlib -> base64 (bezpieczne znaki,
        da się skopiować i wkleić). Suma CRC32 wykrywa ucięty lub zmieniony
        token przy wklejaniu.

        UWAGA: to NIE jest szyfrowanie. Token jest kluczem do danych – kto go
        ma, odczyta oryginały. Nie wysyłaj go do zewnętrznego AI.
        """
        payload = json.dumps(
            {"values": self._token_to_value, "counters": self._counters},
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        data = base64.urlsafe_b64encode(zlib.compress(payload, level=9)).decode("ascii")
        checksum = f"{zlib.crc32(data.encode('ascii')):08x}"
        return f"{VAULT_TOKEN_PREFIX}.{data}.{checksum}"

    @classmethod
    def from_token(cls, token: str) -> "Vault":
        """Odtwarza sejf z tokenu (odwrotność to_token)."""
        # Usuwamy spacje i złamania linii, które mogły wejść przy kopiowaniu.
        token = "".join(token.split())
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != VAULT_TOKEN_PREFIX:
            raise VaultTokenError("To nie jest token sejfu (powinien zaczynać się od "
                                  f"„{VAULT_TOKEN_PREFIX}.”).")
        _, data, checksum = parts
        if f"{zlib.crc32(data.encode('ascii')):08x}" != checksum:
            raise VaultTokenError("Token jest ucięty albo zmieniony (zła suma kontrolna).")
        try:
            content = json.loads(zlib.decompress(base64.urlsafe_b64decode(data)))
        except (ValueError, zlib.error) as error:
            raise VaultTokenError(f"Nie da się odczytać tokenu: {error}") from error

        vault = cls()
        for token_name, value in content["values"].items():
            category = _category_of(token_name)
            vault._token_to_value[token_name] = value
            # Odbudowujemy też kierunek odwrotny, żeby kolejne anonimizacje
            # z tym sejfem dawały te same tokeny dla tych samych wartości.
            vault._value_to_token[(category, _normalize(category, value))] = token_name
        vault._counters = dict(content["counters"])
        return vault

    def get(self, token: str) -> str | None:
        return self._token_to_value.get(token)

    def items(self) -> list[tuple[str, str]]:
        return list(self._token_to_value.items())

    def clear(self) -> None:
        self._token_to_value.clear()
        self._value_to_token.clear()
        self._counters.clear()

    def __len__(self) -> int:
        return len(self._token_to_value)
