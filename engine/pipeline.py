"""Spina etapy w jeden pipeline: Regex -> NER -> LLM -> wynik + metryki."""

import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from engine.llm_masker import (
    DEFAULT_MODEL,
    OLLAMA_URL,
    OllamaError,
    apply_entities,
    detect_entities,
    detect_sensitive,
    find_injection_markers,
    verify_ner_categories,
)
from engine import ner_masker
from engine.file_reader import normalize_text
from engine.regex_masker import mask_with_regex
from engine.vault import Vault


# Kategorie, które można świadomie zostawić jawnie. Celowo BEZ identyfikatorów
# (OSOBA, PESEL, ADRES, POSREDNI…), sekretów i pozostałych danych szczególnych
# (WRAZLIWE: wyznanie, związki zawodowe…) – ich odsłonięcie przekreśla sens
# anonimizacji. Diagnoza bez nazwiska i PESEL-u nie wskazuje człowieka.
KEEPABLE_CATEGORIES = {"ZDROWIE", "STANOWISKO", "KWOTA", "DATA", "FIRMA"}


@dataclass
class AnonymizationResult:
    masked_text: str
    entities: list[dict] = field(default_factory=list)
    regex_ms: float = 0.0
    ner_ms: float = 0.0
    llm_ms: float = 0.0
    ner_error: str | None = None  # NER niedostępny (brak pakietu/modelu)
    # Kategorie NER poprawione przez LLM: {fragment: nowa kategoria}.
    ner_fixes: dict[str, str] = field(default_factory=dict)
    llm_error: str | None = None  # wypełnione, gdy LLM zawiódł (działa sam Regex)
    # Powody, dla których wynik NIE jest gotowy do wysłania bez sprawdzenia
    # (fail-closed): LLM nie działał albo dokument wygląda na prompt injection.
    review_reasons: list[str] = field(default_factory=list)
    # Wykryte, ale zostawione jawnie na życzenie użytkownika (keep_categories).
    kept_entities: list[dict] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return self.regex_ms + self.ner_ms + self.llm_ms

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)


def anonymize(
    text: str,
    vault: Vault,
    use_llm: bool = True,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_URL,
    on_progress: Callable[[int, int], None] | None = None,
    use_ner: bool = False,
    verify_ner: bool = False,
    keep_categories: Iterable[str] = (),
) -> AnonymizationResult:
    """Uruchamia cały pipeline anonimizacji.

    keep_categories: kategorie zostawiane jawnie (tylko z KEEPABLE_CATEGORIES).
    """
    keep = set(keep_categories)
    if not keep <= KEEPABLE_CATEGORIES:
        raise ValueError(f"Tych kategorii nie można zostawić jawnie: {sorted(keep - KEEPABLE_CATEGORIES)}")
    # Etap 0: normalizacja Unicode (niewidoczne znaki, cyfry pełnej szerokości).
    text = normalize_text(text)

    # Etap 1: Regex
    start = time.perf_counter()
    masked, found = mask_with_regex(text, vault)
    result = AnonymizationResult(masked_text=masked)
    result.regex_ms = (time.perf_counter() - start) * 1000

    # Etap 1b: NER (GLiNER2-PII). Szybko zaznacza osoby, firmy, adresy
    # i loginy. Brak pakietu lub modelu nie zatrzymuje pipeline'u – LLM
    # i tak przejrzy tekst, więc to nie jest powód do wstrzymania wyniku.
    if use_ner and masked.strip():
        start = time.perf_counter()
        try:
            ner_entities = ner_masker.detect_entities(masked)
        except ner_masker.NerUnavailableError as error:
            result.ner_error = str(error)
            ner_entities = []
        result.ner_ms = (time.perf_counter() - start) * 1000

        # Opcjonalnie (domyślnie wył.): LLM poprawia KATEGORIE niepewnych
        # wykryć NER. Nigdy ich nie usuwa. qwen2.5:3b częściej psuł kategorie
        # niż poprawiał, dlatego to eksperyment do sprawdzenia z innym modelem.
        if use_llm and verify_ner and ner_entities:
            start = time.perf_counter()
            try:
                fixes = verify_ner_categories(masked, ner_entities, model=model, base_url=base_url)
            except OllamaError:
                fixes = {}
            for entity in ner_entities:
                if entity.original in fixes:
                    entity.tag = fixes[entity.original]
            result.ner_fixes = fixes
            result.llm_ms += (time.perf_counter() - start) * 1000

        fixed = [e for e in ner_entities if e.original in result.ner_fixes]
        masked, fixed_found = apply_entities(masked, fixed, vault, source="ner+llm")
        masked, ner_found = apply_entities(
            masked, [e for e in ner_entities if e.original not in result.ner_fixes], vault,
            source="ner")
        found.extend(fixed_found + ner_found)
        result.masked_text = masked
        if result.ner_error and not use_llm:
            result.review_reasons.append(
                "Ani NER, ani LLM nie sprawdziły tekstu – imiona, nazwiska i firmy mogą "
                f"być NIEZAMASKOWANE. Przyczyna: {result.ner_error}"
            )

    # Etap 2: LLM. Gdy Ollama nie działa, NIE przerywamy – zwracamy wynik
    # z samego Regexa (i NER) i komunikat. Lepsza częściowa ochrona niż żadna.
    # LLM dostaje tekst już zamaskowany przez NER: ma mniej do roboty i może
    # skupić się na tym, co wymaga kontekstu.
    if use_llm and masked.strip():
        start = time.perf_counter()
        try:
            entities = detect_entities(
                masked, model=model, base_url=base_url, on_progress=on_progress
            )
            masked, llm_found = apply_entities(masked, entities, vault)
            found.extend(llm_found)
            # Osobne, wąskie przejście: dane szczególne (art. 9 RODO) – tylko
            # we fragmentach, w których słownik coś podpowiada.
            sensitive = detect_sensitive(masked, model=model, base_url=base_url)
            masked, sensitive_found = apply_entities(masked, sensitive, vault)
            found.extend(sensitive_found)
            result.masked_text = masked
        except OllamaError as error:
            result.llm_error = str(error)
            result.review_reasons.append(
                "Etap LLM nie zadziałał – imiona, nazwiska, firmy i adresy bez stałego "
                f"formatu mogą być NIEZAMASKOWANE. Przyczyna: {error}"
            )
        result.llm_ms += (time.perf_counter() - start) * 1000

        markers = find_injection_markers(masked)
        if markers:
            result.review_reasons.append(
                "Dokument zawiera polecenia skierowane do modelu (możliwy prompt "
                "injection), więc model mógł celowo pominąć dane: "
                + "; ".join(f"„{marker}”" for marker in markers[:5])
            )

    # Etap 2b: globalna propagacja. Wszystko, co sejf już zna (z Regexa lub
    # LLM-a), maskujemy w CAŁYM dokumencie, bez względu na wielkość liter.
    # Łapie np. nazwę firmy WIELKIMI LITERAMI na ostatniej stronie, którą
    # model przegapił, bo przetwarzał tę stronę osobno.
    result.masked_text = vault.mask_known(result.masked_text)
    # Etap 2c: samo nazwisko osoby, którą już zamaskowano w pełnej formie
    # („Nazwisko: PIOTROWSKI” na skanie dowodu obok „DAMIAN PIOTROWSKI”).
    result.masked_text, surname_found = vault.mask_surnames(result.masked_text)
    found.extend(surname_found)

    # Etap 3: kategorie, które użytkownik ŚWIADOMIE zostawia jawnie (np. diagnozy,
    # żeby AI mogło streścić dokumentację medyczną). Wykrywamy je normalnie,
    # a dopiero na końcu przywracamy – dzięki temu tabela pokazuje, co zostało
    # odsłonięte, a czarne paski w PDF ich nie zakrywają.
    result.masked_text, kept_tokens = vault.restore_categories(result.masked_text, keep)

    # Ta sama wartość mogła wystąpić kilka razy – w tabeli pokazujemy raz.
    unique: dict[str, dict] = {}
    for item in found:
        unique.setdefault(item["token"], item)
    result.kept_entities = [e for e in unique.values() if e["token"] in kept_tokens]
    result.entities = [e for e in unique.values() if e["token"] not in kept_tokens]
    return result


def get_process_peak_ram_mb() -> float | None:
    """Szczytowe zużycie RAM przez proces Pythona (bez modelu w Ollamie)."""
    try:
        import resource  # moduł dostępny tylko na Linux/macOS
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS podaje bajty, Linux kilobajty.
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024
