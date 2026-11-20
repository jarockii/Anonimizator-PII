"""Spina etapy w jeden pipeline: Regex -> LLM -> wynik + metryki."""

import sys
import time
from dataclasses import dataclass, field
from typing import Callable

from engine.llm_masker import (
    DEFAULT_MODEL,
    OLLAMA_URL,
    OllamaError,
    apply_entities,
    detect_entities,
    find_injection_markers,
)
from engine.file_reader import normalize_text
from engine.regex_masker import mask_with_regex
from engine.vault import Vault


@dataclass
class AnonymizationResult:
    masked_text: str
    entities: list[dict] = field(default_factory=list)
    regex_ms: float = 0.0
    llm_ms: float = 0.0
    llm_error: str | None = None  # wypełnione, gdy LLM zawiódł (działa sam Regex)
    # Powody, dla których wynik NIE jest gotowy do wysłania bez sprawdzenia
    # (fail-closed): LLM nie działał albo dokument wygląda na prompt injection.
    review_reasons: list[str] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return self.regex_ms + self.llm_ms

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
) -> AnonymizationResult:
    """Uruchamia cały pipeline anonimizacji."""
    # Etap 0: normalizacja Unicode (niewidoczne znaki, cyfry pełnej szerokości).
    text = normalize_text(text)

    # Etap 1: Regex
    start = time.perf_counter()
    masked, found = mask_with_regex(text, vault)
    result = AnonymizationResult(masked_text=masked)
    result.regex_ms = (time.perf_counter() - start) * 1000

    # Etap 2: LLM. Gdy Ollama nie działa, NIE przerywamy – zwracamy wynik
    # z samego Regexa i komunikat. Lepsza częściowa ochrona niż żadna.
    if use_llm and masked.strip():
        start = time.perf_counter()
        try:
            entities = detect_entities(
                masked, model=model, base_url=base_url, on_progress=on_progress
            )
            masked, llm_found = apply_entities(masked, entities, vault)
            found.extend(llm_found)
            result.masked_text = masked
        except OllamaError as error:
            result.llm_error = str(error)
            result.review_reasons.append(
                "Etap LLM nie zadziałał – imiona, nazwiska, firmy i adresy bez stałego "
                f"formatu mogą być NIEZAMASKOWANE. Przyczyna: {error}"
            )
        result.llm_ms = (time.perf_counter() - start) * 1000

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

    # Ta sama wartość mogła wystąpić kilka razy – w tabeli pokazujemy raz.
    unique: dict[str, dict] = {}
    for item in found:
        unique.setdefault(item["token"], item)
    result.entities = list(unique.values())
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
