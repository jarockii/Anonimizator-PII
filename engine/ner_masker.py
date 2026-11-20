"""Etap 1b: NER (GLiNER2-PII) – szybkie wykrywanie osób, firm, adresów i loginów.

Kolejność w pipeline: Regex -> NER -> LLM.
- Regex: dane o stałym formacie (PESEL, IBAN…), mikrosekundy.
- NER: model typu encoder (~0,2–0,3 mld parametrów) zaznacza fragmenty tekstu.
  Nie generuje tekstu, więc nie halucynuje i nie da się go „przekonać”
  poleceniem w dokumencie (prompt injection). Działa na CPU w ułamku czasu LLM.
- LLM (Ollama): dostaje tekst już zamaskowany przez Regex i NER i szuka tego,
  co wymaga rozumienia kontekstu – stanowisk, odmienionych nazwisk, które
  NER przeoczył, nazw firm w nietypowej formie.

GLiNER2-PII był trenowany na EN/FR/ES/DE/IT/PT/NL – po polsku działa
„zero-shot”, dlatego progi pewności dobieramy pomiarem, a LLM zostaje jako
druga linia obrony.

Zależność jest opcjonalna: bez pakietu `gliner2` (i PyTorcha) aplikacja
działa jak wcześniej, tylko bez tego etapu.
"""

import importlib.util
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from engine.llm_masker import CHUNK_CHARS, split_into_chunks

# Model z Hugging Face (Apache-2.0, safetensors). Można podmienić zmienną
# środowiskową, np. na lokalny katalog z dotrenowanym modelem po polsku.
NER_MODEL = os.environ.get("ANONIMIZATOR_NER_MODEL", "fastino/gliner2-privacy-filter-PII-multi")

# Etykieta modelu -> nasza kategoria tokenu. GLiNER2 przyjmuje dowolne
# etykiety tekstowe; „person”, „address”, „username” są z listy, na której
# model był trenowany, „company” to etykieta ogólna GLiNER2.
LABELS = {
    "person": "OSOBA",
    "company": "FIRMA",
    "address": "ADRES",
    "username": "LOGIN",
}

# Progi pewności per etykieta – dobrane pomiarem na przyklady/*.pdf
# i przypadkach testowych z app.py (zero-shot, polski tekst):
# - person 0.5: wszystkie osoby miały 0.86–1.0, ale samo imię „DAMIAN” 0.53,
# - company 0.6: odcina „RZECZPOSPOLITA POLSKA” (0.51) z nagłówka dowodu
#   i „Urzędem Skarbowym” (0.32); prawdziwe firmy miały 0.72–1.0,
# - username 0.95: login „tnowak” w logu ma 1.0, a słowo „zona” (OCR
#   zgubił „ż”) dostało 0.93, śmieci OCR z podpisu – 0.51.
THRESHOLDS = {
    "person": 0.5,
    "company": 0.6,
    "address": 0.5,
    "username": 0.95,
}

# Fragment wejściowy NER. Model widzi więcej kontekstu niż LLM (400 znaków),
# bo nie „gubi uwagi” jak mały LLM, a mniej wywołań = szybciej.
NER_CHUNK_CHARS = 1500


@dataclass
class NerEntity:
    original: str
    tag: str
    confidence: float


class NerUnavailableError(RuntimeError):
    """Pakiet gliner2 nie jest zainstalowany albo model się nie wczytał."""


def ner_installed() -> bool:
    """Czy pakiet gliner2 jest zainstalowany (bez importowania PyTorcha)."""
    return importlib.util.find_spec("gliner2") is not None


def _model_cached() -> bool:
    """Czy wagi modelu są już w lokalnej pamięci podręcznej Hugging Face.

    Sprawdzamy sam katalog (bez importu huggingface_hub), bo tryb offline
    trzeba ustawić PRZED importem biblioteki – czyta go tylko raz.
    """
    if Path(NER_MODEL).is_dir():  # lokalny katalog z modelem
        return True
    hub = Path(os.environ.get("HF_HUB_CACHE")
               or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    snapshots = hub / f"models--{NER_MODEL.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


_model = None
_model_lock = threading.Lock()


def load_model():
    """Wczytuje model raz na proces (kolejne wywołania biorą go z pamięci).

    Pierwsze uruchomienie pobiera wagi z Hugging Face do lokalnej pamięci
    podręcznej; później model ładuje się z dysku. Tekst dokumentów nigdy
    nie jest nigdzie wysyłany – inferencja jest lokalna.
    """
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            if not ner_installed():
                raise NerUnavailableError(
                    "Pakiet gliner2 nie jest zainstalowany (pip install \"gliner2[local]\")."
                )
            # Bez telemetrii, a gdy model jest już pobrany – całkowicie
            # offline: aplikacja nie łączy się z Hugging Face przy starcie.
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            if _model_cached():
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            try:
                from gliner2 import GLiNER2

                _model = GLiNER2.from_pretrained(NER_MODEL)
            except Exception as error:  # brak sieci przy 1. pobraniu, złe wagi…
                raise NerUnavailableError(f"Nie udało się wczytać modelu NER: {error}") from error
    return _model


_preload_started = False


def preload_model() -> None:
    """Zaczyna wczytywać model w tle (raz na proces). Późniejsze
    load_model() poczeka na koniec zamiast wczytywać drugi raz."""
    global _preload_started
    if _preload_started or _model is not None:
        return
    _preload_started = True

    def run() -> None:
        try:
            load_model()
        except NerUnavailableError:
            pass  # błąd pokaże się przy pierwszej anonimizacji

    threading.Thread(target=run, name="ner-preload", daemon=True).start()


def _extract(model, text: str) -> list[NerEntity]:
    """Jedno wywołanie modelu; filtruje wyniki progami per etykieta."""
    result = model.extract_entities(
        text,
        list(LABELS),
        threshold=min(THRESHOLDS.values()),
        include_confidence=True,
    )
    found = []
    for label, items in result.get("entities", {}).items():
        if label not in LABELS:
            continue
        for item in items:
            text_value = item["text"] if isinstance(item, dict) else item
            confidence = item.get("confidence", 1.0) if isinstance(item, dict) else 1.0
            if confidence >= THRESHOLDS[label]:
                found.append(NerEntity(text_value, LABELS[label], float(confidence)))
    return found


def detect_entities(text: str) -> list[NerEntity]:
    """Wykrywa encje w całym tekście (strona po stronie, fragmentami)."""
    model = load_model()
    entities: list[NerEntity] = []
    for chunk in split_into_chunks(text, max_chars=max(NER_CHUNK_CHARS, CHUNK_CHARS)):
        if chunk.strip():
            entities.extend(_extract(model, chunk))
    return entities
