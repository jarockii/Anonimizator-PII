"""Ocena wyniku anonimizacji względem ręcznych oznaczeń (eval/oznaczenia/*.json).

Czyste funkcje, bez modeli – testowane w tests/test_eval.py.

Format oznaczeń:
{
  "plik": "przyklady/1_umowa.pdf",
  "opis": "…",
  "dane": [{"kategoria": "OSOBA", "fragmenty": ["Grzegorza", "Wróblewskiego"], "opis": "…"}],
  "zwykle_slowa": ["Najemcy", "§1"]
}
- pozycja z „dane” WYCIEKŁA, jeśli KTÓRYKOLWIEK jej fragment jest widoczny
  w tekście po anonimizacji (jako całe słowo/słowa, bez względu na wielkość
  liter i podział linii),
- słowo z „zwykle_slowa” zostało NADMIAROWO zamaskowane, jeśli go nie widać.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Item:
    category: str
    fragments: list[str]
    description: str

    @property
    def label(self) -> str:
        """Unikalna etykieta: fragmenty (+ opis). Sam opis bywa wspólny dla kilku
        pozycji („celownik”) – wtedy liczniki wycieków by się zlewały."""
        text = " / ".join(self.fragments)
        return f"{text} ({self.description})" if self.description else text


@dataclass
class Annotation:
    name: str              # nazwa pliku oznaczeń bez .json
    path: Path             # ścieżka do dokumentu (względem katalogu projektu)
    description: str
    items: list[Item]
    plain_words: list[str]


@dataclass
class DocumentScore:
    leaked: list[Item] = field(default_factory=list)
    hidden: list[Item] = field(default_factory=list)
    kept: list[Item] = field(default_factory=list)      # celowo zostawione (keep)
    over_masked: list[str] = field(default_factory=list)


def load_annotations(folder: Path, root: Path) -> list[Annotation]:
    annotations = []
    for file in sorted(folder.glob("*.json")):
        data = json.loads(file.read_text(encoding="utf-8"))
        annotations.append(Annotation(
            name=file.stem,
            path=root / data["plik"],
            description=data.get("opis", ""),
            items=[Item(d["kategoria"], d["fragmenty"], d.get("opis", "")) for d in data["dane"]],
            plain_words=data.get("zwykle_slowa", []),
        ))
    return annotations


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def visible(fragment: str, text: str) -> bool:
    """Czy fragment widać w tekście jako całe słowo/słowa.

    „Grzegorz” NIE jest widoczny w „Grzegorza”, a „Lis” nie w „Lisowski” –
    dzięki temu pozycje się nie dublują. Złamanie linii = spacja.
    """
    pattern = r"(?<!\w)" + re.escape(_normalize(fragment)) + r"(?!\w)"
    return re.search(pattern, _normalize(text)) is not None


def score_document(annotation: Annotation, masked_text: str,
                   kept_categories: set[str] = frozenset()) -> DocumentScore:
    score = DocumentScore()
    for item in annotation.items:
        if item.category in kept_categories:
            score.kept.append(item)
        elif any(visible(fragment, masked_text) for fragment in item.fragments):
            score.leaked.append(item)
        else:
            score.hidden.append(item)
    score.over_masked = [w for w in annotation.plain_words if not visible(w, masked_text)]
    return score


def missing_fragments(annotation: Annotation, source_text: str) -> list[str]:
    """Fragmenty oznaczeń, których NIE MA w tekście źródłowym (literówka w oznaczeniu
    albo zmiana dokumentu/OCR). Takie oznaczenie zawsze „zostałoby ukryte”."""
    fragments = [f for item in annotation.items for f in item.fragments] + annotation.plain_words
    return [f for f in fragments if not visible(f, source_text)]
