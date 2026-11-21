"""Scalanie nakładki LoRA z oryginalnym GLiNER-em – gotowy polski model bez trenowania.

Pełny model ma 1,2 GB (za dużo do gita), a nakładka LoRA z treningu tylko ~10 MB.
Ten skrypt pobiera oryginalny model z Hugging Face, nakłada na niego nakładkę
z repo i zapisuje scalony model, który aplikacja wczytuje przez ANONIMIZATOR_NER_MODEL.

Uruchom z katalogu projektu (raz):
    python training/scal_nakladke.py
    ANONIMIZATOR_NER_MODEL=modele/gliner-pl-v3 streamlit run app.py
Autor: Wiktor Jarocki
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NAKLADKA = ROOT / "nakladki" / "gliner-pl-v3"   # w repo (~10 MB)
WYNIK = ROOT / "modele" / "gliner-pl-v3"        # poza gitem (1,2 GB)


def sciezka(p):
    """Ścieżka względem projektu, jeśli się da (krótsza w komunikatach)."""
    return p.relative_to(ROOT) if p.is_relative_to(ROOT) else p


def scal(nakladka, wyjscie):
    """Oryginalny model + nakładka LoRA -> scalony model zapisany w katalogu `wyjscie`."""
    from gliner2 import GLiNER2
    from peft import PeftModel

    # Model bazowy bierzemy z konfiguracji nakładki, NIE z ANONIMIZATOR_NER_MODEL –
    # ta zmienna może już wskazywać na model dotrenowany.
    baza_nazwa = json.loads((Path(nakladka) / "adapter_config.json").read_text())["base_model_name_or_path"]
    baza = GLiNER2.from_pretrained(baza_nazwa)
    scalony = PeftModel.from_pretrained(baza, str(nakladka)).merge_and_unload()
    scalony.save_pretrained(str(wyjscie))


def main():
    parser = argparse.ArgumentParser(description="Scalanie nakładki LoRA z GLiNER-em")
    parser.add_argument("--nakladka", type=Path, default=NAKLADKA, help="katalog z adapter_config.json")
    parser.add_argument("--wyjscie", type=Path, default=WYNIK, help="gdzie zapisać scalony model")
    args = parser.parse_args()
    nakladka = args.nakladka if args.nakladka.is_absolute() else ROOT / args.nakladka
    wyjscie = args.wyjscie if args.wyjscie.is_absolute() else ROOT / args.wyjscie

    if not (nakladka / "adapter_config.json").exists():
        sys.exit(f"Brak nakładki w {nakladka}")
    print(f"Scalam {sciezka(nakladka)} z oryginalnym modelem (pierwsze pobranie ~1,2 GB)…")
    scal(nakladka, wyjscie)
    print(f"\nModel zapisany w {sciezka(wyjscie)}. Uruchomienie aplikacji z nim:")
    print(f"  ANONIMIZATOR_NER_MODEL={sciezka(wyjscie)} streamlit run app.py")


if __name__ == "__main__":
    main()
