"""Szara strefa GLiNER-a: jak pewny jest NER przy prawdziwych danych, a jak przy zwykłych słowach.

Pomysł: tam, gdzie GLiNER się waha (szara strefa pewności), pomaga mu LLM.
Ten skrypt mierzy na 8 dokumentach z przyklady/, ile prawdziwych danych osobowych
i ile fałszywych alarmów wpada do każdej strefy.

Uruchom z katalogu projektu:
    python eval/szara_strefa.py   # model domyślny (modele/gliner-pl-v3, jeśli jest złożony)
    ANONIMIZATOR_NER_MODEL=fastino/gliner2-privacy-filter-PII-multi python eval/szara_strefa.py --nazwa przed
    python eval/szara_strefa.py --dol 0.2 --gora 0.5
    ANONIMIZATOR_NER_MODEL=modele/gliner-pl python eval/szara_strefa.py --nazwa po

Wynik: tabela w konsoli + eval/wyniki/szara_strefa_<nazwa>.png i .json
(porównanie dwóch pomiarów: python eval/porownaj_ner.py)
Autor: Wiktor Jarocki
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from engine import ner_masker  # noqa: E402
from engine.file_reader import normalize_text, read_uploaded_file  # noqa: E402
from engine.llm_masker import split_into_chunks  # noqa: E402
from engine.regex_masker import mask_with_regex  # noqa: E402
from engine.vault import Vault  # noqa: E402
from scoring import load_annotations, visible  # noqa: E402

# Kategorie z klucza odpowiedzi, którymi zajmuje się GLiNER (reszta to Regex albo LLM).
KATEGORIE_GLINERA = {"OSOBA", "FIRMA", "ADRES", "LOGIN"}
# Najniższa pewność, o jaką pytamy GLiNER-a – poniżej tego nic nie zwraca.
MINIMUM = 0.05


# ---------------------------------------------------------------------------
# 1. GLiNER: wszystkie wykrycia z pewnością
# ---------------------------------------------------------------------------

def wykrycia_glinera(model, tekst):
    """Lista (tekst_encji, etykieta, pewność) dla całego dokumentu.

    Długi tekst dzielimy na kawałki, tak jak aplikacja (GLiNER ma limit długości).
    """
    wykrycia = []
    for kawalek in split_into_chunks(tekst, max_chars=ner_masker.NER_CHUNK_CHARS):
        if not kawalek.strip():
            continue
        wynik = model.extract_entities(kawalek, list(ner_masker.LABELS),
                                       threshold=MINIMUM, include_confidence=True)
        for etykieta, lista in wynik["entities"].items():
            for wykrycie in lista:
                wykrycia.append((wykrycie["text"], etykieta, wykrycie["confidence"]))
    return wykrycia


# ---------------------------------------------------------------------------
# 2. Porównanie z kluczem odpowiedzi (eval/oznaczenia/*.json)
# ---------------------------------------------------------------------------

def pasuje(tekst_encji, item):
    """Czy wykrycie GLiNER-a dotyczy tej danej z klucza?

    Tak, jeśli którykolwiek fragment z klucza jest widoczny w wykryciu
    („mietek” w „mietek żuraw”) albo wykrycie jest częścią fragmentu.
    """
    return any(visible(f, tekst_encji) or visible(tekst_encji, f) for f in item.fragments)


def czy_prawdziwe(tekst_encji, adnotacja):
    """True = wykrycie to prawdziwa dana osobowa (dowolnej kategorii z klucza)."""
    return any(pasuje(tekst_encji, item) for item in adnotacja.items)


def strefa(pewnosc, dol, gora):
    if pewnosc is None or pewnosc < dol:
        return "pomiń"
    if pewnosc < gora:
        return "szara"
    return "maskuj"


# ---------------------------------------------------------------------------
# 3. Pomiar na wszystkich dokumentach
# ---------------------------------------------------------------------------

def zmierz(model):
    """Zwraca dwie listy:
    - prawdziwe: (dokument, dana z klucza, najwyższa pewność GLiNER-a albo None, kategoria),
    - falszywe:  (dokument, wykryty tekst, pewność) – wykrycia spoza klucza.
    """
    prawdziwe, falszywe = [], []
    for adnotacja in load_annotations(ROOT / "eval" / "oznaczenia", ROOT):
        print(f"  {adnotacja.path.name} …", flush=True)
        tekst = read_uploaded_file(adnotacja.path.name, adnotacja.path.read_bytes())
        # Tak jak w aplikacji: GLiNER dostaje tekst już po Regexie
        # (PESEL, IBAN, e-mail itd. są zamienione na [PESEL_1]…).
        tekst_po_regexie, _ = mask_with_regex(normalize_text(tekst), Vault())
        wykrycia = wykrycia_glinera(model, tekst_po_regexie)

        # Prawdziwe dane: dla każdej osoby/firmy/adresu z klucza bierzemy
        # najwyższą pewność, z jaką GLiNER ją znalazł (None = wcale).
        for item in adnotacja.items:
            if item.category not in KATEGORIE_GLINERA:
                continue
            # Regex już to ukrył (np. login w e-mailu) – GLiNER tego nie zobaczy, pomijamy.
            if not any(visible(f, tekst_po_regexie) for f in item.fragments):
                continue
            pewnosci = [p for t, _, p in wykrycia if pasuje(t, item)]
            prawdziwe.append((adnotacja.name, item.label, max(pewnosci) if pewnosci else None,
                              item.category))

        # Fałszywe alarmy: wykrycia, które nie pasują do niczego w kluczu.
        for tekst_encji, _, pewnosc in wykrycia:
            if "[" in tekst_encji:          # token z Regexa, np. [EMAIL_1]
                continue
            if not czy_prawdziwe(tekst_encji, adnotacja):
                falszywe.append((adnotacja.name, tekst_encji, pewnosc))
    return prawdziwe, falszywe


# ---------------------------------------------------------------------------
# 4. Tabela w konsoli
# ---------------------------------------------------------------------------

def policz(pewnosci, dol, gora):
    strefy = [strefa(p, dol, gora) for p in pewnosci]
    return {s: strefy.count(s) for s in ("maskuj", "szara", "pomiń")}


def wypisz(prawdziwe, falszywe, dol, gora):
    p = policz([x[2] for x in prawdziwe], dol, gora)
    f = policz([x[2] for x in falszywe], dol, gora)
    print(f"\nSzara strefa: {dol:.2f} ≤ pewność < {gora:.2f}\n")
    print(f"{'':26}{'maskuj':>10}{'szara→LLM':>12}{'pomiń':>10}")
    print(f"{'Prawdziwe dane osobowe':26}{p['maskuj']:>10}{p['szara']:>12}{p['pomiń']:>10}")
    print(f"{'Fałszywe alarmy':26}{f['maskuj']:>10}{f['szara']:>12}{f['pomiń']:>10}")

    print("\nPrawdziwe dane w szarej strefie (tu LLM może pomóc):")
    for dok, dana, pew, _ in sorted(prawdziwe, key=lambda x: x[2] or 0):
        if strefa(pew, dol, gora) == "szara":
            print(f"  {pew:.2f}  [{dok[:1]}] {dana}")

    print(f"\nPrawdziwe dane, których GLiNER prawie nie widzi (< {dol}) – kaskada nie pomoże:")
    for dok, dana, pew, _ in prawdziwe:
        if strefa(pew, dol, gora) == "pomiń":
            opis = "brak" if pew is None else f"{pew:.2f}"
            print(f"  {opis:>4}  [{dok[:1]}] {dana}")


# ---------------------------------------------------------------------------
# 5. Wykres
# ---------------------------------------------------------------------------

def narysuj(prawdziwe, falszywe, dol, gora, plik, nazwa):
    import matplotlib
    matplotlib.use("Agg")                    # rysuj do pliku, bez okna
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    niebieski, pomaranczowy = "#2a78d6", "#eb6834"
    tusz, szary_tekst, siatka, tlo = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

    # Histogram: ile wykryć w każdym przedziale pewności (co 0.05).
    szer = 0.05
    srodki = [i * szer + szer / 2 for i in range(20)]

    def histogram(pewnosci):
        licznik = [0] * 20
        for p in pewnosci:
            licznik[min(int(p / szer), 19)] += 1
        return licznik

    h_prawdziwe = histogram([x[2] for x in prawdziwe if x[2] is not None])
    h_falszywe = histogram([p for _, _, p in falszywe])
    niewidoczne = sum(1 for x in prawdziwe if x[2] is None)

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    fig.patch.set_facecolor(tlo)
    ax.set_facecolor(tlo)
    wys = max(h_prawdziwe + h_falszywe + [1]) * 1.3

    # Strefy w tle
    ax.axvspan(dol, gora, color="#d9d8d3", alpha=0.6, zorder=0, linewidth=0)
    ax.axvline(gora, color=szary_tekst, linewidth=1.2, linestyle="--", zorder=1)
    for x, napis in ((dol / 2, "pomiń"), ((dol + gora) / 2, "szara strefa\n→ pyta LLM"),
                     ((gora + 1) / 2, "maskuj od razu")):
        ax.text(x, wys * 0.95, napis, ha="center", va="top", fontsize=11,
                color=tusz, fontweight="bold")

    # Słupki: prawdziwe w górę, fałszywe w dół
    ax.bar(srodki, h_prawdziwe, width=szer - 0.008, color=niebieski, zorder=2,
           label="prawdziwe dane osobowe")
    ax.bar(srodki, [-x for x in h_falszywe], width=szer - 0.008, color=pomaranczowy,
           zorder=2, label="fałszywe alarmy (zwykłe słowa)")
    ax.axhline(0, color=tusz, linewidth=0.8, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(-wys, wys)
    ax.set_xlabel("Pewność GLiNER-a", fontsize=11, color=tusz)
    ax.set_ylabel("Liczba wykryć", fontsize=11, color=tusz)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(int(v))}"))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color=siatka, linewidth=0.8, zorder=0)
    for bok in ("top", "right"):
        ax.spines[bok].set_visible(False)
    for bok in ("left", "bottom"):
        ax.spines[bok].set_color(szary_tekst)
    ax.tick_params(colors=szary_tekst)

    szare_p = sum(1 for x in prawdziwe if x[2] is not None and dol <= x[2] < gora)
    szare_f = sum(1 for _, _, p in falszywe if dol <= p < gora)
    fig.suptitle(f"Gdzie GLiNER się waha? Pewność wykryć na 8 dokumentach testowych ({nazwa})",
                 x=0.06, ha="left", fontsize=15, fontweight="bold", color=tusz)
    ax.set_title(f"Szara strefa {dol:.2f}–{gora:.2f}: do LLM trafiłoby {szare_p} prawdziwych "
                 f"danych i {szare_f} fałszywych alarmów.  GLiNER nie widzi wcale: "
                 f"{niewidoczne} z {len(prawdziwe)} osób, firm i adresów.",
                 loc="left", fontsize=10, color=szary_tekst, pad=12)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    plik.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plik, facecolor=tlo)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Szara strefa GLiNER-a")
    parser.add_argument("--dol", type=float, default=0.2, help="dolna granica szarej strefy")
    parser.add_argument("--gora", type=float, default=0.5, help="górna granica (próg maskowania)")
    parser.add_argument("--nazwa", default="przed", help="nazwa pomiaru w plikach wyników, np. przed / po")
    args = parser.parse_args()

    print(f"Wczytuję GLiNER-a: {ner_masker.NER_MODEL}")
    model = ner_masker.load_model()
    print("Mierzę:")
    prawdziwe, falszywe = zmierz(model)
    wypisz(prawdziwe, falszywe, args.dol, args.gora)

    wyniki = ROOT / "eval" / "wyniki"
    plik = wyniki / f"szara_strefa_{args.nazwa}.png"
    narysuj(prawdziwe, falszywe, args.dol, args.gora, plik, args.nazwa)
    dane = {"nazwa": args.nazwa, "model": ner_masker.NER_MODEL, "dol": args.dol, "gora": args.gora,
            "prawdziwe": [dict(zip(("dokument", "dana", "pewnosc", "kategoria"), x)) for x in prawdziwe],
            "falszywe": [dict(zip(("dokument", "tekst", "pewnosc"), x)) for x in falszywe]}
    (wyniki / f"szara_strefa_{args.nazwa}.json").write_text(
        json.dumps(dane, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWykres: {plik.relative_to(ROOT)}  (dane: szara_strefa_{args.nazwa}.json)")


if __name__ == "__main__":
    main()
