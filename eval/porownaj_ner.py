"""Porównanie dwóch pomiarów GLiNER-a (np. przed i po treningu) – tabela + wykres.

Najpierw dwa pomiary:
    ANONIMIZATOR_NER_MODEL=fastino/gliner2-privacy-filter-PII-multi python eval/szara_strefa.py --nazwa przed
    ANONIMIZATOR_NER_MODEL=modele/gliner-pl python eval/szara_strefa.py --nazwa po
Potem:
    python eval/porownaj_ner.py                      # przed vs po
    python eval/porownaj_ner.py --a przed --b po2    # dowolne dwa pomiary

Wynik: eval/wyniki/porownanie_<a>_<b>.png
Autor: Wiktor Jarocki
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WYNIKI = ROOT / "eval" / "wyniki"
KATEGORIE = ["OSOBA", "ADRES", "FIRMA", "LOGIN"]


def wczytaj(nazwa):
    plik = WYNIKI / f"szara_strefa_{nazwa}.json"
    if not plik.exists():
        raise SystemExit(f"Brak {plik.relative_to(ROOT)} – uruchom: python eval/szara_strefa.py --nazwa {nazwa}")
    return json.loads(plik.read_text(encoding="utf-8"))


def wykryte(pomiar, kategoria=None):
    """(ile prawdziwych danych GLiNER maskuje od razu, ile wszystkich) – w danej kategorii."""
    dane = [x for x in pomiar["prawdziwe"] if kategoria is None or x["kategoria"] == kategoria]
    ok = sum(1 for x in dane if x["pewnosc"] is not None and x["pewnosc"] >= pomiar["gora"])
    return ok, len(dane)


def falszywe_maskowane(pomiar):
    return sum(1 for x in pomiar["falszywe"] if x["pewnosc"] >= pomiar["gora"])


def zmiany(a, b):
    """Dane, które zmieniły strefę między pomiarami."""
    po = {(x["dokument"], x["dana"]): x["pewnosc"] for x in b["prawdziwe"]}
    for x in a["prawdziwe"]:
        p1, p2 = x["pewnosc"], po.get((x["dokument"], x["dana"]))
        przed_ok = p1 is not None and p1 >= a["gora"]
        po_ok = p2 is not None and p2 >= b["gora"]
        if przed_ok != po_ok:
            yield ("✅ naprawione" if po_ok else "❌ pogorszone"), x["dana"], p1, p2


def main():
    parser = argparse.ArgumentParser(description="Porównanie dwóch pomiarów GLiNER-a")
    parser.add_argument("--a", default="przed")
    parser.add_argument("--b", default="po")
    args = parser.parse_args()
    a, b = wczytaj(args.a), wczytaj(args.b)

    print(f"{'':10}{args.a:>12}{args.b:>12}")
    for kat in KATEGORIE + [None]:
        (ok1, n1), (ok2, n2) = wykryte(a, kat), wykryte(b, kat)
        if n1:
            print(f"{kat or 'RAZEM':10}{f'{ok1}/{n1}':>12}{f'{ok2}/{n2}':>12}")
    print(f"{'fałszywe':10}{falszywe_maskowane(a):>12}{falszywe_maskowane(b):>12}   (maskowane zwykłe słowa – mniej = lepiej)")

    print("\nZmiany:")
    lista = list(zmiany(a, b))
    for znak, dana, p1, p2 in lista:
        f = lambda p: "brak" if p is None else f"{p:.2f}"
        print(f"  {znak}  {dana}   {f(p1)} → {f(p2)}")
    if not lista:
        print("  brak")

    narysuj(a, b, args.a, args.b, WYNIKI / f"porownanie_{args.a}_{args.b}.png")


def narysuj(a, b, nazwa_a, nazwa_b, plik):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    szary, niebieski = "#a9a7a0", "#2a78d6"
    tusz, szary_tekst, siatka, tlo = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

    kategorie = [k for k in KATEGORIE if wykryte(a, k)[1]] + ["RAZEM"]
    def procent(pomiar, kat):
        ok, n = wykryte(pomiar, None if kat == "RAZEM" else kat)
        return 100 * ok / n, f"{ok}/{n}"

    fig, (lewy, prawy) = plt.subplots(1, 2, figsize=(12, 6), dpi=150,
                                      gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor(tlo)
    x = range(len(kategorie))
    szer = 0.38
    for przesuniecie, pomiar, kolor, etykieta in ((-szer / 2, a, szary, nazwa_a),
                                                   (szer / 2, b, niebieski, nazwa_b)):
        wartosci = [procent(pomiar, k) for k in kategorie]
        slupki = lewy.bar([i + przesuniecie for i in x], [w[0] for w in wartosci],
                          width=szer - 0.03, color=kolor, label=etykieta, zorder=2)
        for slupek, (_, opis) in zip(slupki, wartosci):
            lewy.text(slupek.get_x() + slupek.get_width() / 2, slupek.get_height() + 1.5, opis,
                      ha="center", va="bottom", fontsize=9, color=szary_tekst)
    lewy.set_xticks(list(x), kategorie)
    lewy.set_ylim(0, 112)
    lewy.set_ylabel("% prawdziwych danych zamaskowanych przez GLiNER", fontsize=10, color=tusz)
    lewy.set_title("Prawdziwe dane wykryte pewnie (więcej = lepiej)", loc="left",
                   fontsize=11, color=tusz, fontweight="bold")
    lewy.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, fontsize=10)

    f1, f2 = falszywe_maskowane(a), falszywe_maskowane(b)
    slupki = prawy.bar([0, 1], [f1, f2], width=0.6, color=[szary, niebieski], zorder=2)
    for slupek, w in zip(slupki, (f1, f2)):
        prawy.text(slupek.get_x() + slupek.get_width() / 2, w + 0.2, str(w),
                   ha="center", va="bottom", fontsize=10, color=szary_tekst)
    prawy.set_xticks([0, 1], [nazwa_a, nazwa_b])
    prawy.set_ylim(0, max(f1, f2, 1) * 1.3)
    prawy.set_title("Zwykłe słowa zamaskowane\n(mniej = lepiej)", loc="left",
                    fontsize=11, color=tusz, fontweight="bold")

    for ax in (lewy, prawy):
        ax.set_facecolor(tlo)
        ax.grid(axis="y", color=siatka, linewidth=0.8, zorder=0)
        for bok in ("top", "right"):
            ax.spines[bok].set_visible(False)
        for bok in ("left", "bottom"):
            ax.spines[bok].set_color(szary_tekst)
        ax.tick_params(colors=szary_tekst)

    fig.suptitle(f"GLiNER {nazwa_a} vs {nazwa_b}: 8 dokumentów testowych, próg {b['gora']}",
                 x=0.04, ha="left", fontsize=15, fontweight="bold", color=tusz)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(plik, facecolor=tlo)
    plt.close(fig)
    print(f"\nWykres: {plik.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
