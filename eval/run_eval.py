"""Pomiar jakości anonimizacji na dokumentach z ręcznymi oznaczeniami.

Przykłady (z katalogu projektu):
    python eval/run_eval.py                          # Regex + NER + qwen2.5:3b, 1 przebieg
    python eval/run_eval.py --przebiegi 3            # powtarzalność LLM
    python eval/run_eval.py --etapy                  # który etap ukrywa które dane
    python eval/run_eval.py --model qwen3.5:2b --pliki 5 6
    python eval/run_eval.py --bez-llm                # sam Regex + NER (bez Ollamy)
    python eval/run_eval.py --zostaw ZDROWIE         # jak przełącznik „zostaw jawnie”

Wynik: podsumowanie w konsoli i raport Markdown + JSON w eval/wyniki/.
Wyciek = fragment oznaczonej danej widoczny po anonimizacji (szczegóły: scoring.py).
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import ner_masker  # noqa: E402
from engine.file_reader import join_pages, read_uploaded_pages  # noqa: E402
from engine.llm_masker import DEFAULT_MODEL  # noqa: E402
from engine.pipeline import anonymize  # noqa: E402
from engine.vault import Vault  # noqa: E402
from scoring import load_annotations, missing_fragments, score_document, visible  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model w Ollamie")
    parser.add_argument("--przebiegi", type=int, default=1,
                        help="ile razy uruchomić każdy dokument (LLM nie jest w pełni powtarzalny)")
    parser.add_argument("--pliki", nargs="*", default=[],
                        help="prefiksy nazw oznaczeń, np. 5 6 (domyślnie wszystkie)")
    parser.add_argument("--bez-ner", action="store_true", help="wyłącz etap NER")
    parser.add_argument("--bez-llm", action="store_true", help="wyłącz etap LLM")
    parser.add_argument("--zostaw", nargs="*", default=[], help="kategorie zostawiane jawnie")
    parser.add_argument("--etapy", action="store_true",
                        help="dodatkowo: czy daną ukrył już Regex (R), NER (N) czy dopiero LLM (L)")
    parser.add_argument("--bez-raportu", action="store_true", help="nie zapisuj plików w eval/wyniki")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    use_ner = not args.bez_ner and ner_masker.ner_installed()
    use_llm = not args.bez_llm
    keep = set(args.zostaw)
    annotations = [a for a in load_annotations(ROOT / "eval" / "oznaczenia", ROOT)
                   if not args.pliki or any(a.name.startswith(p) for p in args.pliki)]
    if use_ner:
        ner_masker.load_model()

    config = (f"model={args.model if use_llm else '—'} · NER={'tak' if use_ner else 'nie'} · "
              f"LLM={'tak' if use_llm else 'nie'} · przebiegi={args.przebiegi}"
              + (f" · zostaw={','.join(sorted(keep))}" if keep else ""))
    print(f"Konfiguracja: {config}\n")

    documents = []
    category_total: Counter = Counter()
    category_leaked: Counter = Counter()
    for annotation in annotations:
        text = join_pages(read_uploaded_pages(annotation.path.name, annotation.path.read_bytes()))
        missing = missing_fragments(annotation, text)
        if missing:
            print(f"⚠️  {annotation.name}: fragmentów oznaczeń nie ma w dokumencie: {missing}")

        leak_count: Counter = Counter()
        over_count: Counter = Counter()
        times, reviews = [], 0
        for _ in range(args.przebiegi):
            start = time.perf_counter()
            result = anonymize(text, Vault(), use_llm=use_llm, model=args.model,
                               use_ner=use_ner, keep_categories=keep)
            times.append(time.perf_counter() - start)
            reviews += bool(result.needs_review)
            last = score_document(annotation, result.masked_text, keep)
            leak_count.update(item.label for item in last.leaked)
            over_count.update(last.over_masked)
            for item in last.leaked + last.hidden:
                category_total[item.category] += 1
            for item in last.leaked:
                category_leaked[item.category] += 1

        stages = {}
        if args.etapy:
            # Regex i Regex+NER są deterministyczne – wystarczy jeden przebieg.
            only_regex = anonymize(text, Vault(), use_llm=False, keep_categories=keep).masked_text
            regex_ner = (anonymize(text, Vault(), use_llm=False, use_ner=True,
                                   keep_categories=keep).masked_text if use_ner else only_regex)
            for item in annotation.items:
                if item.category in keep:
                    continue
                if not score_document_item_visible(item, only_regex):
                    stages[item.label] = "R"
                elif not score_document_item_visible(item, regex_ner):
                    stages[item.label] = "N"
                elif leak_count[item.label] < args.przebiegi:
                    stages[item.label] = "L"
                else:
                    stages[item.label] = "✗"

        countable = [i for i in annotation.items if i.category not in keep]
        leaked_avg = sum(leak_count.values()) / args.przebiegi
        documents.append({
            "nazwa": annotation.name,
            "opis": annotation.description,
            "danych": len(countable),
            "wycieki_srednio": round(leaked_avg, 2),
            "wycieki": {label: n for label, n in leak_count.most_common()},
            "nadmiar": {word: n for word, n in over_count.most_common()},
            "zostawione": [i.label for i in annotation.items if i.category in keep],
            "czas_s": round(sum(times) / len(times), 1),
            "do_sprawdzenia": reviews,
            "etapy": stages,
            "brakujace_fragmenty": missing,
        })
        print(f"{annotation.name[:40]:40} wycieki {leaked_avg:4.1f}/{len(countable):<3} "
              f"nadmiar {sum(over_count.values()) / args.przebiegi:3.1f}/"
              f"{len(annotation.plain_words):<2} czas {sum(times) / len(times):5.1f} s"
              + (f"  ⛔ do sprawdzenia: {reviews}/{args.przebiegi}" if reviews else ""))
        for label, n in leak_count.most_common():
            print(f"    ✗ {label}" + (f"  ({n}/{args.przebiegi} przebiegów)" if args.przebiegi > 1 else ""))

    total = sum(category_total.values())
    leaked = sum(category_leaked.values())
    print(f"\nRAZEM: ukryto {total - leaked}/{total} "
          f"({100 * (total - leaked) / max(total, 1):.1f}%) danych we wszystkich przebiegach")
    per_category = {
        cat: {"ukryto": category_total[cat] - category_leaked[cat], "razem": category_total[cat]}
        for cat in sorted(category_total)
    }
    print("Per kategoria: " + ", ".join(
        f"{cat} {v['ukryto']}/{v['razem']}" for cat, v in per_category.items()
        if v["ukryto"] < v["razem"]) + " (pozostałe 100%)")

    if not args.bez_raportu:
        write_report(args, config, documents, per_category, total, leaked)
    return 0


def score_document_item_visible(item, text: str) -> bool:
    return any(visible(fragment, text) for fragment in item.fragments)


def write_report(args, config, documents, per_category, total, leaked) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    model = (args.model if not args.bez_llm else "bez-llm").replace(":", "_")
    # Bez with_suffix(): „qwen2.5_3b” ma kropkę i with_suffix ucinał „.5_3b”.
    md_path = ROOT / "eval" / "wyniki" / f"{stamp}_{model}.md"
    json_path = md_path.with_name(f"{stamp}_{model}.json")
    lines = [
        f"# Pomiar anonimizacji – {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"**Konfiguracja:** {config}",
        "",
        f"**Ukryto {total - leaked}/{total} danych ({100 * (total - leaked) / max(total, 1):.1f}%)** "
        "– suma po wszystkich dokumentach i przebiegach.",
        "",
        "| Dokument | Wycieki (średnio) | Nadmiar | Czas | Do sprawdzenia |",
        "|---|---|---|---|---|",
    ]
    for doc in documents:
        over = sum(doc["nadmiar"].values()) / args.przebiegi
        lines.append(f"| {doc['nazwa']} | {doc['wycieki_srednio']}/{doc['danych']} | {over:.1f} | "
                     f"{doc['czas_s']} s | {doc['do_sprawdzenia']}/{args.przebiegi} |")
    lines += ["", "## Per kategoria", "", "| Kategoria | Ukryto |", "|---|---|"]
    lines += [f"| {cat} | {v['ukryto']}/{v['razem']} |" for cat, v in per_category.items()]
    lines += ["", "## Szczegóły", ""]
    for doc in documents:
        lines.append(f"### {doc['nazwa']}")
        lines.append(f"_{doc['opis']}_")
        lines.append("")
        if doc["wycieki"]:
            lines += [f"- ✗ wyciek: {label}" + (f" ({n}/{args.przebiegi})" if args.przebiegi > 1 else "")
                      for label, n in doc["wycieki"].items()]
        if doc["nadmiar"]:
            lines += [f"- ⚠️ nadmiarowo zamaskowane: „{word}”" for word in doc["nadmiar"]]
        if doc["zostawione"]:
            lines.append("- 👁️ zostawione jawnie: " + ", ".join(doc["zostawione"]))
        if doc["etapy"]:
            counts = Counter(doc["etapy"].values())
            lines.append(f"- etapy: Regex {counts['R']}, NER {counts['N']}, LLM {counts['L']}, "
                         f"wyciek {counts['✗']}")
        if not (doc["wycieki"] or doc["nadmiar"]):
            lines.append("- bez wycieków i nadmiarowego maskowania")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(
        {"konfiguracja": config, "dokumenty": documents, "kategorie": per_category},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nRaport: {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
