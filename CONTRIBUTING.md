# 🤝 Jak współtworzyć projekt

Dzięki, że chcesz pomóc! Ten dokument opisuje, jak przygotować środowisko, czego pilnujemy
w kodzie i jak wygląda droga od pomysłu do scalonego PR-a.

Lukę bezpieczeństwa (np. wyciek danych) zgłoś prywatnie – patrz [SECURITY.md](SECURITY.md).

## Zasady, których się trzymamy

1. **Prawdziwe dane nigdy nie opuszczają komputera.** Żadnych wywołań do usług w chmurze,
   telemetrii ani zapisu danych z dokumentów na dysk. Adres Ollamy musi pozostać lokalny.
2. **Mały LLM zostaje w pipeline.** To założenie projektu (hackathon) – zmiany mają go
   wspierać, a nie zastępować.
3. **Limit parametrów: LLM + NER ≤ 4 mld** (dziś qwen2.5:3b 3,09 mld + GLiNER 0,31 mld = 3,39 mld).
   Nowy model zgłaszaj z liczbą parametrów; aplikacja pokazuje sumę w pasku bocznym.
4. **Fail-closed.** Jeśli etap nie zadziałał albo wynik jest podejrzany, lepiej wstrzymać wynik
   do ręcznego sprawdzenia, niż wypuścić niezamaskowane dane.
5. **Model nie przepisuje tekstu.** LLM i NER zwracają tylko fragmenty, a podmianę robi Python.
6. **Tylko fikcyjne dane** w testach, przykładach, oznaczeniach `eval/` i zgłoszeniach.
   PESEL-e, NIP-y i IBAN-y generuj z poprawną sumą kontrolną, ale nigdy nie kopiuj prawdziwych.

## Przygotowanie środowiska

```bash
git clone <adres twojego forka>
cd Anonimizator-PII
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ner.txt    # opcjonalnie: NER (PyTorch CPU, GLiNER2, peft)
python training/scal_nakladke.py       # opcjonalnie: polski GLiNER v3 (~1,2 GB w modele/)
ollama pull qwen2.5:3b
```

OCR wymaga modeli Tesseracta `pol` i `eng` – instrukcja w [README](README.md#️-ocr-tylko-na-obrazach).

## Testy

```bash
pytest -m "not slow"     # szybko, bez Ollamy i bez prawdziwego NER
pytest                   # pełny zestaw (testy „slow” ładują prawdziwy model GLiNER)
```

- Testy **nie mogą** wołać prawdziwej Ollamy – `tests/conftest.py` blokuje takie wywołania.
  LLM podmieniaj atrapą (`monkeypatch` na `requests.post` albo na funkcji w module, który ją woła).
- Każda poprawka błędu dostaje test, który bez niej nie przechodzi.
- Nowy wzorzec Regex: test, że łapie poprawną wartość (także rozbitą spacjami, jak robi to OCR),
  i test, że **nie** łapie wartości podobnej, ale błędnej (zła suma kontrolna, numer faktury itp.).

## Pomiar jakości (`eval/`)

Zmiana, która wpływa na to, co jest maskowane (Regex, NER, prompt, progi, model), wymaga pomiaru:

```bash
python eval/run_eval.py --bez-llm              # szybko: Regex + NER
python eval/run_eval.py --przebiegi 2          # z LLM; wynik modelu 3B bywa różny między przebiegami
```

W opisie PR-a podaj wynik **przed i po** (ukryte dane, nadmiar). Raport z `eval/wyniki/`
commituj tylko wtedy, gdy jest nowym punktem odniesienia – przypadkowe raporty z prób usuń.
Nowy dokument testowy: PDF w `przyklady/`, oznaczenia w `eval/oznaczenia/` i `pytest tests/test_eval.py`
(sprawdza, czy oznaczenia zgadzają się z tekstem odczytanym z pliku). Danych z `eval/oznaczenia/`
nie wolno używać w treningu modeli (`training/`).

## Styl kodu

- Python 3.12+, nazwy i struktura po angielsku, **komentarze, docstringi i komunikaty po polsku**.
- Komentarz wyjaśnia **dlaczego** (zwykle z pomiarem albo przykładem, który się psuł), a nie co robi kod.
- Ciężkie biblioteki (PyTorch, gliner2) importuj leniwie, żeby aplikacja działała bez nich.
- Nie dodawaj zależności bez potrzeby. Nowa zależność trafia do `requirements.txt`
  (albo `requirements-ner.txt`, jeśli jest potrzebna tylko do NER i treningu) z dolną
  granicą wersji lub dokładnym numerem.

## Commity i pull requesty

1. Zrób fork i gałąź od `main` o opisowej nazwie, np. `fix/pesel-ocr`, `ner-progi`.
2. Commity po polsku, w trybie rozkazującym albo jako krótki opis zmiany:
   „Regex odporny na spacje wstawione przez OCR w numerach”. W treści – co i dlaczego.
3. Jeden PR = jedna sprawa. Refaktoryzację oddziel od zmiany zachowania.
4. Przed wysłaniem: `pytest -m "not slow"` przechodzi, a przy zmianach w wykrywaniu – wynik `eval/` w opisie.
5. Zmiana zachowania widoczna dla użytkownika (nowa kategoria, przełącznik, model) = aktualizacja README.
6. Nie commituj: modeli (`modele/`), danych treningowych (`training/dane/`), `tessdata/`,
   plików `.env` ani niczego z prawdziwymi danymi.

## Licencja

Wysyłając PR, zgadzasz się, że Twój wkład będzie udostępniony na licencji MIT, jak reszta
projektu ([LICENSE](LICENSE)).
