# 📏 Pomiar jakości anonimizacji

Ręcznie oznaczone dane w przykładowych PDF-ach (`przyklady/1–8`) i skrypt, który sprawdza,
ile z nich zostało ukrytych. Pozwala porównywać modele, progi i reguły na tych samych danych
zamiast „na oko”.

```bash
python eval/run_eval.py                          # Regex + NER + qwen2.5:3b, 1 przebieg
python eval/run_eval.py --przebiegi 3            # powtarzalność LLM (wynik bywa różny)
python eval/run_eval.py --etapy                  # który etap ukrywa które dane: R / N / L
python eval/run_eval.py --model qwen3.5:2b       # inny model (sprawdź limit 4 mld!)
python eval/run_eval.py --bez-llm --pliki 1 2    # szybko, bez Ollamy, wybrane pliki
python eval/run_eval.py --zostaw ZDROWIE         # jak przełącznik „zostaw jawnie”
```

Wynik: podsumowanie w konsoli oraz raport `eval/wyniki/<data>_<model>.md` (+ `.json`).

## Zestaw kontrolny (`--zestaw kontrolny`)

Na `przyklady/1–8` stroiliśmy reguły, progi i trening GLiNER-a, więc wynik na nich jest optymistyczny.
Zestaw kontrolny ([`przyklady/kontrolne/`](../przyklady/kontrolne), oznaczenia w `oznaczenia_kontrolne/`)
służy **tylko do pomiaru**: 4 dokumenty (wyciąg bankowy ze znakiem wodnym, wniosek `.docx` ze zdjęciem
dowodu, skan wypisu ze szpitala z podpisem, potoczny protokół zebrania wspólnoty), 47 oznaczonych danych.

- Żadne słowo z jego osób i adresów nie występuje w zestawie podstawowym (pilnuje tego test),
  a `training/gliner_dane.py` wyklucza je z danych treningowych.
- Dokumentów kontrolnych **nie poprawiamy pod wynik**. Gdy znajdziemy błąd, poprawiamy aplikację
  i dodajemy podobny przypadek do zestawu podstawowego, a kontrolny zostaje bez zmian.
  Nowe przypadki kontrolne dokładamy jako kolejne pliki (`python przyklady/kontrolne/generuj_kontrolne.py`).

## Co liczymy

- **Wyciek** – którykolwiek fragment oznaczonej danej jest widoczny po anonimizacji jako całe
  słowo (bez względu na wielkość liter i złamania linii). Samo nazwisko bez imienia to też wyciek.
- **Nadmiar** – zwykłe słowo z listy `zwykle_slowa` zostało zamaskowane (np. „Najemcy”, „§1”).
  Nie szkodzi prywatności, ale psuje tekst wysyłany do AI.
- **Etapy** (`--etapy`) – R: ukrył już Regex, N: dopiero NER, L: dopiero LLM, ✗: nikt.
- **Do sprawdzenia** – ile przebiegów zakończyło się wstrzymaniem wyniku (fail-closed),
  np. przez prompt injection w pliku 4.

## Format oznaczeń (`oznaczenia/*.json`)

```json
{
  "plik": "przyklady/1_umowa_z_pieczatka_i_podpisami.pdf",
  "opis": "Umowa najmu: tekst + pieczątka (OCR)",
  "dane": [
    {"kategoria": "OSOBA", "fragmenty": ["Grzegorza", "Wróblewskiego"], "opis": "prezes w dopełniaczu"},
    {"kategoria": "PESEL", "fragmenty": ["87062364800"]}
  ],
  "zwykle_slowa": ["Najemcy", "§1", "netto"]
}
```

Fragmenty muszą pochodzić z tekstu, który aplikacja **naprawdę odczytuje** (z OCR włącznie –
np. PESEL ze skanu to „561 10446533”). Pilnuje tego test `tests/test_eval.py`: literówka
w oznaczeniu sprawiłaby, że dana zawsze liczy się jako ukryta i zawyża wynik.

Nowy dokument: wrzuć PDF do `przyklady/`, dodaj plik oznaczeń i uruchom `pytest tests/test_eval.py`.
