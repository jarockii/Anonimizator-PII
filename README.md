# 🛡️ Lokalny Anonimizator PII

Lokalny system anonimizacji i de-anonimizacji danych wrażliwych.
Tekst z danymi osobowymi zamieniamy na tokeny (`[OSOBA_1]`, `[PESEL_1]`…),
wysyłamy do dowolnego AI w chmurze, a odpowiedź automatycznie „odmaskowujemy”.
**Prawdziwe dane nigdy nie opuszczają komputera.**

## Uruchomienie

```bash
# 1. Model językowy (jednorazowo, ~1,9 GB)
ollama pull qwen2.5:3b

# 2. Środowisko Pythona
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Testy
pytest -v

# 4. Aplikacja (http://localhost:8501, dostępna tylko z tego komputera)
streamlit run app.py --server.address localhost
```

Bez działającej Ollamy aplikacja nadal działa: włącza się tryb samego Regexa i pokazuje ostrzeżenie.

## 🧠 Etap NER (GLiNER2-PII) – opcjonalny

Między Regexem a LLM-em działa model NER [GLiNER2-PII](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
(Apache-2.0, ok. 0,3 mld parametrów, CPU). Zaznacza osoby, firmy, adresy i loginy w ułamku sekundy,
a mały LLM dostaje tekst już po nim i szuka tego, co wymaga kontekstu (stanowiska, odmienione formy).
NER nie generuje tekstu, więc nie halucynuje i nie da się go „przekonać” poleceniem w dokumencie.

```bash
pip install -r requirements-ner.txt   # PyTorch CPU + gliner2; model ~1,2 GB pobierze się raz
```

Pomiar na `przyklady/*.pdf` (Regex + NER + qwen2.5:3b): NER 0,16–0,47 s na dokument, LLM 5,7–14,1 s,
**0 wycieków i 0 nadmiarowo zamaskowanych słów** na sprawdzanej liście. NER znalazł wszystkie osoby
we wszystkich plikach (pewność 0,86–1,0) i login `tnowak` z przypadku „Log z błędem”, którego model 3B nie łapał.
Koszt: **~1,4–1,7 GB RAM** (wagi float32). Kwantyzacja int8 psuła model (0 encji), a bfloat16 była 15× wolniejsza
na CPU – dlatego zostaje float32; NER można wyłączyć w pasku bocznym.

Model był trenowany na EN/FR/ES/DE/IT/PT/NL, po polsku działa zero-shot. Progi pewności per etykieta
(`engine/ner_masker.py`) są dobrane pomiarem. Kolejny krok: dotrenowanie na polskich danych.

## 🔒 Zabezpieczenia

- **Tylko localhost.** `.streamlit/config.toml` wymusza `server.address = "localhost"`, limit
  pliku 25 MB i wyłącza telemetrię Streamlita. Adres Ollamy musi być lokalny (`localhost`,
  `127.0.0.1`, `::1`) – inny adres jest odrzucany, zanim jakikolwiek tekst zostanie wysłany.
- **Normalizacja Unicode** całego wejścia (NFKC + usunięcie znaków niewidocznych), więc PESEL
  z zero-width space czy cyframi pełnej szerokości nie omija Regexa.
- **Fail-closed.** Gdy etap LLM nie zadziała albo dokument zawiera polecenia dla modelu
  (prompt injection, np. „zignoruj instrukcje”), tekst do wysłania jest ukryty, dopóki
  użytkownik nie potwierdzi, że go sprawdził. Tekst trafia do modelu między znacznikami
  `<<<TEKST>>>…<<<KONIEC>>>`, a prompt każe ignorować zawarte w nim polecenia.
- Regex wykrywa też karty płatnicze (suma Luhna), klucze PEM, tokeny GitHub/Slack/Google/Stripe
  i adresy IPv6.

## Pliki

Tekst można wkleić, wybrać przypadek testowy albo wczytać plik: `.txt`, `.md`, `.csv`, `.log`, `.json`,
`.pdf`, `.docx` (także tabele w Wordzie). Pliki są czytane w pamięci i nigdzie nie zapisywane.
Odpowiedź AI też można wczytać z pliku. Zanonimizowany i przywrócony tekst można pobrać jako `.txt`.
### 🖼️ OCR tylko na obrazach

Skany, pieczątki i zdjęcia dokumentów w PDF są odczytywane przez OCR (Tesseract przez PyMuPDF,
bez dodatkowych bibliotek Pythona). Żeby było szybko i oszczędnie, OCR ([engine/ocr.py](engine/ocr.py)):

- działa **tylko na obrazach** osadzonych w PDF – strona bez obrazów nie trafia do OCR,
  a renderowany jest wyłącznie prostokąt obrazu (300 dpi, skala szarości),
- pomija obrazy za małe (ikony, kreski) i takie, które mają już warstwę tekstową (skan z OCR-em),
- ten sam obraz (np. logo na każdej stronie) rozpoznaje **raz**, a wynik całego pliku zapamiętuje,
  więc odczyt tekstu i zaczernianie nie liczą OCR dwa razy,
- słowa z OCR mają współrzędne na stronie, więc dostają czarne paski jak zwykły tekst.

Wymaga Tesseracta z modelami `pol` i `eng`
(Windows: instalator UB Mannheim, macOS: `brew install tesseract tesseract-lang`,
Linux: `apt install tesseract-ocr tesseract-ocr-pol`). PyMuPDF ma wbudowany silnik Tesseracta
i potrzebuje tylko plików modeli, więc zamiast instalacji systemowej wystarczy wrzucić
`pol.traineddata` i `eng.traineddata` (z [tesseract-ocr/tessdata](https://github.com/tesseract-ocr/tessdata))
do katalogu `tessdata/` w projekcie – bez uprawnień administratora. Bez modeli aplikacja działa dalej,
ale gdy PDF ma obrazy z możliwym tekstem, wynik jest wstrzymany do ręcznego sprawdzenia.
Obrazy w plikach `.docx` nie są jeszcze rozpoznawane.

PDF jest czytany **strona po stronie** (`page.get_text("text", sort=True)`), więc tabele, kolumny
i pieczęcie z Karty Podpisów (np. Autenti) wychodzą w kolejności czytania. Rozwijana sekcja
„🔍 Podgląd surowego tekstu” pokazuje dokładnie, co odczytał parser, z podziałem na strony.
Tekst jest normalizowany: myślnik zapisany w PDF jako SOFT HYPHEN (U+00AD) zamieniamy na „-”.
Bez tego NIP, UUID i daty z Karty Podpisów wyglądały poprawnie, ale regex ich nie widział.

## 🔑 Token sejfu

Zamiast tabeli mapowań (token → oryginał) aplikacja pokazuje **jeden token**, np. `PIIVAULT1.eNpF…`,
który zawiera cały sejf: JSON → kompresja zlib → base64 plus suma kontrolna CRC32.
Żeby odszyfrować odpowiedź AI, wklej ją w zakładce 2 razem z tokenem. Działa to także po zamknięciu
karty przeglądarki albo na innym komputerze, a dalsza anonimizacja z tym samym sejfem daje te same tokeny.

⚠️ To **nie jest szyfrowanie**: token jest kluczem do danych, więc kto go ma, odczyta oryginały.
Przechowuj go jak hasło i nigdy nie wysyłaj do zewnętrznego AI. Ucięty lub zmieniony token
jest odrzucany z czytelnym komunikatem.

## ⬛ Czarne paski (PDF)

Po anonimizacji można pobrać PDF, w którym dane wrażliwe są zakryte czarnymi paskami, jak w ocenzurowanych aktach.
Paski nie są nakładką na tekst: prawdziwych danych nie ma w pliku, więc nie da się ich skopiować spod paska.

- **Tekst wklejony lub z .txt/.docx:** PDF jest budowany od nowa, a w miejscu danych wstawiane jest czarne tło.
  Reszta tekstu da się zaznaczyć.
- **Wgrany PDF:** paski są rysowane na oryginalnym układzie strony, a strony są „spłaszczane” do obrazu (150 dpi).
  W wyniku nie ma żadnej warstwy tekstowej.
  Wbudowana redakcja PyMuPDF (`apply_redactions`) przesuwała tekst za paskiem, przez co „netto” zmieniało się w „etto”.

Wartości, których nie udało się znaleźć w PDF, aplikacja wypisuje do ręcznego sprawdzenia.

**Podpisy odręczne, pieczątki, kody QR.** Checkbox „Zamazać również podpisy odręczne, pieczątki i obrazy”
(domyślnie włączony) wycina z wgranego PDF-a wszystkie osadzone obrazy. Na każdy obraz nakładana jest
adnotacja redakcyjna, a `apply_redactions(images=PDF_REDACT_IMAGE_REMOVE, text=PDF_REDACT_TEXT_NONE)`
fizycznie usuwa go z pliku. Tekstu ten krok nie rusza, dlatego nie wraca błąd z przesuwaniem liter.
Potem strona jest spłaszczana jak wcześniej, więc w wyniku każda strona to jeden obraz całej strony.
Ograniczenie: podpis zapisany jako grafika wektorowa (linie), a nie obraz, nie jest wykrywany.

**Skany z warstwą OCR.** Obraz zajmujący co najmniej połowę strony to tło albo skan, a nie podpis,
więc nie jest zamazywany. Wcześniej zeskanowane pismo (każda strona = obraz A4 300 dpi) wychodziło
całkowicie czarne. Na takich stronach małe obrazy usuwamy trybem `PDF_REDACT_IMAGE_PIXELS`, bo tryb REMOVE
skasowałby cały skan, którego dotyka. Paski na tekście działają dzięki warstwie OCR. Podpis odręczny
**wewnątrz** skanu nie jest osobnym obrazem i nie zostanie wykryty.

**Paski nad polami formularza.** W Karcie Podpisów Autenti dane podpisujących (imię, e-mail, NIP, IP)
leżą pod polami podpisu (widgety PDF) z białym tłem. Te pola są rysowane na wierzchu treści strony i zasłaniały
paski. Dlatego paski są **wypalane w pikselach** obrazu strony po jej wyrenderowaniu. Nie są osobnym prostokątem,
który dałoby się usunąć w edytorze PDF.

**Paski tylko na całych słowach.** Wartość jest szukana jako ciąg całych słów strony (`get_text("words")`),
także przez złamanie linii. Wcześniejsze `search_for()` szukało podciągów, a tryb awaryjny „słowo po słowie”
zaczerniał każde „ul.” i „2009”. Na 6-stronicowym skanie było to 844 paski zamiast 123.

## Architektura hybrydowa

```
 tekst z PII
     │
     ▼
┌─────────────────────┐   24 wzorce, m.in. PESEL, NIP, REGON, KRS, BDO, dowód, paszport,
│ Etap 1: Regex       │   IBAN, kwoty, adresy, daty, tablice, VIN, UUID, IP, e-mail,
│ engine/regex_masker │   telefon, hasła i klucze API; sumy kontrolne → < 2 ms
└─────────┬───────────┘
          ▼
┌─────────────────────┐   osoby, firmy, adresy, kwoty, stanowiska
│ Etap 2: LLM ≤ 4B    │   strona po stronie (fragmenty ≤ 400 znaków), temperature 0.0,
│ engine/llm_masker   │   "format" = JSON Schema (Structured Outputs)
└─────────┬───────────┘   → model zwraca TYLKO {"entities": [...]}
          ▼
┌─────────────────────┐   Python podmienia encje w CAŁYM dokumencie:
│ Podmiana + Sejf     │   bez względu na wielkość liter, najdłuższe najpierw,
│ engine/vault        │   granice słów, odrzucenie encji spoza tekstu
└─────────┬───────────┘
          ▼
 tekst z tokenami ──► ChatGPT / Claude / … ──► odpowiedź z tokenami
                                                     │
                         Etap 3: vault.restore() ◄───┘  → oryginalne dane
```

### Dlaczego tak?

| Decyzja | Powód |
|---|---|
| Regex przed LLM | Dane o stałym formacie wykrywa szybko i bez błędów. Model 3B zajmuje się tylko kontekstem. |
| Kwoty w regexie | „18 500 zł”, „12 999,99 PLN”, „1.500,00 €”, „$1,200.50”, „15 tys. zł”, także kwota złamana na dwie linie w PDF (model 3B ją gubił). Liczby bez waluty (daty, procenty, numery) nie są ruszane. Kwoty idą przed telefonem, bo „150 000 000 zł” wyglądałoby jak numer. |
| Sumy kontrolne | PESEL, NIP, REGON, IBAN, dowód osobisty i paszport mają cyfrę kontrolną. Dzięki temu 11 przypadkowych cyfr (numer zamówienia) nie zostanie PESEL-em, a kod produktu „ABC123456” dowodem. Algorytmy sprawdzone na wzorach MSWiA (dowód `ABA300000`, paszport `ZS0000177`). |
| Kontekst zamiast zgadywania | KRS i NIP to oba 10 cyfr, a REGON-9 i BDO wyglądają jak telefon. Dlatego KRS łapiemy po słowie „KRS” lub prefiksie `0000`, REGON-9 i BDO tylko po swoim słowie kluczowym. Numery faktur (`FV1234567`) przypadkiem przechodziły sumę paszportu – prefiksy księgowe są wykluczone. |
| Strona po stronie | Model 3B traci uwagę pod koniec długiego tekstu (ostatnia strona = Karta Podpisów). Fragment nigdy nie łączy dwóch stron i ma ≤ 400 znaków (pomiar: przy całym logu model zwracał pustą listę, przy 200 znakach gubił osoby). |
| Globalna propagacja | Każda wykryta encja jest podmieniana w całym dokumencie bez względu na wielkość liter (`re.IGNORECASE`), więc „Nordis Logistyka Spółka…” i „NORDIS LOGISTYKA SPÓŁKA…” to jeden `[FIRMA_1]`. Działa też, gdy model przegapił nazwę na którejś stronie. Wyjątek: hasła i klucze, bo „Zima2025” ≠ „zima2025”. |
| JSON Schema w `format` | Ollama blokuje każdy token łamiący schemat (constrained decoding), więc mały model zawsze zwraca poprawny JSON z kategoriami z `enum`. |
| LLM nie przepisuje tekstu | Model 3B przy „przepisz z tagami” zmienia słowa i gubi zdania. U nas tekst zmienia tylko Python. |
| Weryfikacja `original in text` | Encja zmyślona przez model jest odrzucana. Halucynacja nie trafia do wyniku. |
| Fragmenty po ~400 znaków | Na całym logu model 3B zwracał pustą listę. Te same linie podane w krótszych fragmentach wykrywa poprawnie. |
| Dopasowanie odmiany osób | Model 3B „poprawia” odmianę („Piotr Wiśniewski” zamiast „Piotra Wiśniewskiego”). Python szuka rdzeni słów i maskuje tylko formy, które naprawdę są w tekście. Każda forma ma własny token, więc przywracanie jest dokładne co do litery. |
| Bezpieczniki przeciw pętli | Model 3B przy `temperature 0` potrafi się zapętlić: na krótkim pliku wygenerował 8800 tokenów zamiast ~150, a po 3 minutach wynik przepadł. Teraz `maxItems: 20` w JSON Schema (grammar Ollamy sam zamyka listę), `num_predict: 600`, limit czasu 60 s na fragment. Z uciętej odpowiedzi ratujemy kompletne encje. |
| `keep_alive: 30m`, `num_ctx: 2048` | Model nie jest zwalniany z pamięci między kliknięciami w demo. Mniejszy kontekst wystarcza i oszczędza ok. 300 MB RAM. |
| Tokeny nietykalne | Podmiana przeskakuje istniejące tokeny. Wcześniej encja „TEL_1” zwrócona przez model psuła token na `[[ADRES_3]]`. |
| Numeracja w Pythonie | Model podaje tylko kategorię, a `_1`, `_2` nadaje sejf. Ta sama wartość zawsze dostaje ten sam token. |
| Tolerancyjne `restore` | Rozpoznaje `[ OSOBA_1 ]`, `\[OSOBA_1\]`, `[osoba_1]`, bo tak potrafi przerobić tokeny ChatGPT. Nieznane tokeny zostają i są zgłaszane. |
| Fallback | Gdy Ollama nie działa, Regex nadal chroni dane. Lepsza częściowa ochrona niż żadna. |

## Struktura

```
app.py                  # Streamlit: zakładki Anonimizacja / De-anonimizacja + 4 przypadki testowe,
                        # wczytywanie plików, podgląd surowego tekstu, pobieranie wyników
engine/file_reader.py   # Odczyt plików: .txt .md .csv .log .json (UTF-8/cp1250), .pdf, .docx
engine/redactor.py      # PDF z czarnymi paskami w miejscu danych wrażliwych
engine/regex_masker.py  # Etap 1: wzorce + walidatory sum kontrolnych
engine/llm_masker.py    # Etap 2: Ollama + schemat Pydantic + obsługa błędów
engine/vault.py         # Sejf mapowań token <-> oryginał + eksport/import jako jeden token
engine/pipeline.py      # Spina etapy, mierzy czas
tests/test_pipeline.py  # 131 testów pipeline'u (LLM mockowany, Ollama niepotrzebna)
tests/test_file_reader.py # 12 testów odczytu plików (m.in. strony PDF, SOFT HYPHEN)
tests/test_redactor.py  # 14 testów zaczerniania (m.in. kolor pikseli pod paskiem)
```

## 📊 Metryki dla jury

Pomiary na MacBooku (Apple Silicon), Python 3.12, `qwen2.5:3b` w Ollamie, `temperature 0.0`.
Każdy przypadek uruchomiony 3 razy, podana mediana (model był już w pamięci; pierwsze ładowanie trwa ok. 7 s).

| Przypadek testowy | Regex | LLM (3B) | Encje Regex + LLM | Powtarzalność | Round-trip |
|---|---|---|---|---|---|
| 🏥 Dokument medyczny | 0,4 ms | **5,6 s** | 7 + 7 | 3/3 identyczne | 1:1 ✅ |
| 📄 Umowa B2B | 0,6 ms | **3,8 s** | 10 + 5 | 3/3 identyczne | 1:1 ✅ |
| 🐞 Log z błędem | 0,5 ms | **1,4 s** | 12 + 1 | 3/3 identyczne | 1:1 ✅ |
| ✍️ Umowa e-podpis, 3 strony z Kartą Podpisów | 1,1 ms | **9,4 s** | 30 łącznie | 2/2 identyczne | wielkość liter* |

\* Firma pisana raz normalnie, a raz WIELKIMI literami ma jeden token, więc po przywróceniu obie formy
mają postać z pierwszego wystąpienia. Dane są poprawne, różni się tylko wielkość liter.

Test umowy e-podpis zrobiono też na prawdziwym 3-stronicowym PDF-ie (Arial, myślniki jako SOFT HYPHEN):
**0 z 24 wrażliwych wartości widocznych** w tekście, a w zaczernionym PDF-ie wszystkie dostały pasek
(44 paski, 0 nieznalezionych).

| Zasoby | Wynik |
|---|---|
| RAM modelu `qwen2.5:3b` (z `/api/ps`, `num_ctx 2048`) | **~2,0 GB** |
| Krótka notatka (2 linie) w aplikacji, od kliknięcia do wyniku | **2,0 s** |
| Najgorszy przypadek (model zapętlony) | ograniczony do ok. 11 s na fragment (wcześniej 180 s i utrata wyniku) |
| RAM procesu Pythona (pipeline) | **~45 MB** (z interfejsem Streamlit ~75 MB) |
| Testy jednostkowe | **164/164** zaliczonych w < 1 s, bez Ollamy |

**Co zostało wykryte:** wszystkie osoby we wszystkich przypadkach (także w odmianie: „Jana Kowalskiego”,
„Piotra Wiśniewskiego”, „Katarzynę Zielińską”), wszystkie kwoty, adresy, PESEL, NIP, IBAN, telefony,
e-maile, IP oraz 4 sekrety w logu (hasło w JSON, Bearer, hasło w URL bazy, `api_key`).

**Co nie zostało wykryte lub zostało zamaskowane nadmiarowo** (model 3B):
- login `tnowak` w logu nie został zamaskowany (z etapem NER jest już maskowany jako `[LOGIN_1]`);
- w umowie e-podpis model oznaczył słowa „Pracownik” i „Spółka” jako encje, a „Toyota Corolla” jako firmę (nadmiarowo, ale bezpiecznie);
- „Prezesa Zarządu” (odmienione) zostało widoczne, a „PREZES ZARZĄDU” na Karcie Podpisów zamaskowane jako `[STANOWISKO_1]`;
- nazwa platformy „Autenti Sp. z o.o.” w pieczęci raz jest maskowana, a raz nie (zależy od fragmentu);
- kod choroby `I48` i numer księgi `2025/00417` zostały zamaskowane, chociaż nie są danymi osobowymi (to bezpieczny rodzaj błędu).

Aplikacja pokazuje te metryki na żywo: czas Regexa, czas LLM-a, liczbę encji, RAM procesu i RAM załadowanego modelu.

## Ograniczenia (uczciwie)

- Odmiana przez przypadki: „Jan Kowalski” i „Jana Kowalskiego” dostają różne tokeny (`[OSOBA_1]`, `[OSOBA_2]`). Zewnętrzne AI nie wie, że to ta sama osoba, ale po de-anonimizacji obie formy wracają poprawnie.
- Model 3B może przeoczyć część firm lub loginów (patrz metryki). Regex nadal chroni wtedy dane o stałym formacie.
- Tablice rejestracyjne nie mają sumy kontrolnej. Regex wymaga kodu województwa na początku i cyfry w numerze oraz pomija znane słowa (`GET 2000`, `PLN 1000`), ale rzadkie fałszywe trafienia są możliwe.
- Paszport i dowód z błędną cyfrą kontrolną (np. przykładowy `ABC123456`) celowo nie są maskowane.
- Propagacja korzysta z całego sejfu sesji. Wartości z wcześniej anonimizowanego dokumentu też zostaną zamaskowane w kolejnym („🗑️ Wyczyść sejf” zaczyna od zera).
- Spłaszczony PDF nie ma warstwy tekstowej (nie da się w nim wyszukiwać ani zaznaczać tekstu).
- Podział na fragmenty wydłuża przetwarzanie długich dokumentów (jedno zapytanie na ~400 znaków).
- Sejf żyje w pamięci sesji przeglądarki. Żeby wrócić do pracy po zamknięciu karty, zapisz token sejfu.
- Token sejfu nie jest szyfrowany hasłem: chroni go tylko to, że nikt poza Tobą go nie ma.
