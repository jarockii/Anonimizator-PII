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
python -m venv .venv
.venv\Scripts\activate             # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ner.txt   # opcjonalnie: etap NER (PyTorch CPU + GLiNER2)

# 3. Testy (bez Ollamy; "not slow" pomija testy z prawdziwym modelem NER)
pytest -m "not slow"

# 4. Aplikacja (http://localhost:8501, dostępna tylko z tego komputera)
streamlit run app.py
```

Bez działającej Ollamy aplikacja nadal działa: włącza się tryb samego Regexa (+ NER), a wynik jest
wstrzymany do ręcznego sprawdzenia (fail-closed). OCR obrazów w PDF wymaga modeli Tesseracta – patrz
[OCR tylko na obrazach](#️-ocr-tylko-na-obrazach). Jakość mierzy skrypt [`eval/run_eval.py`](eval/README.md).

## 🧠 Etap NER (GLiNER2-PII) – opcjonalny

Między Regexem a LLM-em działa model NER [GLiNER2-PII](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)
(Apache-2.0, ok. 0,3 mld parametrów, CPU). Zaznacza osoby, firmy, adresy i loginy w ułamku sekundy,
a mały LLM dostaje tekst już po nim i szuka tego, co wymaga kontekstu (stanowiska, odmienione formy).
NER nie generuje tekstu, więc nie halucynuje i nie da się go „przekonać” poleceniem w dokumencie.

```bash
pip install -r requirements-ner.txt   # PyTorch CPU + gliner2; model ~1,2 GB pobierze się raz
```

Pomiar na `przyklady/1–4` (Regex + NER + qwen2.5:3b): NER 0,16–0,47 s na dokument, LLM 5,7–14,1 s,
**0 wycieków i 0 nadmiarowo zamaskowanych słów** na sprawdzanej liście. NER znalazł wszystkie osoby
we wszystkich plikach (pewność 0,86–1,0) i login `tnowak` z przypadku „Log z błędem”, którego model 3B nie łapał.
Koszt: **~1,4–1,7 GB RAM** (wagi float32). Kwantyzacja int8 psuła model (0 encji), a bfloat16 była 15× wolniejsza
na CPU – dlatego zostaje float32; NER można wyłączyć w pasku bocznym.

Model był trenowany na EN/FR/ES/DE/IT/PT/NL, po polsku działa zero-shot. Progi pewności per etykieta
(`engine/ner_masker.py`) są dobrane pomiarem.

### Dotrenowany polski GLiNER (v3) – domyślny po złożeniu

Model dotrenowaliśmy (LoRA, 1 epoka, CPU, ~3 min) na syntetycznych polskich zdaniach z
[`training/gliner_dane.py`](training/gliner_dane.py): odmiana nazwisk we wszystkich przypadkach, osoby po tytule
(„dr”, „mgr inż.”) oraz przykłady negatywne (samo „PESEL”, „wew. 214”, nazwy miast, stanowiska). Żadne słowo
z oznaczeń testowych nie trafia do treningu. W repo jest sama nakładka LoRA (~10 MB, [`nakladki/gliner-pl-v3`](nakladki/gliner-pl-v3)),
a skrypt składa z niej gotowy model w `modele/gliner-pl-v3` (poza gitem, ~1,2 GB):

```bash
python training/scal_nakladke.py      # raz: pobiera oryginał i nakłada LoRA
streamlit run app.py                  # sam wykryje modele/gliner-pl-v3
```

Gdy `modele/gliner-pl-v3` istnieje, aplikacja (i `eval/run_eval.py`) **używa go domyślnie**. Pasek boczny pokazuje,
który model NER działa. Inny model albo oryginał wymusza zmienna `ANONIMIZATOR_NER_MODEL`:

```powershell
$env:ANONIMIZATOR_NER_MODEL = "fastino/gliner2-privacy-filter-PII-multi"; streamlit run app.py   # Windows
```
```bash
ANONIMIZATOR_NER_MODEL=fastino/gliner2-privacy-filter-PII-multi streamlit run app.py             # Linux/macOS
```

| Model | Osoby (sam NER) | Zwykłe słowa zamaskowane | Regex + NER | Regex + NER + LLM |
|---|---|---|---|---|
| oryginalny GLiNER2-PII | 40/43 | 6 | 100/114 | 218/228 (95,6%) |
| **gliner-pl-v3** | **43/43** | **4** | **105/114** | **222/228 (97,4%)** |

**Zestaw kontrolny** (4 nowe dokumenty, 47 danych, niewidziane przy strojeniu i treningu – [eval/README](eval/README.md#zestaw-kontrolny---zestaw-kontrolny)):

| Model | Regex + NER | Regex + NER + LLM (2 przebiegi) |
|---|---|---|
| oryginalny GLiNER2-PII | 31/47 (66,0%) | 80/94 (85,1%) |
| gliner-pl-v3 | 34/47 (72,3%) | 82/94 (87,2%) |
| **gliner-pl-v3 + reguły kontekstowe** | **42/47 (89,4%)** | **94/94 (100%)** |

v3 jest lepszy także na nowych dokumentach, więc poprawa nie jest tylko dopasowaniem do zestawu podstawowego.
Stałe wycieki z pierwszego pomiaru kontrolnego naprawiły ogólne reguły (a nie poprawki pod konkretne słowa –
testy w `tests/test_context_rules.py` używają innych przykładów), opisane niżej w „Regułach kontekstowych”.
Na zestawie podstawowym ten sam zestaw zmian dał **224/228 (98,2%)** (wcześniej 222/228). Zostały:
„chemia / onkolog” (choroba mamy w czacie) i „brata prezesa”.

**Reguły kontekstowe** ([engine/context_rules.py](engine/context_rules.py)), uruchamiane po Regexie:
- firma przed formą prawną („BUD-MAX RZESZÓW SP. Z O.O.”), także gdy tabela PDF wstawia między słowa nazwy
  liczby z sąsiednich kolumn; słowa-funkcje z przodu („Prezes Zarządu …”) są odcinane,
- numer kredytu, pożyczki, polisy, klienta po słowie kluczowym („pożyczki gotówkowej nr PG/2023/118204”),
- zdrobnienia imion z odmianą („Bartek”, „Zbyszka”, „Kasi”); dwuznaczne („Kuba”) pominięte,
- „jedyny/jedyna + rzeczownik” jako identyfikacja pośrednia („jedyna kobieta”), ale nie „jedyny sposób”,
- potoczny opis osoby po „ten co / ta co” („ten co sprzedaje miód na targu”),
- stopień i symbol niepełnosprawności („umiarkowanym stopniu niepełnosprawności”, „10-N”).

LLM dostał dwa zabezpieczenia przed utratą trafień: fragment danych szczególnych, który model „poprawił”
(„chorobie płuc” zamiast „choroby płuc”, „jest [OSOBA_1] Solidarność”), jest odnajdywany w tekście po rdzeniach
słów, a opis osoby bez słowa o relacji jest przyjmowany, gdy w tym samym zdaniu stoi „ten co…” albo „pan z…”.

Naprawione m.in. celowniki „Szewczence”, „Brzęczyszczykiewiczowi” i nazwisko po „née”. Pomiar:
`eval/szara_strefa.py` i `eval/run_eval.py` ([wyniki](eval/wyniki)). Uwaga: to te same 8 dokumentów, na których
porównywaliśmy kolejne wersje treningu, więc wynik może być nieco optymistyczny. Bez złożonego modelu
aplikacja używa oryginału. Liczba parametrów po scaleniu LoRA się nie zmienia (0,31 mld).

## 🧩 Podział pracy: Regex, NER, LLM

| Etap | Co robi | Przykłady |
|---|---|---|
| Regex (+ liczby słownie) | dane o stałym formacie, deterministycznie | PESEL, IBAN (też zagraniczny), księga wieczysta, działka, paszport po słowie „paszport”, daty po angielsku, e-mail „(at)… kropka”, profil LinkedIn, `skype:`/`login:`, „na Długiej 5 m. 12”, adres wiejski, **telefon/kwota/PESEL/data zapisane słownie** |
| NER (GLiNER2, 0,31 mld) | osoby, firmy, adresy, loginy | „zbyszek kowal”, „Krzysiu”, „Grzegorzowi Brzęczyszczykiewiczowi” |
| LLM (qwen2.5:3b) | to, co wymaga rozumienia sensu | **identyfikacja pośrednia** („syn burmistrza Kutna”, `[POSREDNI_n]`), **dane szczególne z art. 9 RODO** (zdrowie, wyznanie, związki zawodowe – osobne, wąskie zapytanie tylko dla fragmentów wskazanych przez słownik, `[WRAZLIWE_n]`), formy, których NER nie zna |

Pomiar zestawem [`eval/`](eval/README.md) na `przyklady/1–8` (114 ręcznie oznaczonych danych, 2 przebiegi,
Regex + NER + qwen2.5:3b): **ukryto 218/228 (95,6%)**, pliki 1–4 bez wycieków. Pozostałe przypadki wymagają wnioskowania („jedyna kobieta na tym stanowisku”) i zmieniają się między
przebiegami – wynik LLM 3B nie jest w pełni powtarzalny. Model może też nadmiarowo zamaskować pojedyncze słowa
(„Najemcy”). Weryfikacja kategorii NER przez LLM jest dostępna jako opcja eksperymentalna (domyślnie wyłączona:
qwen2.5:3b częściej psuł kategorie, niż je poprawiał).

**Zostaw jawnie (per kategoria).** W pasku bocznym można świadomie nie maskować wybranych kategorii,
np. **diagnoz** – żeby AI mogło streścić dokumentację medyczną. Dostępne są tylko kategorie, które same
nie wskazują człowieka: ZDROWIE, STANOWISKO, KWOTA, DATA, FIRMA. Osoby, PESEL-e, adresy, sekrety,
identyfikatory pośrednie i pozostałe dane szczególne (wyznanie, związki zawodowe…) są maskowane zawsze.
Dane o zdrowiu są wydzielone z WRAZLIWE słownikiem (nie przez LLM): fragment mieszany („katolik po zawale”)
albo bez medycznego słowa zostaje WRAZLIWE i jest zamaskowany. Zostawione dane nie dostają też paska w PDF,
a aplikacja wypisuje je w osobnej tabeli.

**Limit parametrów:** qwen2.5:3b (3,09 mld) + GLiNER2-PII (0,31 mld) = **3,39 mld ≤ 4 mld**. Pasek boczny
pokazuje sumę i ostrzega, gdy model jest za duży (np. qwen3.5:4b ma 4,21 mld – sam przekracza limit).

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
`.pdf`, `.docx` (także tabele i obrazy w Wordzie). Pliki są czytane w pamięci i nigdzie nie zapisywane.
Odpowiedź AI też można wczytać z pliku. Zanonimizowany i przywrócony tekst można pobrać jako `.txt`.

**Word (`.docx`) w całości** ([engine/docx_reader.py](engine/docx_reader.py)). Zwykłe `document.paragraphs`
z python-docx pomija sporo miejsc, w których w pismach stoją dane, więc czytamy XML sami, w kolejności dokumentu:
nagłówki, treść, przypisy dolne i końcowe, stopki. W treści także pola tekstowe (pieczątka jako kształt),
kontrolki formularza („Najemca: [pole]” – w akapicie i jako blok) i tabele zagnieżdżone w komórkach.
Pole tekstowe Word zapisuje dwa razy (DrawingML + kopia VML dla starych wersji) – kopię pomijamy, inaczej tekst
byłby podwojony. Scalona komórka nie jest już powtarzana w każdej kolumnie, a tekst usunięty w śledzeniu zmian
nie trafia do wyniku. Na pliku testowym z tymi elementami wcześniej znikało 8 z 8 danych – teraz każda jest
odczytana dokładnie raz. Komentarze recenzentów i metadane (autor pliku) nie są czytane.

### 🖼️ OCR tylko na obrazach

Skany, pieczątki i zdjęcia dokumentów w PDF są odczytywane przez OCR (Tesseract przez PyMuPDF,
bez dodatkowych bibliotek Pythona). Żeby było szybko i oszczędnie, OCR ([engine/ocr.py](engine/ocr.py)):

- działa **tylko na obrazach** osadzonych w PDF – strona bez obrazów nie trafia do OCR,
  a renderowany jest wyłącznie prostokąt obrazu (300 dpi, RGB – w skali szarości wbudowany
  Tesseract zwracał pusty tekst),
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

**Obrazy w Wordzie (`.docx`).** OCR przechodzi też po obrazach wstawionych w treść dokumentu
(zdjęcie dowodu, zeskanowana pieczątka, zrzut ekranu – także w komórkach tabeli). Rozpoznany tekst
trafia **zaraz po akapicie z obrazem**, więc model widzi go w kontekście
(„Skan dowodu w załączniku:” + treść dowodu); obraz w komórce tabeli – w tej komórce. Obrazy są czytane
także z nagłówków, stopek, pól tekstowych i przypisów. Ten sam obraz wstawiony kilka razy (logo) jest rozpoznawany raz.
Obraz nie ma tu rozmiaru na stronie, więc:

- pomijane są obrazy mniejsze niż 40 × 16 px (punktory, ikony),
- obrazy węższe niż 1600 px są powiększane (maks. ×3) – bez tego tekst 11 pt z obrazu 72 dpi
  dawał „PESEL 92031802349” (zła cyfra, więc Regex go nie łapał), a z powiększeniem jest bezbłędny.

Bez Tesseracta wynik dla `.docx` z obrazami jest wstrzymany do sprawdzenia, jak przy PDF. Obrazy
w formatach wektorowych Worda (EMF/WMF) nie są odczytywane – wstrzymują wynik jako błąd OCR.

PDF jest czytany **strona po stronie** (`page.get_text("text", sort=True)`), więc tabele, kolumny
i pieczęcie z Karty Podpisów (np. Autenti) wychodzą w kolejności czytania. Rozwijana sekcja
„🔍 Co odczytano z pliku” pokazuje dokładnie, co odczytał parser, z podziałem na strony.
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

**Podpisy wektorowe.** Podpis narysowany liniami (tablet, część platform e-podpisu, edytor PDF) nie jest
obrazem, więc wycinanie obrazów go nie usuwało. [engine/signatures.py](engine/signatures.py) rozpoznaje go
po kształcie ścieżek: ciągłe pociągnięcia z wieloma odcinkami, w większości ukośne, z powrotami pióra w lewo
(pętle liter). Ramki tabel, podkreślenia, wykresy liniowe i kołowe oraz gęste kreskowanie nie spełniają tych
warunków. Wykryty podpis dostaje pasek wypalony w pikselach. Koszt: 3–7 ms na stronę przykładowych PDF-ów,
bez modelu. Ograniczenie: podpis zapisany jako **wypełniony** kontur liter jest pomijany, bo tak wyglądają też
logo i wykresy.

**Skany z warstwą OCR.** Obraz zajmujący co najmniej połowę strony to tło albo skan, a nie podpis,
więc nie jest zamazywany. Wcześniej zeskanowane pismo (każda strona = obraz A4 300 dpi) wychodziło
całkowicie czarne. Na takich stronach małe obrazy usuwamy trybem `PDF_REDACT_IMAGE_PIXELS`, bo tryb REMOVE
skasowałby cały skan, którego dotyka. Paski na tekście działają dzięki warstwie OCR.

**Podpisy wewnątrz skanu.** Podpis na skanie nie jest osobnym obrazem ani ścieżką – to zwykłe piksele.
Bez modelu i bez pobierania czegokolwiek: przy słowie „podpis” („Podpis:”, „(podpis)”, „Podpisy”, „signature”)
z warstwy tekstowej skanu albo z OCR bierzemy okolicę, czyścimy w niej piksele rozpoznanego tekstu, linie
do podpisu („……”, „____”) i drobne plamki, a to, co zostało ciemnego, to pismo odręczne i dostaje pasek.
„Niżej podpisany” ani „podpisania” nie są kotwicą – to treść pisma. Działa tylko na stronach-skanach,
na obrazie strony, który redaktor i tak renderuje: ok. 75 ms na stronę ze słowem „podpis”, bez niego ~0.
Ograniczenie: podpis bez żadnego podpisu-etykiety w pobliżu (np. parafa na marginesie) nie zostanie
wykryty – aplikacja przypomina o sprawdzeniu podglądu skanów.

**Paski nad polami formularza.** W Karcie Podpisów Autenti dane podpisujących (imię, e-mail, NIP, IP)
leżą pod polami podpisu (widgety PDF) z białym tłem. Te pola są rysowane na wierzchu treści strony i zasłaniały
paski. Dlatego paski są **wypalane w pikselach** obrazu strony po jej wyrenderowaniu. Nie są osobnym prostokątem,
który dałoby się usunąć w edytorze PDF.

**Paski tylko na całych słowach.** Wartość jest szukana jako ciąg całych słów strony (`get_text("words")`),
także przez złamanie linii. Wcześniejsze `search_for()` szukało podciągów, a tryb awaryjny „słowo po słowie”
zaczerniał każde „ul.” i „2009”. Na 6-stronicowym skanie było to 844 paski zamiast 123.

**Tekst po skosie (znaki wodne).** Słowo w obróconej linii dostaje pasek w kształcie obróconego czworokąta,
a nie prostokąt otaczający. Ukośny znak wodny „SPECIMEN” (92 pt) na wyciągu bankowym miał prostokąt
446 × 387 pt i pasek na nim zakrywał całą tabelę. Blady, półprzezroczysty napis (na białym tle jaśniejszy niż
200/255) nie dostaje czarnego czworokąta: w jego obrębie wybielamy tylko jasne piksele, więc napis znika,
a ciemny tekst tabeli, który na nim leży, zostaje czytelny.

## Architektura hybrydowa

```
 tekst z PII (plik → engine/file_reader + engine/ocr: OCR tylko na obrazach PDF i .docx)
     │
     ▼
┌─────────────────────┐   normalizacja Unicode (NFKC, znaki niewidoczne, myślniki)
│ Etap 1: Regex       │   42 wzorce w 24 kategoriach, m.in. PESEL, NIP, REGON, KRS, dowód,
│ engine/regex_masker │   paszport, IBAN, karta, kwoty, adresy, daty, IP, e-mail, sekrety;
│ engine/numerals     │   sumy kontrolne + liczby zapisane słownie → kilka ms
└─────────┬───────────┘
          ▼
┌─────────────────────┐   GLiNER2-PII (0,31 mld, CPU, opcjonalny):
│ Etap 2: NER         │   osoby, firmy, adresy, loginy
│ engine/ner_masker   │
└─────────┬───────────┘
          ▼
┌─────────────────────┐   to, czego NER nie zna: stanowiska, kwoty, identyfikacja pośrednia,
│ Etap 3: LLM ≤ 4B    │   + wąskie przejście dla danych szczególnych (art. 9 RODO);
│ engine/llm_masker   │   fragmenty ≤ 400 znaków, temperature 0.0, "format" = JSON Schema
└─────────┬───────────┘   → model zwraca TYLKO {"entities": [...]}
          ▼
┌─────────────────────┐   Python podmienia encje w CAŁYM dokumencie:
│ Podmiana + Sejf     │   bez względu na wielkość liter, najdłuższe najpierw,
│ engine/vault        │   granice słów, odrzucenie encji spoza tekstu
└─────────┬───────────┘
          ▼
 tekst z tokenami ──► ChatGPT / Claude / … ──► odpowiedź z tokenami
                                                     │
                De-anonimizacja: vault.restore() ◄───┘  → oryginalne dane
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
| `keep_alive: 30m`, `num_ctx: 3072` | Model nie jest zwalniany z pamięci między kliknięciami w demo. Kontekst mieści prompt z 3 przykładami (~800 tokenów), fragment i odpowiedź; domyślne 4096+ zużywałoby więcej RAM-u. |
| `127.0.0.1` zamiast `localhost` | Windows rozwiązuje `localhost` najpierw na IPv6 (`::1`), a Ollama słucha tylko na `127.0.0.1` – **każde** zapytanie czekało ok. 2,1 s na nieudaną próbę. Przy 4–6 zapytaniach na dokument to 8–12 s straty. Teraz adres jest domyślnie `127.0.0.1`, a wpisane `localhost` jest na nie zamieniane. Mediana czasu dokumentu (Regex + NER + LLM, GPU): **~15 s → ~7,6 s**, wyniki bez zmian. |
| Bez zapytań równoległych | Ollama i tak przetwarza je kolejno (2 i 4 wątki: 7,3–7,4 s wobec 7,6 s), a wynik przestawał być powtarzalny. |
| Tokeny nietykalne | Podmiana przeskakuje istniejące tokeny. Wcześniej encja „TEL_1” zwrócona przez model psuła token na `[[ADRES_3]]`. |
| Numeracja w Pythonie | Model podaje tylko kategorię, a `_1`, `_2` nadaje sejf. Ta sama wartość zawsze dostaje ten sam token. |
| Tolerancyjne `restore` | Rozpoznaje `[ OSOBA_1 ]`, `\[OSOBA_1\]`, `[osoba_1]`, bo tak potrafi przerobić tokeny ChatGPT. Nieznane tokeny zostają i są zgłaszane. |
| Fallback | Gdy Ollama nie działa, Regex nadal chroni dane. Lepsza częściowa ochrona niż żadna. |

## Struktura

```
app.py                  # Streamlit: zakładki Anonimizacja / De-anonimizacja, przykłady,
                        # wczytywanie plików, „zostaw jawnie”, pobieranie wyników
.streamlit/config.toml  # tylko localhost, limit pliku 25 MB, bez telemetrii
engine/file_reader.py   # Odczyt plików: .txt .md .csv .log .json (UTF-8/cp1250), .pdf, .docx + normalizacja
engine/docx_reader.py   # Word w całości: nagłówki, stopki, przypisy, pola tekstowe, kontrolki, tabele zagnieżdżone
engine/ocr.py           # OCR (Tesseract w PyMuPDF) tylko na obrazach w PDF i .docx, z pamięcią podręczną
engine/signatures.py    # Podpisy odręczne: wektorowe (kształt ścieżek) i na skanach (atrament przy „podpis”)
engine/redactor.py      # PDF z czarnymi paskami w miejscu danych wrażliwych
engine/regex_masker.py  # Etap 1: wzorce + walidatory sum kontrolnych
engine/numerals.py      # Liczby zapisane słownie (telefon, kwota, PESEL, data)
engine/ner_masker.py    # Etap 2: GLiNER2-PII (opcjonalny)
engine/llm_masker.py    # Etap 3: Ollama + schemat Pydantic + obrona przed prompt injection
engine/vault.py         # Sejf mapowań token <-> oryginał + eksport/import jako jeden token
engine/pipeline.py      # Spina etapy, mierzy czas, decyduje o wstrzymaniu wyniku (fail-closed)
eval/                   # Pomiar jakości: ręczne oznaczenia danych + run_eval.py (eval/README.md)
przyklady/              # 8 testowych PDF-ów (generuj_pdf.py), m.in. skany, pieczątki, prompt injection
tests/                  # pytest: LLM mockowany, Ollama niepotrzebna; marker „slow” = prawdziwy NER
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
| Testy jednostkowe | **379/379** zaliczonych w ok. 17 s (z prawdziwym NER), bez Ollamy |

**Co zostało wykryte:** wszystkie osoby we wszystkich przypadkach (także w odmianie: „Jana Kowalskiego”,
„Piotra Wiśniewskiego”, „Katarzynę Zielińską”), wszystkie kwoty, adresy, PESEL, NIP, IBAN, telefony,
e-maile, IP oraz 4 sekrety w logu (hasło w JSON, Bearer, hasło w URL bazy, `api_key`).

**Co nie zostało wykryte lub zostało zamaskowane nadmiarowo** (model 3B):
- login `tnowak` w logu nie został zamaskowany (z etapem NER jest już maskowany jako `[LOGIN_1]`);
- w umowie e-podpis model oznaczył słowa „Pracownik” i „Spółka” jako encje, a „Toyota Corolla” jako firmę (nadmiarowo, ale bezpiecznie; same nazwy ról jak „Pracownik” trafiają teraz do STANOWISKO zamiast OSOBA);
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

## Współpraca, bezpieczeństwo, licencja

- Jak przygotować środowisko, testy, pomiar `eval/` i PR: [CONTRIBUTING.md](CONTRIBUTING.md).
- Lukę bezpieczeństwa (np. wyciek danych) zgłoś prywatnie: [SECURITY.md](SECURITY.md).
- Kod projektu jest na licencji **MIT** ([LICENSE](LICENSE)).

Zależności i modele mają własne licencje, które obowiązują niezależnie od naszej:

| Składnik | Licencja |
|---|---|
| PyMuPDF (odczyt i zaczernianie PDF, OCR) | **AGPL-3.0** albo komercyjna licencja Artifex – udostępniając aplikację innym (także przez sieć), trzeba spełnić warunki AGPL |
| qwen2.5:3b (Ollama) | Qwen Research License – sprawdź warunki przed użyciem komercyjnym |
| GLiNER2-PII (`fastino/gliner2-privacy-filter-PII-multi`), gliner2, peft, PyTorch, Streamlit | Apache-2.0 (PyTorch: BSD-style) |
| Tesseract i modele `tessdata` | Apache-2.0 |
| python-docx, pydantic | MIT |
