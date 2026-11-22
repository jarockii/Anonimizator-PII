# 🔒 Polityka bezpieczeństwa

Anonimizator przetwarza dane osobowe, więc za lukę bezpieczeństwa uważamy nie tylko klasyczny
błąd w kodzie, ale też każdą sytuację, w której **prawdziwe dane mogą opuścić komputer** albo
trafić do tekstu przeznaczonego dla zewnętrznego AI.

## Wspierane wersje

Projekt jest rozwijany na gałęzi `main` i nie ma wydań z numerami wersji. Poprawki trafiają
tylko na `main` – zgłaszając problem, sprawdź go na najnowszym commicie.

| Wersja | Wsparcie |
|---|---|
| `main` (najnowszy commit) | ✅ |
| starsze commity, forki | ❌ |

## Jak zgłosić lukę

**Nie zakładaj publicznego issue** dla luki, która pozwala ujawnić dane.

1. Wejdź w zakładkę **Security** repozytorium na GitHubie i wybierz
   **Report a vulnerability** (prywatne zgłoszenie – widzą je tylko opiekunowie).
2. Opisz:
   - co wycieka albo co da się obejść i jaki jest skutek,
   - kroki do odtworzenia: wersja (hash commita), system, model w Ollamie, czy NER i OCR są włączone,
   - **przykład na fikcyjnych danych** – nigdy nie wklejaj prawdziwych danych osobowych,
     nawet swoich. Do PESEL-i, NIP-ów i IBAN-ów użyj wartości z poprawną sumą kontrolną
     wygenerowanych na potrzeby testu.
3. Jeśli masz pomysł na poprawkę, dołącz go do zgłoszenia, a PR otwórz dopiero po
   uzgodnieniu z opiekunami.

To projekt hackathonowy rozwijany po godzinach. Postaramy się potwierdzić zgłoszenie
w ciągu **7 dni** i ustalić z Tobą termin ujawnienia. Po wydaniu poprawki chętnie wymienimy
Cię w opisie zmiany, jeśli sobie tego życzysz.

## Co uznajemy za lukę

- **Wyciek danych do tekstu wynikowego**, który powinien być zamaskowany: dana z kategorii
  maskowanej zawsze (osoba, PESEL, adres, sekret, dane szczególne…) jest widoczna
  w „Tekście do wysłania” albo nie dostaje paska w PDF, mimo że aplikacja nie wstrzymała
  wyniku do sprawdzenia.
- **Obejście fail-closed**: dokument z poleceniami dla modelu (prompt injection), awaria LLM
  albo brak OCR nie powoduje wstrzymania wyniku.
- **Dane w zaczernionym PDF**: tekst lub obraz spod paska da się odzyskać z pliku.
- **Ruch sieciowy poza komputer**: tekst albo dane trafiają do innego hosta niż lokalna Ollama,
  aplikacja nasłuchuje na interfejsie innym niż `localhost`, telemetria jest włączona.
- **Sejf**: mapowania albo token sejfu trafiają na dysk, do logów albo do tekstu wysyłanego do AI.
- **Wykonanie kodu** przez spreparowany plik wejściowy (PDF, DOCX, JSON).

## Co nie jest luką

Poniższe zachowania są opisane w README jako znane ograniczenia. Zgłoś je jako zwykłe issue
albo dodaj przypadek do [`eval/`](eval/README.md):

- dana wykryta tylko częściowo przez model w przypadku wymagającym wnioskowania
  („jedyna kobieta na tym stanowisku”), gdy taki fragment jest w `eval/` oznaczony jako znany wyciek,
- nadmiarowe maskowanie zwykłych słów,
- dane z kategorii, które użytkownik **sam** zostawił jawnie przełącznikiem „Zostaw jawnie”,
- dostęp do danych przez osobę, która ma token sejfu – token **nie jest szyfrowany**
  i należy go chronić jak hasło; suma CRC32 chroni go tylko przed przypadkowym uszkodzeniem,
  więc celowo zmieniony token zmieni przywracane dane,
- ataki wymagające dostępu do konta użytkownika albo do komputera, na którym działa aplikacja,
- luki w zależnościach (Streamlit, PyMuPDF, PyTorch, Ollama), które nie są wykorzystywalne
  przez sposób, w jaki ich używamy – zgłoś je bezpośrednio autorom tych projektów.

## Model zagrożeń w skrócie

- Aplikacja działa **wyłącznie lokalnie**: Streamlit na `localhost`, Ollama pod adresem
  pętli zwrotnej (inny adres jest odrzucany przed wysłaniem tekstu), modele NER i OCR na CPU.
- Za niezaufane uważamy **treść dokumentu** (może zawierać polecenia dla modelu) i **odpowiedź
  zewnętrznego AI** (wraca do de-anonimizacji). Model językowy nigdy nie przepisuje tekstu –
  zwraca tylko listę fragmentów w JSON Schema, a podmianę robi Python.
- Ostatnią linią obrony jest **człowiek**: gdy którykolwiek etap zawiedzie, aplikacja wstrzymuje
  wynik i prosi o ręczne sprawdzenie (fail-closed).
