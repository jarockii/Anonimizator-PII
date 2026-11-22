"""Interfejs webowy (Streamlit): 1) anonimizacja, 2) de-anonimizacja.

Uruchomienie:  streamlit run app.py
"""

import streamlit as st

from engine.file_reader import (
    PAGE_BREAK,
    SUPPORTED_EXTENSIONS,
    FileReadError,
    join_pages,
    read_uploaded_pages,
)
from engine.llm_masker import (
    DEFAULT_MODEL,
    OLLAMA_URL,
    OllamaAddressError,
    check_ollama,
    get_model_memory_mb,
    get_model_params,
    validate_ollama_url,
)
from engine.ner_masker import (
    NER_MODEL,
    NER_PARAMS,
    ORIGINAL_NER_MODEL,
    ner_installed,
    ner_model_label,
    preload_model as preload_ner_model,
)
from engine.ocr import clear_cache as clear_ocr_cache
from engine.ocr import ocr_available, ocr_docx, ocr_pdf
from engine.pipeline import KEEPABLE_CATEGORIES, anonymize, get_process_peak_ram_mb
from engine.redactor import page_count, redact_pdf, redact_text_to_pdf, render_preview
from engine.vault import Vault, VaultTokenError

# Limit hackathonu na liczbę parametrów modelu językowego.
PARAM_LIMIT = 4_000_000_000

# Kategorie, które można zostawić jawnie (pipeline.KEEPABLE_CATEGORIES).
KEEP_OPTIONS = {
    "ZDROWIE": "🩺 Diagnozy i leczenie (ZDROWIE)",
    "STANOWISKO": "👔 Stanowiska",
    "KWOTA": "💰 Kwoty",
    "DATA": "📅 Daty",
    "FIRMA": "🏢 Firmy i instytucje",
}
assert set(KEEP_OPTIONS) == KEEPABLE_CATEGORIES, "KEEP_OPTIONS musi odpowiadać KEEPABLE_CATEGORIES"

st.set_page_config(page_title="Lokalny Anonimizator PII", page_icon="🛡️", layout="wide")

# Przycisk „Przykłady” obok pola pliku ma mieć tę samą wysokość co pole
# (pomiar w przeglądarce: pole 68 px, zwykły przycisk 40 px). Ta sama
# minimalna wysokość dla obu – pole przy wąskim oknie i tak może urosnąć.
st.markdown(
    """<style>
    [data-testid="stFileUploaderDropzone"] { min-height: 4.25rem; }
    .st-key-examples_box [data-testid="stPopoverButton"] { min-height: 4.25rem; }
    </style>""",
    unsafe_allow_html=True,
)

# --- Trudne przypadki testowe ----------------------------------------------
# Celowo zawierają: odmianę nazwisk, różne formaty numerów, dane „ukryte”
# w logach i liczby, które NIE są danymi osobowymi (pułapki na false positive).
TEST_CASES = {
    "🏥 Dokument medyczny": (
        "KARTA INFORMACYJNA LECZENIA SZPITALNEGO\n"
        "Szpital Specjalistyczny im. św. Łukasza w Krakowie, Oddział Kardiologii\n\n"
        "Pacjent: Jan Kowalski, PESEL 44051401359, zam. ul. Długa 15/3, 31-147 Kraków.\n"
        "Tel. kontaktowy: +48 601 234 567, e-mail: jan.kowalski@poczta.pl\n"
        "Osoba upoważniona: Anna Kowalska (córka), tel. 512-345-678.\n\n"
        "Rozpoznanie: migotanie przedsionków (I48). Pacjent przyjęty 12.03.2025, "
        "skierowany przez lek. Piotra Wiśniewskiego. Stan Jana Kowalskiego po "
        "kardiowersji stabilny. Zalecono kontrolę za 3 miesiące.\n"
        "Koszt leczenia poza NFZ: 4 850,00 zł. Nr księgi głównej: 2025/00417."
    ),
    "📄 Umowa B2B": (
        "UMOWA O ŚWIADCZENIE USŁUG B2B nr 17/2025\n\n"
        "zawarta w Warszawie pomiędzy:\n"
        "TechNova Sp. z o.o. z siedzibą przy ul. Marszałkowskiej 84/92, 00-514 Warszawa, "
        "NIP 526-025-02-74, reprezentowaną przez Katarzynę Zielińską – Prezesa Zarządu,\n"
        "a\n"
        "Marek Lewandowski prowadzący działalność gospodarczą pod firmą "
        "ML Software Marek Lewandowski, NIP 7771234567, ul. Polna 7, 60-535 Poznań.\n\n"
        "§3. Wynagrodzenie miesięczne wynosi 18 500 zł netto plus VAT. "
        "Kara umowna za zwłokę: 250 zł za każdy dzień.\n"
        "Płatność na rachunek: PL61 1090 1014 0000 0712 1981 2874.\n"
        "Kontakt: k.zielinska@technova.pl, m.lewandowski@mlsoftware.dev, tel. (22) 628 41 00."
    ),
    "🐞 Log z błędem": (
        "2025-03-14 09:41:22,117 ERROR [auth-service] Login failed for user "
        "tomasz.nowak@firma-xyz.com from 192.168.10.45\n"
        "2025-03-14 09:41:22,118 DEBUG Request payload: {\"username\": \"tnowak\", "
        "\"password\": \"Zima2025!@#\", \"remember\": true}\n"
        "2025-03-14 09:41:23,004 WARN  Retrying with header Authorization: "
        "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0bm93YWsifQ.abc123XYZ\n"
        "2025-03-14 09:41:23,512 ERROR PaymentGateway: transfer 12 999,99 PLN to "
        "PL29 1140 2000 4000 0350 2753 4900 rejected (customer: Tomasz Nowak, "
        "PESEL 92031502349, phone 600100200)\n"
        "2025-03-14 09:41:24,001 INFO  db_url=postgres://admin:S3cr3t@10.0.0.12:5432/prod "
        "api_key=sk-live-9f8e7d6c5b4a3210 retry_count=3 order_id=10000000001"
    ),
    # „\f” = nowa strona (tak jak w tekście wyciągniętym z PDF). Ostatnia
    # strona to Karta Podpisów: nazwy WIELKIMI literami, UUID, daty i IP.
    "✍️ Umowa e-podpis (3 strony)": (
        "UMOWA POWIERZENIA MIENIA NR 42/2025\n"
        "zawarta w dniu 12.03.2025 r. w Gdańsku pomiędzy:\n"
        "Nordis Logistyka Spółka z ograniczoną odpowiedzialnością z siedzibą\n"
        "przy ul. Grunwaldzkiej 82, 80-244 Gdańsk, KRS 0000436998,\n"
        "NIP 526-025-02-74, REGON 123456785, BDO 000012345,\n"
        "reprezentowaną przez Prezesa Zarządu Tomasza Wiśniewskiego,\n"
        "a\n"
        "Panem Pawłem Zielińskim, PESEL 92031502349, dowód osobisty ABA300000,\n"
        "zam. ul. Kwiatowa 3/12, 81-301 Gdynia, tel. +48 601 234 567,\n"
        "e-mail: p.zielinski@poczta.pl"
        "\f"
        "§2. Przedmiot umowy\n"
        "Spółka powierza Pracownikowi samochód służbowy Toyota Corolla\n"
        "o nr rej. GD 1234A, VIN 1HGCM82633A004352.\n"
        "§3. Odpowiedzialność\n"
        "Za szkody Pracownik odpowiada do kwoty 8 500,00 zł.\n"
        "Rozliczenia na rachunek PL61 1090 1014 0000 0712 1981 2874."
        "\f"
        "KARTA PODPISÓW\n"
        "Identyfikator dokumentu: 24150c17-720c-4d3b-974b-abb25301120e\n"
        "Podpisujący          Rola                 Data i adres IP\n"
        "TOMASZ WIŚNIEWSKI    PREZES ZARZĄDU       2025-03-12 10:41:07   IP 31.175.12.10\n"
        "NORDIS LOGISTYKA SPÓŁKA Z OGRANICZONĄ ODPOWIEDZIALNOŚCIĄ\n"
        "PAWEŁ ZIELIŃSKI      PRACOWNIK            2025-03-12 11:02:55   IP 83.24.117.9\n"
        "Pieczęć elektroniczna: Autenti Sp. z o.o."
    ),
}


def get_vault() -> Vault:
    """Sejf trzymamy w st.session_state – przeżywa odświeżenia strony,
    ale jest osobny dla każdej karty przeglądarki i znika po jej zamknięciu."""
    if "vault" not in st.session_state:
        st.session_state.vault = Vault()
    return st.session_state.vault


def load_test_case(name: str) -> None:
    # Callback przycisku: ustawiamy wartość pola tekstowego PRZED jego
    # narysowaniem (Streamlit nie pozwala zmieniać widgetu po narysowaniu).
    st.session_state.input_text = TEST_CASES[name]
    # Tekst nie pochodzi już z wgranego PDF-a – czarne paski nałożymy
    # na PDF wygenerowany z tekstu.
    st.session_state.pop("source_pdf", None)
    st.session_state.pop("ocr_report", None)
    st.session_state.raw_pages = {"name": name, "pages": TEST_CASES[name].split(PAGE_BREAK)}


def load_uploaded_file(uploader_key: str, target_key: str) -> None:
    """Callback uploadera: wczytuje plik do wskazanego pola tekstowego.

    Działa jak load_test_case – zmieniamy wartość pola, zanim zostanie
    narysowane. Błąd zapisujemy w session_state, żeby pokazać go niżej.
    """
    uploaded = st.session_state.get(uploader_key)
    st.session_state.pop(f"{uploader_key}_error", None)
    if uploaded is None:  # użytkownik usunął plik – nic nie robimy
        return
    try:
        pages = read_uploaded_pages(uploaded.name, uploaded.getvalue())
    except FileReadError as error:
        st.session_state[f"{uploader_key}_error"] = f"{uploaded.name}: {error}"
        return
    st.session_state[target_key] = join_pages(pages)

    if target_key == "input_text":
        # Surowy tekst ze stron – do podglądu „co odczytał parser”.
        st.session_state.raw_pages = {"name": uploaded.name, "pages": pages}
        # Oryginalny PDF zachowujemy, żeby zaczernić dane W NIM (z zachowaniem
        # układu, logo, tabel), a nie w PDF-ie wygenerowanym z samego tekstu.
        name = uploaded.name.lower()
        if name.endswith(".pdf"):
            st.session_state.source_pdf = {"name": uploaded.name, "bytes": uploaded.getvalue()}
            # Wynik OCR jest już w pamięci (read_uploaded_pages go policzył).
            st.session_state.ocr_report = ocr_pdf(uploaded.getvalue())
        else:
            st.session_state.pop("source_pdf", None)
            if name.endswith(".docx"):
                st.session_state.ocr_report = ocr_docx(uploaded.getvalue())
            else:
                st.session_state.pop("ocr_report", None)


def ocr_review_reasons() -> list[str]:
    """Powody wstrzymania wyniku związane z OCR (obrazy, których nie odczytano)."""
    report = st.session_state.get("ocr_report")
    if report is None:
        return []
    reasons = []
    if report.pending:
        reasons.append(f"Plik zawiera obrazy z możliwym tekstem ({report.pending}), a OCR "
                       "jest niedostępny – dane na skanach i pieczątkach NIE zostały "
                       "sprawdzone. Zainstaluj Tesseract z językiem polskim.")
    if report.errors:
        reasons.append("OCR nie odczytał części obrazów: " + "; ".join(report.errors[:3]))
    return reasons


@st.cache_data(show_spinner="Zaczerniam dane w PDF...", max_entries=8)
def make_redacted_pdf(
    source_pdf: bytes | None, text: str, values: tuple[str, ...], redact_images: bool
):
    """Liczy zaczerniony PDF. Wynik jest zapamiętywany (cache), więc
    przełączanie checkboxa tam i z powrotem nie liczy wszystkiego od nowa."""
    if source_pdf:
        return redact_pdf(source_pdf, list(values), redact_images=redact_images)
    return redact_text_to_pdf(text, list(values))


def build_redacted_pdf(text: str, entities: list[dict]) -> None:
    """Zapamiętuje dane potrzebne do zaczernienia PDF (liczonego przy wyświetlaniu)."""
    source = st.session_state.get("source_pdf")
    st.session_state.redaction = {
        "source_pdf": source["bytes"] if source else None,
        "text": text,
        "values": tuple(entity["oryginał"] for entity in entities),
        "file_name": (source["name"].rsplit(".", 1)[0] + "_zaczerniony.pdf"
                      if source else "dokument_zaczerniony.pdf"),
    }


def show_upload_error(uploader_key: str) -> None:
    error = st.session_state.get(f"{uploader_key}_error")
    if error:
        st.error(error)


def show_vault_token(current_vault: Vault, key: str) -> None:
    """Pokazuje cały sejf jako JEDEN token (zamiast tabeli token -> oryginał).

    Wywoływane wewnątrz zwiniętej sekcji z tytułem – bez własnego nagłówka.
    """
    if not len(current_vault):
        st.caption("Sejf jest pusty – token pojawi się po pierwszej anonimizacji.")
        return
    token = current_vault.to_token()
    st.caption(f"Cały sejf ({len(current_vault)} mapowań) w jednym ciągu znaków. "
               "Wklej go w zakładce 2, żeby odszyfrować odpowiedź AI – także po zamknięciu "
               "karty albo na innym komputerze.")
    st.warning("Token to klucz do prawdziwych danych: kto go ma, odczyta oryginały. "
               "Nie wysyłaj go do ChatGPT ani nikomu, kto nie powinien ich widzieć.", icon="⚠️")
    st.code(token, language=None, wrap_lines=True)
    st.download_button("💾 Pobierz token (.txt)", data=token,
                       file_name="token_sejfu.txt", mime="text/plain", key=f"token_dl_{key}")


def use_masked_as_demo_response() -> None:
    result = st.session_state.get("last_result")
    if result:
        st.session_state.cloud_response = (
            "Oto streszczenie przygotowane przez zewnętrzne AI:\n\n" + result.masked_text
        )


def reset_vault() -> None:
    get_vault().clear()
    st.session_state.pop("last_result", None)
    st.session_state.pop("restored", None)
    st.session_state.pop("redaction", None)
    clear_ocr_cache()  # wyniki OCR zawierają dane wrażliwe


vault = get_vault()

# --- Pasek boczny: konfiguracja ----------------------------------------------
with st.sidebar:
    # Układ: na wierzchu tylko to, czego używa się przy każdym dokumencie
    # (etapy + stan). Rzadkie ustawienia są w zwiniętych sekcjach – pasek
    # mieści się na ekranie bez przewijania.
    st.header("⚙️ Ustawienia")
    status_box = st.container()  # wypełniany niżej, gdy znamy stan Ollamy

    use_ner = st.toggle("NER – osoby, firmy, adresy", value=ner_installed(),
                        disabled=not ner_installed(),
                        help=f"{ner_model_label()} (~0,3 mld parametrów, CPU). Szybko zaznacza osoby, "
                             "firmy, adresy i loginy; LLM sprawdza tekst po nim.")
    use_llm = st.toggle("LLM – kontekst i dane szczególne", value=True,
                        help="Mały model w Ollamie: stanowiska, identyfikacja pośrednia, "
                             "dane o zdrowiu, wyznaniu, związkach zawodowych…")
    if use_ner and ner_installed():
        # Wczytanie modelu z dysku trwa ok. 9 s – robimy to w tle od razu
        # po starcie, a nie przy pierwszym kliknięciu „Anonimizuj”.
        preload_ner_model()

    with st.expander("🙈 Zostaw jawnie"):
        st.caption("Wybrane dane NIE będą maskowane. Osoby, PESEL-e, adresy i pozostałe "
                   "dane szczególne są maskowane zawsze.")
        # Tylko kategorie, które same z siebie nie wskazują człowieka.
        keep_categories = [
            category for category, label in KEEP_OPTIONS.items()
            if st.checkbox(label, value=False, key=f"keep_{category}")
        ]
    if keep_categories:
        # Poza zwiniętą sekcją – odsłonięte dane muszą być widoczne zawsze.
        st.warning("Jawnie do AI: " + ", ".join(
            KEEP_OPTIONS[c].split(" ", 1)[1] for c in keep_categories), icon="👁️")

    with st.expander("🔧 Zaawansowane"):
        base_url = st.text_input("Adres Ollamy", value=OLLAMA_URL)
        model = st.text_input("Model (≤ 4 mld parametrów)", value=DEFAULT_MODEL)
        verify_ner = st.checkbox(
            "Eksperymentalne: LLM poprawia kategorie NER", value=False,
            disabled=not (use_ner and use_llm),
            help="LLM sprawdza kategorie niepewnych wykryć NER (nigdy ich nie odznacza). "
                 "Wyłączone domyślnie: qwen2.5:3b częściej psuł kategorie, niż je poprawiał.")

    st.divider()
    # Rezerwujemy miejsce i wypełniamy je na końcu skryptu – wtedy licznik
    # uwzględnia tokeny dodane w tym samym kliknięciu „Anonimizuj”.
    vault_counter = st.empty()
    st.button("🗑️ Wyczyść sejf", on_click=reset_vault, width="stretch")
    st.caption("🔒 Wszystko działa lokalnie. Dane nie opuszczają tego komputera.")

# --- Stan komponentów: jedna zwięzła lista u góry paska ---------------------
try:
    validate_ollama_url(base_url)
    address_error = None
except OllamaAddressError as error:
    address_error = str(error)
models = None if address_error else check_ollama(base_url)

with status_box:
    lines = []
    if address_error:
        lines.append("🔴 **Ollama:** adres spoza komputera – LLM wyłączony")
    elif models is None:
        lines.append("🔴 **Ollama** nie odpowiada – uruchom `ollama serve`")
    elif model not in models:
        lines.append(f"🟠 **{model}** nie jest pobrany – `ollama pull {model}`")
    else:
        lines.append(f"🟢 **LLM:** {model}")
    if not ner_installed():
        lines.append("⚪ **NER:** brak (`pip install -r requirements-ner.txt`)")
    elif NER_MODEL == ORIGINAL_NER_MODEL:
        # Polski v3 jest lepszy, ale trzeba go raz złożyć z nakładki LoRA.
        lines.append("🟢 **NER:** GLiNER2-PII (oryginał) · polski v3: "
                     "`python training/scal_nakladke.py`")
    else:
        lines.append(f"🟢 **NER:** {ner_model_label()}")
    lines.append("🟢 **OCR:** Tesseract" if ocr_available()
                 else "⚪ **OCR:** brak Tesseracta (pol)")

    # Limit hackathonu: LLM ≤ 4 mld parametrów. Pokazujemy też sumę z NER,
    # na wypadek gdyby jury liczyło wszystkie modele razem.
    llm_params = get_model_params(model, base_url) if use_llm and models else None
    if llm_params:
        total = llm_params + (NER_PARAMS if use_ner else 0)
        icon = "🔴" if llm_params > PARAM_LIMIT else ("🟠" if total > PARAM_LIMIT else "🟢")
        lines.append(f"{icon} **Parametry:** {total / 1e9:.2f} / {PARAM_LIMIT / 1e9:.0f} mld")
    st.markdown("  \n".join(lines))
    if address_error:
        st.error(address_error)
    elif llm_params and llm_params > PARAM_LIMIT:
        st.error(f"Sam LLM ma {llm_params / 1e9:.2f} mld parametrów – ponad limit hackathonu.")

st.title("🛡️ Lokalny Anonimizator PII")
st.caption("Regex + NER + mały LLM (Ollama) + OCR obrazów – wszystko lokalnie. "
           "Do AI w chmurze trafia tekst z tokenami, a odpowiedź wraca z prawdziwymi danymi.")

tab_mask, tab_restore = st.tabs(["1️⃣ Anonimizacja", "2️⃣ De-anonimizacja"])

# --- Zakładka 1: anonimizacja -------------------------------------------------
with tab_mask:
    # Wejście: plik albo przykład w jednym rzędzie (przykłady w menu, żeby
    # cztery długie przyciski nie zajmowały całej szerokości). Etykieta pola
    # jest ukryta – lista rozszerzeń i tak widnieje w samym polu – a oba
    # elementy są wyrównane do GÓRY: po wgraniu pliku pod polem pojawia się
    # jego nazwa i wyrównanie do dołu rozjeżdżało przycisk.
    col_file, col_examples = st.columns([4, 1], vertical_alignment="top")
    with col_file:
        st.file_uploader(
            f"Wczytaj plik ({', '.join(SUPPORTED_EXTENSIONS)})",
            type=SUPPORTED_EXTENSIONS,
            key="input_file",
            on_change=load_uploaded_file,
            args=("input_file", "input_text"),
            label_visibility="collapsed",
        )
    with col_examples, st.container(key="examples_box"):
        with st.popover("🧪 Przykłady", width="stretch"):
            st.caption("Trudne przypadki testowe")
            for name in TEST_CASES:
                st.button(name, on_click=load_test_case, args=(name,), width="stretch")
    show_upload_error("input_file")

    ocr_report = st.session_state.get("ocr_report")
    if ocr_report and ocr_report.pending:
        # Ostrzeżenie zostaje na wierzchu – dotyczy bezpieczeństwa danych.
        st.warning(f"{ocr_report.pending} obraz(ów) może zawierać tekst, ale OCR jest "
                   "niedostępny (brak Tesseracta z językiem polskim). Dane na skanach "
                   "i pieczątkach nie zostaną wykryte.")

    raw = st.session_state.get("raw_pages")
    if raw:
        pages = raw["pages"]
        with st.expander(f"🔍 Co odczytano z pliku · {raw['name']} · stron: {len(pages)}"):
            if ocr_report and ocr_report.images_total:
                parts = [f"obrazów: {ocr_report.images_total}", f"OCR: {ocr_report.images_ocr}"]
                if ocr_report.images_reused:
                    parts.append(f"z pamięci: {ocr_report.images_reused}")
                if ocr_report.skipped_text:
                    parts.append(f"mają już tekst: {ocr_report.skipped_text}")
                if ocr_report.skipped_small:
                    parts.append(f"za małe: {ocr_report.skipped_small}")
                st.caption("🖼️ " + " · ".join(parts)
                           + f" · czas OCR: {ocr_report.ocr_ms / 1000:.1f} s")
            st.caption(f"Znaków: {sum(len(page) for page in pages)}. Dokładnie to trafia do "
                       "Regexa, NER i (strona po stronie) do modelu.")
            for number, page in enumerate(pages, start=1):
                st.markdown(f"**Strona {number}** · {len(page)} znaków")
                st.code(page or "(pusta strona – skan bez tekstu?)",
                        language=None, wrap_lines=True)

    st.text_area("Tekst wejściowy (z danymi wrażliwymi)", key="input_text", height=200)

    if st.button("🔒 Anonimizuj", type="primary"):
        text = st.session_state.get("input_text", "")
        if not text.strip():
            st.info("Wklej tekst, wczytaj plik albo wybierz przykład.")
        else:
            # Pasek postępu: przy dłuższym pliku widać, że model pracuje
            # (fragment 3 z 7), zamiast samego kręcącego się kółka.
            progress = st.progress(0.0, text="Regex + NER + lokalny model analizują tekst...")

            def show_progress(done: int, total: int) -> None:
                progress.progress(done / total, text=f"LLM: fragment {done} z {total}")

            st.session_state.last_result = anonymize(
                text, vault, use_llm=use_llm, model=model, base_url=base_url,
                on_progress=show_progress, use_ner=use_ner, verify_ner=verify_ner,
                keep_categories=keep_categories,
            )
            progress.empty()
            st.session_state.last_result.review_reasons.extend(ocr_review_reasons())
            build_redacted_pdf(text, st.session_state.last_result.entities)

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        # Fail-closed: gdy LLM nie zadziałał albo dokument wygląda na prompt
        # injection, NIE pokazujemy tekstu „do wysłania” bez świadomej decyzji.
        # Ten komunikat jest PIERWSZY – nie może zginąć pod metrykami.
        show_output = True
        if result.needs_review:
            st.error("⛔ Wynik wymaga sprawdzenia przed wysłaniem do chmury:\n\n"
                     + "\n\n".join(f"- {reason}" for reason in result.review_reasons))
            show_output = st.checkbox(
                "Sprawdziłem tekst – pokaż wynik mimo to",
                value=False,
                key=f"review_ack_{id(result)}",
            )
        elif not use_llm:
            st.warning("Etap LLM wyłączony: dane wymagające kontekstu (stanowiska, dane "
                       "szczególne) nie są wykrywane. Przejrzyj tekst przed wysłaniem.")
        if result.ner_error:
            st.info(f"Etap NER pominięty: {result.ner_error}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Wykryte dane", len(result.entities))
        m2.metric("Regex", f"{result.regex_ms:.1f} ms")
        m3.metric("NER", f"{result.ner_ms / 1000:.2f} s"
                  if use_ner and not result.ner_error else "wył.")
        m4.metric("LLM", f"{result.llm_ms / 1000:.1f} s" if use_llm else "wył.")
        ram = get_process_peak_ram_mb()
        model_ram = get_model_memory_mb(model, base_url)
        st.caption(" · ".join(part for part in [
            f"RAM aplikacji: {ram:.0f} MB" if ram else "",
            f"RAM modelu LLM: {model_ram:.0f} MB" if model_ram else "",
        ] if part))

        tab_text, tab_pdf, tab_found = st.tabs(
            ["📝 Tekst do wysłania", "⬛ PDF z paskami",
             f"📋 Wykryte dane ({len(result.entities)})"])

        with tab_text:
            if show_output:
                # st.code daje przycisk „kopiuj” w prawym górnym rogu.
                st.code(result.masked_text, language=None, wrap_lines=True)
                st.download_button(
                    "💾 Pobierz zanonimizowany tekst (.txt)",
                    data=result.masked_text,
                    file_name="zanonimizowany.txt",
                    mime="text/plain",
                )
                with st.expander("🔑 Token sejfu – zapisz, żeby później odszyfrować odpowiedź"):
                    show_vault_token(vault, key="tab1")
            else:
                st.info("Najpierw sprawdź tekst i potwierdź powyżej.")

        with tab_pdf:
            redaction = st.session_state.get("redaction")
            if not show_output:
                st.info("Najpierw sprawdź tekst i potwierdź powyżej.")
            elif redaction:
                from_source = redaction["source_pdf"] is not None
                # Obrazy istnieją tylko w wgranym PDF – w PDF-ie z tekstu ich nie ma.
                redact_images = st.checkbox(
                    "Zamazać również podpisy odręczne, pieczątki i obrazy",
                    value=True,
                    disabled=not from_source,
                    help="Wycina z PDF wszystkie osadzone obrazy (podpis, pieczęć, kod QR, "
                         "logo) i zostawia w ich miejscu czarny prostokąt. Zaczernia też podpisy "
                         "narysowane liniami, a na skanach – pismo odręczne przy słowie "
                         "„podpis”. Dotyczy wgranego PDF.",
                )
                redacted = make_redacted_pdf(
                    redaction["source_pdf"], redaction["text"], redaction["values"], redact_images
                )
                if from_source:
                    source_info = ("oryginalny PDF, strony spłaszczone do obrazu – "
                                   "w pliku nie ma tekstu, który dałoby się skopiować")
                    if redact_images:
                        source_info += f" · usuniętych obrazów: {redacted.images}"
                        signature_count = redacted.vector_signatures + redacted.scan_signatures
                        if signature_count:
                            source_info += f" · podpisów odręcznych: {signature_count}"
                else:
                    source_info = ("PDF z tekstu – w miejscu danych są tylko paski, "
                                   "prawdziwe wartości nie trafiły do pliku")
                st.caption(f"Pasków: {redacted.bars} · {source_info}.")
                if from_source and redact_images and redacted.background_pages:
                    st.info(f"Stron będących skanem: {redacted.background_pages}. Obraz całej "
                            "strony nie jest zamazywany (strona byłaby czarna). Paski na tekście "
                            "działają dzięki warstwie OCR, a podpis odręczny WEWNĄTRZ skanu jest "
                            "zaczerniany tylko przy słowie „podpis” – sprawdź podgląd.")
                if redacted.missing:
                    st.warning("Nie znaleziono w PDF (sprawdź ręcznie): "
                               + ", ".join(redacted.missing))
                st.download_button(
                    "⬛ Pobierz zaczerniony PDF",
                    data=redacted.pdf,
                    file_name=redaction["file_name"],
                    mime="application/pdf",
                )
                pages = page_count(redacted.pdf)
                page = 1
                if pages > 1:
                    page = st.number_input("Strona podglądu", min_value=1, max_value=pages, value=1)
                st.image(render_preview(redacted.pdf, page - 1))

        with tab_found:
            if result.entities:
                st.dataframe(result.entities, hide_index=True, width="stretch")
            else:
                st.write("Nie wykryto danych wrażliwych.")
            if result.kept_entities:
                st.markdown("**👁️ Zostawione jawnie (na Twoje życzenie)**")
                st.caption("Wykryte, ale NIE zamaskowane i bez czarnego paska w PDF – "
                           "zmień w pasku bocznym „Zostaw jawnie”.")
                st.dataframe(result.kept_entities, hide_index=True, width="stretch")

# --- Zakładka 2: de-anonimizacja ---------------------------------------------
with tab_restore:
    st.caption("Wklej odpowiedź z ChatGPT / innego AI zawierającą tokeny "
               "(np. `[OSOBA_1]`). Aplikacja podstawi oryginalne dane z lokalnego sejfu.")
    col_file, col_demo = st.columns([4, 1], vertical_alignment="bottom")
    with col_file:
        st.file_uploader(
            "Wczytaj odpowiedź AI z pliku",
            type=SUPPORTED_EXTENSIONS,
            key="response_file",
            on_change=load_uploaded_file,
            args=("response_file", "cloud_response"),
        )
    with col_demo:
        st.button("🧪 Demo", on_click=use_masked_as_demo_response,
                  disabled="last_result" not in st.session_state, width="stretch",
                  help="Wstawia zanonimizowany tekst jako „odpowiedź AI” – do pokazu.")
    show_upload_error("response_file")

    st.text_area("Odpowiedź z chmury", key="cloud_response", height=200)
    st.text_input(
        "🔑 Token sejfu (opcjonalnie)",
        key="vault_token_input",
        type="password",
        placeholder="PIIVAULT1.…  – wklej, jeśli wracasz po zamknięciu karty",
        help="Puste pole = użyj sejfu z tej sesji. Token zastępuje tabelę mapowań: "
             "zawiera cały sejf w jednym ciągu znaków.",
    )

    if st.button("🔓 Przywróć dane", type="primary"):
        response = st.session_state.get("cloud_response", "")
        token_text = st.session_state.get("vault_token_input", "").strip()
        try:
            # Token z pola ma pierwszeństwo przed sejfem z bieżącej sesji.
            source_vault = Vault.from_token(token_text) if token_text else vault
        except VaultTokenError as error:
            st.error(f"Niepoprawny token sejfu: {error}")
        else:
            # Zapamiętujemy wynik – kliknięcie „Pobierz” odświeża stronę
            # i bez tego przywrócony tekst by zniknął.
            st.session_state.restored = {
                "text": source_vault.restore(response),
                "unknown": source_vault.unknown_tokens(response),
            }

    restored = st.session_state.get("restored")
    if restored:
        st.subheader("Odpowiedź z przywróconymi danymi")
        st.code(restored["text"], language=None, wrap_lines=True)
        st.download_button(
            "💾 Pobierz przywrócony tekst (.txt)",
            data=restored["text"],
            file_name="przywrocony.txt",
            mime="text/plain",
        )
        if restored["unknown"]:
            st.warning("Tokeny nieobecne w sejfie (AI mogło je wymyślić): "
                       + ", ".join(restored["unknown"]))

    with st.expander("🔑 Token sejfu z tej sesji"):
        show_vault_token(vault, key="tab2")

vault_counter.metric("Tokenów w sejfie", len(vault))
