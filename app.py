"""Interfejs webowy (Streamlit): anonimizacja i przywracanie danych.

Uruchomienie:  streamlit run app.py

Wygląd: motyw w .streamlit/config.toml, komponenty w ui.py.
"""

import time

import streamlit as st

import ui
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
    "ZDROWIE": "Diagnozy i leczenie",
    "STANOWISKO": "Stanowiska",
    "KWOTA": "Kwoty",
    "DATA": "Daty",
    "FIRMA": "Firmy i instytucje",
}
assert set(KEEP_OPTIONS) == KEEPABLE_CATEGORIES, "KEEP_OPTIONS musi odpowiadać KEEPABLE_CATEGORIES"

st.set_page_config(page_title="Anonimizator PII", page_icon=":material/shield_lock:",
                   layout="wide", initial_sidebar_state="expanded")
ui.apply_styles()

# --- Przypadki testowe ----------------------------------------------------------
# Celowo zawierają: odmianę nazwisk, różne formaty numerów, dane „ukryte”
# w logach i liczby, które NIE są danymi osobowymi (pułapki na false positive).
# Wartość: (ikona, tekst).
TEST_CASES = {
    "Dokument medyczny": ("medical_information", (
        "KARTA INFORMACYJNA LECZENIA SZPITALNEGO\n"
        "Szpital Specjalistyczny im. św. Łukasza w Krakowie, Oddział Kardiologii\n\n"
        "Pacjent: Jan Kowalski, PESEL 44051401359, zam. ul. Długa 15/3, 31-147 Kraków.\n"
        "Tel. kontaktowy: +48 601 234 567, e-mail: jan.kowalski@poczta.pl\n"
        "Osoba upoważniona: Anna Kowalska (córka), tel. 512-345-678.\n\n"
        "Rozpoznanie: migotanie przedsionków (I48). Pacjent przyjęty 12.03.2025, "
        "skierowany przez lek. Piotra Wiśniewskiego. Stan Jana Kowalskiego po "
        "kardiowersji stabilny. Zalecono kontrolę za 3 miesiące.\n"
        "Koszt leczenia poza NFZ: 4 850,00 zł. Nr księgi głównej: 2025/00417."
    )),
    "Umowa B2B": ("handshake", (
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
    )),
    "Log z błędem": ("terminal", (
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
    )),
    # „\f” = nowa strona (tak jak w tekście wyciągniętym z PDF). Ostatnia
    # strona to Karta Podpisów: nazwy WIELKIMI literami, UUID, daty i IP.
    "Umowa z e-podpisem (3 strony)": ("signature", (
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
    )),
}


# --- Stan i akcje -------------------------------------------------------------------

def get_vault() -> Vault:
    """Sejf trzymamy w st.session_state – przeżywa odświeżenia strony,
    ale jest osobny dla każdej karty przeglądarki i znika po jej zamknięciu."""
    if "vault" not in st.session_state:
        st.session_state.vault = Vault()
    return st.session_state.vault


def load_test_case(name: str) -> None:
    # Callback przycisku: ustawiamy wartość pola tekstowego PRZED jego
    # narysowaniem (Streamlit nie pozwala zmieniać widgetu po narysowaniu).
    text = TEST_CASES[name][1]
    st.session_state.input_text = text
    # Tekst nie pochodzi już z wgranego PDF-a – czarne paski nałożymy
    # na PDF wygenerowany z tekstu.
    st.session_state.pop("source_pdf", None)
    st.session_state.pop("ocr_report", None)
    st.session_state.raw_pages = {"name": name, "pages": text.split(PAGE_BREAK)}


def clear_input() -> None:
    st.session_state.input_text = ""
    for key in ("source_pdf", "ocr_report", "raw_pages", "last_result", "redaction"):
        st.session_state.pop(key, None)


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


@st.cache_data(show_spinner="Zaczerniam dane w PDF…", max_entries=8)
def make_redacted_pdf(
    source_pdf: bytes | None, text: str, values: tuple[str, ...], redact_images: bool
):
    """Liczy zaczerniony PDF. Wynik jest zapamiętywany (cache), więc
    przełączanie opcji tam i z powrotem nie liczy wszystkiego od nowa."""
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
        st.error(error, icon=":material/error:")


def use_masked_as_demo_response() -> None:
    result = st.session_state.get("last_result")
    if result:
        st.session_state.cloud_response = (
            "Oto streszczenie przygotowane przez zewnętrzne AI:\n\n" + result.masked_text
        )


def reset_vault() -> None:
    get_vault().clear()
    for key in ("last_result", "restored", "redaction"):
        st.session_state.pop(key, None)
    clear_ocr_cache()  # wyniki OCR zawierają dane wrażliwe


# --- Okna dialogowe ------------------------------------------------------------------

@st.dialog("Token sejfu", width="large")
def vault_token_dialog() -> None:
    """Cały sejf jako JEDEN token (zamiast tabeli token -> oryginał)."""
    current_vault = get_vault()
    if not len(current_vault):
        st.markdown("Sejf jest pusty. Token pojawi się po pierwszej anonimizacji.")
        return
    token = current_vault.to_token()
    st.markdown(f"Cały sejf ({len(current_vault)} mapowań) w jednym ciągu znaków. Wklej go "
                "na karcie **Przywracanie danych**, żeby odszyfrować odpowiedź AI – także po "
                "zamknięciu przeglądarki albo na innym komputerze.")
    st.warning("Token to klucz do prawdziwych danych: kto go ma, odczyta oryginały. Nie wysyłaj "
               "go do zewnętrznego AI ani nikomu, kto nie powinien ich widzieć.",
               icon=":material/key:")
    st.code(token, language=None, wrap_lines=True)
    st.download_button("Pobierz token (.txt)", data=token, file_name="token_sejfu.txt",
                       mime="text/plain", icon=":material/download:", key="token_download")


@st.dialog("Wyczyścić sejf?")
def clear_vault_dialog() -> None:
    st.markdown("Wszystkie mapowania tokenów na prawdziwe dane zostaną usunięte z tej sesji. "
                "Odpowiedzi AI z tokenami będzie można przywrócić tylko z zapisanym wcześniej "
                "tokenem sejfu.")
    cancel, confirm = st.columns(2)
    if cancel.button("Anuluj", width="stretch"):
        st.rerun()
    with confirm, st.container(key="danger_confirm_clear"):
        if st.button("Wyczyść sejf", icon=":material/delete:", width="stretch"):
            reset_vault()
            st.toast("Sejf wyczyszczony.", icon=":material/delete:")
            st.rerun()


vault = get_vault()

# --- Pasek boczny ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("### :material/shield_lock: Anonimizator PII")
    st.caption("Dane osobowe zamieniane na tokeny przed wysłaniem do AI. Wszystko lokalnie.")

    st.markdown("##### Stan systemu")
    status_box = st.container(key="status_list")  # wypełniany niżej, gdy znamy stan Ollamy

    st.markdown("##### Etapy wykrywania")
    use_ner = st.toggle("NER – osoby, firmy, adresy", value=ner_installed(),
                        disabled=not ner_installed(),
                        help=f"{ner_model_label()} (~0,3 mld parametrów, CPU). Szybko zaznacza osoby, "
                             "firmy, adresy i loginy; LLM sprawdza tekst po nim.")
    use_llm = st.toggle("LLM – kontekst i dane szczególne", value=True,
                        help="Mały model w Ollamie: stanowiska, identyfikacja pośrednia, "
                             "dane o zdrowiu, wyznaniu, związkach zawodowych.")
    if use_ner and ner_installed():
        # Wczytanie modelu z dysku trwa ok. 9 s – robimy to w tle od razu
        # po starcie, a nie przy pierwszym kliknięciu „Anonimizuj”.
        preload_ner_model()

    with st.expander("Zostaw jawnie", icon=":material/visibility:"):
        st.caption("Wybrane kategorie NIE będą maskowane. Osoby, numery, adresy i pozostałe "
                   "dane szczególne są maskowane zawsze.")
        # Tylko kategorie, które same z siebie nie wskazują człowieka.
        keep_categories = [
            category for category, label in KEEP_OPTIONS.items()
            if st.checkbox(label, value=False, key=f"keep_{category}")
        ]
    if keep_categories:
        # Poza zwiniętą sekcją – odsłonięte dane muszą być widoczne zawsze.
        st.warning("Jawnie do AI: " + ", ".join(KEEP_OPTIONS[c].lower() for c in keep_categories),
                   icon=":material/visibility:")

    with st.expander("Zaawansowane", icon=":material/tune:"):
        base_url = st.text_input("Adres Ollamy", value=OLLAMA_URL)
        model = st.text_input("Model (≤ 4 mld parametrów)", value=DEFAULT_MODEL)
        verify_ner = st.checkbox(
            "LLM poprawia kategorie NER (eksperymentalne)", value=False,
            disabled=not (use_ner and use_llm),
            help="LLM sprawdza kategorie niepewnych wykryć NER (nigdy ich nie odznacza). "
                 "Wyłączone domyślnie: qwen2.5:3b częściej psuł kategorie, niż je poprawiał.")

    with st.container(key="card_vault"):
        # Miejsce rezerwujemy i wypełniamy na końcu skryptu – wtedy licznik
        # uwzględnia tokeny dodane w tym samym kliknięciu „Anonimizuj”.
        vault_counter = st.empty()
        vault_counter.metric("Tokeny w sejfie", len(vault))
        # Jeden pod drugim – w wąskim panelu dwa obok siebie łamały etykiety.
        if st.button("Pokaż token sejfu", icon=":material/key:", width="stretch",
                     help="Token jest potrzebny, żeby przywrócić dane później."):
            vault_token_dialog()
        with st.container(key="danger_clear"):
            if st.button("Wyczyść sejf", icon=":material/delete:", width="stretch",
                         help="Usuń wszystkie mapowania z tej sesji."):
                clear_vault_dialog()

    st.caption(":material/lock: Dane nie opuszczają tego komputera: modele działają lokalnie, "
               "a aplikacja jest dostępna tylko pod adresem localhost.")

# --- Stan komponentów: wypełnienie listy u góry paska -----------------------------------
try:
    validate_ollama_url(base_url)
    address_error = None
except OllamaAddressError as error:
    address_error = str(error)
models = None if address_error else check_ollama(base_url)
llm_params = get_model_params(model, base_url) if use_llm and models else None

with status_box:
    if address_error:
        llm_line = ui.status_line("error", "LLM", "adres spoza komputera – wyłączony")
    elif models is None:
        llm_line = ui.status_line("error", "LLM", "Ollama nie odpowiada – uruchom `ollama serve`")
    elif model not in models:
        llm_line = ui.status_line("warn", "LLM", f"brak modelu – `ollama pull {model}`")
    elif not use_llm:
        llm_line = ui.status_line("off", "LLM", f"{model}, wyłączony")
    else:
        llm_line = ui.status_line("ok", "LLM", model)
    if not ner_installed():
        ner_line = ui.status_line("off", "NER", "nie zainstalowany")
    elif NER_MODEL == ORIGINAL_NER_MODEL:
        ner_line = ui.status_line("ok", "NER", "GLiNER2-PII (oryginał)")
    else:
        ner_line = ui.status_line("ok" if use_ner else "off", "NER", ner_model_label())
    ocr_line = (ui.status_line("ok", "OCR", "Tesseract") if ocr_available()
                else ui.status_line("warn", "OCR", "brak modeli Tesseracta (pol)"))
    lines = [llm_line, ner_line, ocr_line]
    # Limit hackathonu: LLM ≤ 4 mld parametrów. Pokazujemy też sumę z NER,
    # na wypadek gdyby jury liczyło wszystkie modele razem.
    if llm_params:
        total = llm_params + (NER_PARAMS if use_ner else 0)
        state = "error" if llm_params > PARAM_LIMIT else ("warn" if total > PARAM_LIMIT else "ok")
        lines.append(ui.status_line(state, "Parametry",
                                    f"{total / 1e9:.2f} z {PARAM_LIMIT / 1e9:.0f} mld"))
    st.markdown("  \n".join(lines))
    if address_error:
        st.error(address_error, icon=":material/error:")
    elif llm_params and llm_params > PARAM_LIMIT:
        st.error(f"Sam LLM ma {llm_params / 1e9:.2f} mld parametrów – ponad limit.",
                 icon=":material/error:")
    if ner_installed() and NER_MODEL == ORIGINAL_NER_MODEL:
        st.caption("Polski model NER (dokładniejszy): `python training/scal_nakladke.py`")

# --- Nagłówek i nawigacja ---------------------------------------------------------------
ui.page_header("Anonimizacja dokumentów",
               "Regex, NER i mały model językowy wykrywają dane osobowe. Do AI w chmurze trafia "
               "tekst z tokenami, a odpowiedź wraca z prawdziwymi danymi.")

tab_mask, tab_restore = st.tabs([":material/shield: Anonimizacja",
                                 ":material/settings_backup_restore: Przywracanie danych"])

# --- Karta 1: anonimizacja ----------------------------------------------------------------
with tab_mask:
    with st.container(key="card_input"):
        ui.section("description", "Dokument",
                   "Wczytaj plik, wklej tekst albo wybierz przykład. Plik jest czytany w pamięci "
                   "i nigdzie nie zapisywany.")
        # Plik i przykłady w jednym rzędzie, wyrównane do GÓRY: po wgraniu pliku
        # pod polem pojawia się jego nazwa i wyrównanie do dołu rozjeżdżało przycisk.
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
            with st.popover("Przykłady", icon=":material/science:", width="stretch"):
                st.caption("Trudne przypadki testowe")
                for name, (icon, _) in TEST_CASES.items():
                    st.button(name, icon=f":material/{icon}:", on_click=load_test_case,
                              args=(name,), width="stretch", type="tertiary")
        show_upload_error("input_file")

        ocr_report = st.session_state.get("ocr_report")
        if ocr_report and ocr_report.pending:
            # Ostrzeżenie zostaje na wierzchu – dotyczy bezpieczeństwa danych.
            st.warning(f"{ocr_report.pending} obraz(ów) może zawierać tekst, ale OCR jest "
                       "niedostępny (brak Tesseracta z językiem polskim). Dane na skanach "
                       "i pieczątkach nie zostaną wykryte.", icon=":material/image_not_supported:")

        st.text_area("Treść dokumentu", key="input_text", height=220,
                     placeholder="Wklej tekst z danymi osobowymi albo wczytaj plik powyżej.")

        raw = st.session_state.get("raw_pages")
        if raw:
            pages = raw["pages"]
            with st.expander(f"Co odczytano z pliku · {raw['name']} · stron: {len(pages)}",
                             icon=":material/plagiarism:"):
                if ocr_report and ocr_report.images_total:
                    parts = [f"obrazów: {ocr_report.images_total}", f"OCR: {ocr_report.images_ocr}"]
                    if ocr_report.images_reused:
                        parts.append(f"z pamięci: {ocr_report.images_reused}")
                    if ocr_report.skipped_text:
                        parts.append(f"mają już tekst: {ocr_report.skipped_text}")
                    if ocr_report.skipped_small:
                        parts.append(f"za małe: {ocr_report.skipped_small}")
                    st.caption(":material/image: " + " · ".join(parts)
                               + f" · czas OCR: {ocr_report.ocr_ms / 1000:.1f} s")
                st.caption(f"Znaków: {sum(len(page) for page in pages)}. Dokładnie to trafia do "
                           "Regexa, NER i (strona po stronie) do modelu.")
                for number, page in enumerate(pages, start=1):
                    st.markdown(f"**Strona {number}** · {len(page)} znaków")
                    st.code(page or "(pusta strona – skan bez tekstu?)",
                            language=None, wrap_lines=True)

        col_run, col_clear, _ = st.columns([2, 1.3, 4])
        run = col_run.button("Anonimizuj", type="primary", icon=":material/lock:", width="stretch")
        col_clear.button("Wyczyść", type="tertiary", icon=":material/close:", on_click=clear_input,
                         help="Usuń tekst i wynik (sejf zostaje).")

    if run:
        text = st.session_state.get("input_text", "")
        if not text.strip():
            st.info("Wklej tekst, wczytaj plik albo wybierz przykład.", icon=":material/info:")
        else:
            started = time.perf_counter()
            stages = "Regex" + (" · NER" if use_ner else "") + (" · LLM" if use_llm else "")
            with st.status(f"Analizuję dokument ({stages})…", expanded=True) as status:
                progress = st.progress(0.0, text="Regex i NER…")
                hint = st.container(key="progress_hint").empty()
                result = ui.run_with_progress(
                    lambda on_progress: anonymize(
                        text, vault, use_llm=use_llm, model=model, base_url=base_url,
                        on_progress=on_progress, use_ner=use_ner, verify_ner=verify_ner,
                        keep_categories=keep_categories,
                    ),
                    progress, hint, use_llm,
                )
                hint.empty()
                result.review_reasons.extend(ocr_review_reasons())
                elapsed = time.perf_counter() - started
                status.update(label=f"Gotowe · {len(result.entities)} danych w {elapsed:.1f} s",
                              state="error" if result.needs_review else "complete",
                              expanded=False)
            st.session_state.last_result = result
            build_redacted_pdf(text, result.entities)
            if not result.needs_review:
                st.toast(f"Zanonimizowano {len(result.entities)} danych.", icon=":material/check_circle:")

    result = st.session_state.get("last_result")
    if not result:
        ui.empty_state("empty_mask", "shield", "Wynik pojawi się tutaj",
                       "Po anonimizacji zobaczysz tekst z tokenami gotowy do wysłania, PDF "
                       "z zaczernionymi danymi i listę wykrytych danych.")
    else:
        # Fail-closed: gdy LLM nie zadziałał albo dokument wygląda na prompt
        # injection, NIE pokazujemy tekstu „do wysłania” bez świadomej decyzji.
        # Ten komunikat jest PIERWSZY – nie może zginąć pod metrykami.
        show_output = True
        if result.needs_review:
            st.error("**Wynik wymaga sprawdzenia przed wysłaniem do chmury**\n\n"
                     + "\n".join(f"- {reason}" for reason in result.review_reasons),
                     icon=":material/gpp_maybe:")
            show_output = st.checkbox("Sprawdziłem tekst – pokaż wynik mimo to", value=False,
                                      key=f"review_ack_{id(result)}")
        elif not use_llm:
            st.warning("Etap LLM wyłączony: dane wymagające kontekstu (stanowiska, dane "
                       "szczególne) nie są wykrywane. Przejrzyj tekst przed wysłaniem.",
                       icon=":material/warning:")
        if result.ner_error:
            st.info(f"Etap NER pominięty: {result.ner_error}", icon=":material/info:")

        kpis = [
            ("Wykryte dane", str(len(result.entities))),
            ("Regex", f"{result.regex_ms:.1f} ms"),
            ("NER", f"{result.ner_ms / 1000:.2f} s" if use_ner and not result.ner_error else "wyłączony"),
            ("Model językowy", f"{result.llm_ms / 1000:.1f} s" if use_llm else "wyłączony"),
        ]
        for index, (column, (label, value)) in enumerate(zip(st.columns(4), kpis)):
            with column, st.container(key=f"kpi_{index}"):
                st.metric(label, value)
        ram = get_process_peak_ram_mb()
        model_ram = get_model_memory_mb(model, base_url)
        memory = " · ".join(part for part in [
            f"pamięć aplikacji: {ram:.0f} MB" if ram else "",
            f"pamięć modelu językowego: {model_ram:.0f} MB" if model_ram else "",
        ] if part)
        if memory:
            st.caption(memory[0].upper() + memory[1:])

        with st.container(key="card_result"):
            tab_text, tab_pdf, tab_found = st.tabs(
                [":material/article: Tekst do wysłania", ":material/picture_as_pdf: PDF z zaczernieniem",
                 f":material/table_view: Wykryte dane ({len(result.entities)})"])

            with tab_text:
                if show_output:
                    st.caption("Tekst z tokenami zamiast danych. Skopiuj go przyciskiem w rogu pola "
                               "albo pobierz jako plik.")
                    # st.code daje przycisk „kopiuj” w prawym górnym rogu.
                    st.code(ui.page_break_view(result.masked_text, PAGE_BREAK),
                            language=None, wrap_lines=True)
                    col_dl, col_token, _ = st.columns([1.6, 1.4, 3])
                    col_dl.download_button("Pobierz tekst (.txt)", data=result.masked_text,
                                           file_name="zanonimizowany.txt", mime="text/plain",
                                           icon=":material/download:", width="stretch")
                    if col_token.button("Token sejfu", icon=":material/key:", width="stretch",
                                        key="token_from_result"):
                        vault_token_dialog()
                else:
                    st.info("Najpierw sprawdź tekst i potwierdź powyżej.", icon=":material/info:")

            with tab_pdf:
                redaction = st.session_state.get("redaction")
                if not show_output:
                    st.info("Najpierw sprawdź tekst i potwierdź powyżej.", icon=":material/info:")
                elif redaction:
                    from_source = redaction["source_pdf"] is not None
                    # Obrazy istnieją tylko w wgranym PDF – w PDF-ie z tekstu ich nie ma.
                    redact_images = st.toggle(
                        "Zamaż też podpisy odręczne, pieczątki i obrazy",
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
                                "działają dzięki warstwie OCR, a podpis odręczny wewnątrz skanu jest "
                                "zaczerniany tylko przy słowie „podpis” – sprawdź podgląd.",
                                icon=":material/document_scanner:")
                    if redacted.missing:
                        st.warning("Nie znaleziono w PDF (sprawdź ręcznie): "
                                   + ", ".join(redacted.missing), icon=":material/search_off:")
                    pages = page_count(redacted.pdf)
                    col_dl, col_page, _ = st.columns([1.6, 1.4, 3], vertical_alignment="bottom")
                    col_dl.download_button("Pobierz PDF", data=redacted.pdf,
                                           file_name=redaction["file_name"], mime="application/pdf",
                                           icon=":material/download:", width="stretch")
                    page = 1
                    if pages > 1:
                        page = col_page.number_input("Strona podglądu", min_value=1, max_value=pages,
                                                     value=1)
                    st.image(render_preview(redacted.pdf, page - 1))

            with tab_found:
                if result.entities:
                    st.dataframe(result.entities, hide_index=True, width="stretch")
                else:
                    st.markdown("Nie wykryto danych wrażliwych.")
                if result.kept_entities:
                    st.markdown("##### :material/visibility: Zostawione jawnie")
                    st.caption("Wykryte, ale NIE zamaskowane i bez czarnego paska w PDF – zgodnie "
                               "z ustawieniem „Zostaw jawnie” w panelu bocznym.")
                    st.dataframe(result.kept_entities, hide_index=True, width="stretch")

# --- Karta 2: przywracanie danych -----------------------------------------------------------
with tab_restore:
    with st.container(key="card_restore"):
        ui.section("settings_backup_restore", "Odpowiedź z AI",
                   "Wklej odpowiedź z ChatGPT lub innego AI zawierającą tokeny (np. [OSOBA_1]). "
                   "Aplikacja podstawi oryginalne dane z lokalnego sejfu.")
        col_file, col_demo = st.columns([4, 1], vertical_alignment="top")
        with col_file:
            st.file_uploader(
                "Wczytaj odpowiedź AI z pliku",
                type=SUPPORTED_EXTENSIONS,
                key="response_file",
                on_change=load_uploaded_file,
                args=("response_file", "cloud_response"),
                label_visibility="collapsed",
            )
        with col_demo, st.container(key="examples_box_restore"):
            st.button("Wstaw przykład", icon=":material/science:",
                      on_click=use_masked_as_demo_response,
                      disabled="last_result" not in st.session_state, width="stretch",
                      help="Wstawia przykładową odpowiedź AI zbudowaną z Twojego ostatniego "
                           "zanonimizowanego tekstu – żeby sprawdzić przywracanie bez wysyłania "
                           "czegokolwiek do chmury. Dostępne po anonimizacji.")
        show_upload_error("response_file")

        st.text_area("Odpowiedź z tokenami", key="cloud_response", height=200,
                     placeholder="Oto streszczenie umowy: [OSOBA_1] zobowiązuje się…")
        st.text_input(
            "Token sejfu (opcjonalnie)",
            key="vault_token_input",
            type="password",
            placeholder="PIIVAULT1.… – wklej, jeśli wracasz po zamknięciu przeglądarki",
            help="Puste pole = sejf z tej sesji. Token zawiera cały sejf w jednym ciągu znaków.",
        )
        col_run, _ = st.columns([2, 5.3])
        restore_clicked = col_run.button("Przywróć dane", type="primary",
                                         icon=":material/lock_open:", width="stretch")

    if restore_clicked:
        response = st.session_state.get("cloud_response", "")
        token_text = st.session_state.get("vault_token_input", "").strip()
        try:
            # Token z pola ma pierwszeństwo przed sejfem z bieżącej sesji.
            source_vault = Vault.from_token(token_text) if token_text else vault
        except VaultTokenError as error:
            st.error(f"Niepoprawny token sejfu: {error}", icon=":material/key_off:")
        else:
            # Zapamiętujemy wynik – kliknięcie „Pobierz” odświeża stronę
            # i bez tego przywrócony tekst by zniknął.
            st.session_state.restored = {
                "text": source_vault.restore(response),
                "unknown": source_vault.unknown_tokens(response),
            }

    restored = st.session_state.get("restored")
    if not restored:
        ui.empty_state("empty_restore", "settings_backup_restore", "Przywrócony tekst pojawi się tutaj",
                       "Tokeny z odpowiedzi AI zostaną zamienione na prawdziwe dane – lokalnie, "
                       "bez wysyłania czegokolwiek.")
    else:
        with st.container(key="card_restored"):
            ui.section("task_alt", "Odpowiedź z przywróconymi danymi")
            if restored["unknown"]:
                st.warning("Tokeny nieobecne w sejfie (AI mogło je wymyślić): "
                           + ", ".join(restored["unknown"]), icon=":material/help:")
            st.code(ui.page_break_view(restored["text"], PAGE_BREAK), language=None, wrap_lines=True)
            col_dl, _ = st.columns([1.6, 4.4])
            col_dl.download_button("Pobierz tekst (.txt)", data=restored["text"],
                                   file_name="przywrocony.txt", mime="text/plain",
                                   icon=":material/download:", width="stretch")

with vault_counter.container():
    st.metric("Tokeny w sejfie", len(vault))
