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
from engine.llm_masker import DEFAULT_MODEL, OLLAMA_URL, check_ollama, get_model_memory_mb
from engine.pipeline import anonymize, get_process_peak_ram_mb
from engine.redactor import page_count, redact_pdf, redact_text_to_pdf, render_preview
from engine.vault import Vault, VaultTokenError

st.set_page_config(page_title="Lokalny Anonimizator PII", page_icon="🛡️", layout="wide")

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
        if uploaded.name.lower().endswith(".pdf"):
            st.session_state.source_pdf = {"name": uploaded.name, "bytes": uploaded.getvalue()}
        else:
            st.session_state.pop("source_pdf", None)


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
    """Pokazuje cały sejf jako JEDEN token (zamiast tabeli token -> oryginał)."""
    st.subheader("🔑 Token sejfu")
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


vault = get_vault()

# --- Pasek boczny: konfiguracja ----------------------------------------------
with st.sidebar:
    st.header("⚙️ Konfiguracja")
    base_url = st.text_input("Adres Ollamy", value=OLLAMA_URL)
    model = st.text_input("Model (≤ 4B parametrów)", value=DEFAULT_MODEL)
    use_llm = st.checkbox("Etap 2: wykrywanie kontekstowe (LLM)", value=True)

    models = check_ollama(base_url)
    if models is None:
        st.error("Ollama nie odpowiada – działa tylko Regex.\n\nUruchom: `ollama serve`")
    elif model not in models:
        st.warning(f"Model nie jest pobrany. Uruchom:\n\n`ollama pull {model}`")
    else:
        st.success(f"Ollama OK · {model}")

    st.divider()
    # Rezerwujemy miejsce i wypełniamy je na końcu skryptu – wtedy licznik
    # uwzględnia tokeny dodane w tym samym kliknięciu „Anonimizuj”.
    vault_counter = st.empty()
    st.button("🗑️ Wyczyść sejf", on_click=reset_vault)
    st.caption("Wszystko działa lokalnie. Dane nie opuszczają tego komputera.")

st.title("🛡️ Lokalny Anonimizator PII")
st.caption("Hybryda: Regex (dane o stałym formacie) + mały LLM przez Ollamę "
           "(dane kontekstowe) + odwracalny sejf mapowań.")

tab_mask, tab_restore = st.tabs(["1️⃣ Anonimizacja", "2️⃣ De-anonimizacja"])

# --- Zakładka 1: anonimizacja -------------------------------------------------
with tab_mask:
    st.write("**Trudne przypadki testowe:**")
    columns = st.columns(len(TEST_CASES))
    for column, name in zip(columns, TEST_CASES):
        column.button(name, on_click=load_test_case, args=(name,))

    st.file_uploader(
        f"…albo wczytaj plik ({', '.join(SUPPORTED_EXTENSIONS)})",
        type=SUPPORTED_EXTENSIONS,
        key="input_file",
        on_change=load_uploaded_file,
        args=("input_file", "input_text"),
    )
    show_upload_error("input_file")

    raw = st.session_state.get("raw_pages")
    if raw:
        pages = raw["pages"]
        with st.expander("🔍 Podgląd surowego tekstu wyciągniętego z pliku"):
            st.caption(f"{raw['name']} · stron: {len(pages)} · znaków: "
                       f"{sum(len(page) for page in pages)}. Dokładnie to trafia do "
                       "Regexa i (strona po stronie) do modelu.")
            for number, page in enumerate(pages, start=1):
                st.markdown(f"**Strona {number}** · {len(page)} znaków")
                st.code(page or "(pusta strona – skan bez tekstu?)",
                        language=None, wrap_lines=True)

    st.text_area("Tekst wejściowy (z danymi wrażliwymi)", key="input_text", height=230)

    if st.button("🔒 Anonimizuj", type="primary"):
        text = st.session_state.get("input_text", "")
        if not text.strip():
            st.info("Wklej tekst albo wybierz przypadek testowy.")
        else:
            # Pasek postępu: przy dłuższym pliku widać, że model pracuje
            # (fragment 3 z 7), zamiast samego kręcącego się kółka.
            progress = st.progress(0.0, text="Regex + lokalny model analizują tekst...")

            def show_progress(done: int, total: int) -> None:
                progress.progress(done / total, text=f"LLM: fragment {done} z {total}")

            st.session_state.last_result = anonymize(
                text, vault, use_llm=use_llm, model=model, base_url=base_url,
                on_progress=show_progress,
            )
            progress.empty()
            build_redacted_pdf(text, st.session_state.last_result.entities)

    result = st.session_state.get("last_result")
    if result:
        if result.llm_error:
            st.warning(f"Etap LLM pominięty: {result.llm_error}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Regex", f"{result.regex_ms:.1f} ms")
        m2.metric("LLM", f"{result.llm_ms / 1000:.2f} s" if use_llm else "wył.")
        m3.metric("Wykryte encje", len(result.entities))
        ram = get_process_peak_ram_mb()
        m4.metric("RAM aplikacji", f"{ram:.0f} MB" if ram else "—")
        model_ram = get_model_memory_mb(model, base_url)
        m5.metric("RAM modelu", f"{model_ram:.0f} MB" if model_ram else "—")

        st.subheader("Tekst bezpieczny do wysłania do chmury")
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

        redaction = st.session_state.get("redaction")
        if redaction:
            from_source = redaction["source_pdf"] is not None
            st.subheader("⬛ Dokument z czarnymi paskami")
            # Obrazy istnieją tylko w wgranym PDF – w PDF-ie z tekstu ich nie ma.
            redact_images = st.checkbox(
                "Zamazać również podpisy odręczne, pieczątki i obrazy",
                value=True,
                disabled=not from_source,
                help="Wycina z PDF wszystkie osadzone obrazy (podpis, pieczęć, kod QR, "
                     "logo) i zostawia w ich miejscu czarny prostokąt. Dotyczy wgranego PDF.",
            )
            redacted = make_redacted_pdf(
                redaction["source_pdf"], redaction["text"], redaction["values"], redact_images
            )
            if from_source:
                source_info = ("oryginalny PDF, strony spłaszczone do obrazu – "
                               "w pliku nie ma tekstu, który dałoby się skopiować")
                if redact_images:
                    source_info += f" · usuniętych obrazów: {redacted.images}"
            else:
                source_info = ("PDF z tekstu – w miejscu danych są tylko paski, "
                               "prawdziwe wartości nie trafiły do pliku")
            st.caption(f"Pasków: {redacted.bars} · {source_info}.")
            if from_source and redact_images and redacted.background_pages:
                st.info(f"Stron będących skanem: {redacted.background_pages}. Obraz całej "
                        "strony nie jest zamazywany (strona byłaby czarna). Paski na tekście "
                        "działają dzięki warstwie OCR, ale podpis odręczny WEWNĄTRZ skanu "
                        "nie jest osobnym obrazem i nie zostanie wykryty.")
            st.download_button(
                "⬛ Pobierz zaczerniony PDF",
                data=redacted.pdf,
                file_name=redaction["file_name"],
                mime="application/pdf",
            )
            if redacted.missing:
                st.warning("Nie znaleziono w PDF (sprawdź ręcznie): "
                           + ", ".join(redacted.missing))
            with st.expander("Podgląd zaczernionego dokumentu"):
                pages = page_count(redacted.pdf)
                page = 1
                if pages > 1:
                    page = st.number_input("Strona", min_value=1, max_value=pages, value=1)
                st.image(render_preview(redacted.pdf, page - 1))

        st.subheader("Wykryte encje")
        if result.entities:
            st.dataframe(result.entities, hide_index=True)
        else:
            st.write("Nie wykryto danych wrażliwych.")

# --- Zakładka 2: de-anonimizacja ---------------------------------------------
with tab_restore:
    st.write("Wklej odpowiedź z ChatGPT / innego AI zawierającą tokeny "
             "(np. `[OSOBA_1]`). Aplikacja podstawi oryginalne dane z lokalnego sejfu.")
    st.button("🧪 Demo: wstaw zanonimizowany tekst jako odpowiedź AI",
              on_click=use_masked_as_demo_response,
              disabled="last_result" not in st.session_state)

    st.file_uploader(
        "…albo wczytaj odpowiedź AI z pliku",
        type=SUPPORTED_EXTENSIONS,
        key="response_file",
        on_change=load_uploaded_file,
        args=("response_file", "cloud_response"),
    )
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

    st.divider()
    show_vault_token(vault, key="tab2")

vault_counter.metric("Tokenów w sejfie", len(vault))
