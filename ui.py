"""System komponentów interfejsu (Streamlit).

Kolory, promienie, fonty i kolory komunikatów są w motywie
(.streamlit/config.toml, jasny i ciemny). Tu jest tylko to, czego motyw nie obejmuje:
karty, nagłówki sekcji z ikoną, wiersze stanu, przyciski destrukcyjne
i drobne poprawki rytmu odstępów.

Ikony: wyłącznie Material Symbols Rounded (wbudowane w Streamlit, działają
offline), w zapisie „:material/nazwa:”. Bez emoji.
"""

import threading
import time

import streamlit as st

# Tokeny projektu – te same wartości co w motywie (config.toml), osobno dla
# motywu jasnego i ciemnego. Reguły CSS niżej używają ich jako zmiennych --ui-*.
THEMES = {
    "light": {
        "primary": "#1F2F4D",
        "secondary": "#4B5D78",
        "accent": "#2456D6",
        "background": "#F5F6F8",
        "surface": "#FFFFFF",
        "text": "#111827",
        "text-secondary": "#4B5563",
        "border": "#E2E5EA",
        "success": "#15803D",
        "warning": "#B45309",
        "error": "#B91C1C",
        "error-bg": "#FDF2F2",
        "error-border": "#F1C5C5",
        "danger-fill": "#B91C1C",        # biały tekst 6.5:1
        "danger-fill-hover": "#991B1B",
        "info": "#1D4ED8",
    },
    "dark": {
        "primary": "#C7D2E3",
        "secondary": "#A3B3CC",
        "accent": "#3B6FE0",
        "background": "#0F141B",
        "surface": "#161C25",
        "text": "#E6EAF0",
        "text-secondary": "#A3AEBD",
        "border": "#2B3442",
        "success": "#4ADE80",
        "warning": "#FBBF24",
        "error": "#F87171",
        "error-bg": "#2A1517",
        "error-border": "#5C2A2E",
        "danger-fill": "#DC2626",        # biały tekst 4.8:1
        "danger-fill-hover": "#B91C1C",
        "info": "#60A5FA",
    },
}
COLORS = THEMES["light"]


def _variables(palette: dict[str, str]) -> str:
    return " ".join(f"--ui-{name}: {value};" for name, value in palette.items())


# Skala odstępów: 4 / 8 / 12 / 16 / 24 / 32 px.
_RULES = """
/* --- Układ ---------------------------------------------------------------- */
.block-container { max-width: 1180px; padding-top: 2.25rem; padding-bottom: 4rem; }
[data-testid="stDecoration"] { display: none; }
/* Kotwice-linki przy nagłówkach (ikona łańcucha po najechaniu) – zbędne w produkcie. */
[data-testid="stHeaderActionElements"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* --- Karty: kontenery z kluczem „card_…” / „kpi_…” --------------------------- */
[class*="st-key-card_"], [class*="st-key-kpi_"] {
    background: var(--ui-surface);
    border: 1px solid var(--ui-border);
    border-radius: 12px;
    padding: 20px 22px;
}
[class*="st-key-kpi_"] { padding: 14px 16px; gap: 0.25rem; }
@media (max-width: 640px) {
    [class*="st-key-card_"] { padding: 16px 14px; }
    .block-container { padding-top: 1.5rem; }
}
[class*="st-key-kpi_"] [data-testid="stMetricLabel"] p {
    color: var(--ui-text-secondary); font-size: 13px; font-weight: 500;
}
[class*="st-key-card_empty"] { text-align: center; padding: 36px 22px; border-style: dashed; }
[class*="st-key-card_empty"] [data-testid="stMarkdownContainer"] p { color: var(--ui-text-secondary); }

/* Nagłówek sekcji w karcie: opis tuż pod tytułem. */
[class*="st-key-card_"] h4 { padding-bottom: 0; }
[class*="st-key-card_"] [data-testid="stCaptionContainer"] { margin-top: -0.35rem; }

/* --- Przyciski destrukcyjne: klucz „danger_…” ------------------------------- */
[class*="st-key-danger_"] button {
    color: var(--ui-error);
    border-color: var(--ui-error-border);
    background: var(--ui-surface);
}
[class*="st-key-danger_"] button:hover {
    color: var(--ui-error);
    border-color: var(--ui-error);
    background: var(--ui-error-bg);
}
[class*="st-key-danger_confirm"] button {
    color: #FFFFFF; background: var(--ui-danger-fill); border-color: var(--ui-danger-fill);
}
[class*="st-key-danger_confirm"] button:hover {
    color: #FFFFFF; background: var(--ui-danger-fill-hover); border-color: var(--ui-danger-fill-hover);
}

/* --- Pola ------------------------------------------------------------------ */
/* Pole pliku i przyciski obok niego tej samej wysokości (pomiar: pole 68 px). */
[data-testid="stFileUploaderDropzone"] { min-height: 4.25rem; }
.st-key-examples_box [data-testid="stPopoverButton"],
.st-key-examples_box_restore button { min-height: 4.25rem; }
textarea { line-height: 1.5 !important; }

/* Lista w menu (np. przykłady): pozycje wyrównane do lewej, jak w menu systemowym. */
[data-testid="stPopoverBody"] button[kind="tertiary"] { justify-content: flex-start; width: 100%; }
[data-testid="stPopoverBody"] button[kind="tertiary"] > div { justify-content: flex-start; }

/* --- Nawigacja (zakładki) -------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] { gap: 1.25rem; }
.stTabs [data-baseweb="tab"] p { font-weight: 500; }

/* --- Pasek boczny: lista stanu ---------------------------------------------- */
.st-key-status_list [data-testid="stMarkdownContainer"] p { margin-bottom: 0.35rem; font-size: 14px; }

/* --- Postęp: podpowiedź pod paskiem, gdy analiza trwa długo ---------------- */
.st-key-progress_hint [data-testid="stCaptionContainer"] { color: var(--ui-text-secondary); }
"""


# Streamlit nie oznacza aktywnego motywu w DOM, a st.context.theme przy
# pierwszym wczytaniu potrafi podać zły typ (sprawdzone: „light” przy
# ciemnym). Dlatego motyw rozpoznaje przeglądarka: jasność tła aplikacji ->
# atrybut data-ui-theme na <html>. Skrypt jest stały (bez danych z aplikacji)
# i sprawdza tło co 400 ms, więc reaguje też na zmianę motywu w menu.
_THEME_SCRIPT = r"""
<script>
(() => {
  if (window.__uiThemeWatch) return;
  window.__uiThemeWatch = true;
  const detect = () => {
    const app = document.querySelector('.stApp');
    if (!app) return;
    const rgb = getComputedStyle(app).backgroundColor.match(/\d+/g);
    if (!rgb) return;
    const [r, g, b] = rgb.map(Number);
    const theme = (0.299 * r + 0.587 * g + 0.114 * b) < 128 ? 'dark' : 'light';
    if (document.documentElement.dataset.uiTheme !== theme) {
      document.documentElement.dataset.uiTheme = theme;
    }
  };
  detect();
  setInterval(detect, 400);
})();
</script>
"""


def apply_styles() -> None:
    """Style komponentów + zmienne kolorów dla motywu jasnego i ciemnego.

    Zanim skrypt rozpozna motyw, kolory wybiera ustawienie systemu
    (prefers-color-scheme) – tak samo jak domyślnie robi to Streamlit.
    """
    light, dark = _variables(THEMES["light"]), _variables(THEMES["dark"])
    variables = (
        f":root {{ {light} }}\n"
        f"@media (prefers-color-scheme: dark) {{ :root:not([data-ui-theme]) {{ {dark} }} }}\n"
        f':root[data-ui-theme="light"] {{ {light} }}\n'
        f':root[data-ui-theme="dark"] {{ {dark} }}'
    )
    st.html(f"<style>{variables}\n{_RULES}</style>")
    st.html(_THEME_SCRIPT, unsafe_allow_javascript=True)


def page_header(title: str, description: str) -> None:
    st.markdown(f"# {title}")
    st.caption(description)


def section(icon: str, title: str, description: str | None = None) -> None:
    """Nagłówek sekcji w karcie: ikona + tytuł (+ jedno zdanie opisu)."""
    st.markdown(f"#### :material/{icon}: {title}")
    if description:
        st.caption(description)


# Stan komponentu -> ikona i kolor (kolory z motywu: green/orange/red/gray).
_STATE = {
    "ok": ("green", "check_circle"),
    "warn": ("orange", "error"),
    "error": ("red", "cancel"),
    "off": ("gray", "do_not_disturb_on"),
}


def status_line(state: str, label: str, detail: str) -> str:
    """Jedna linia listy stanu (Markdown): kolorowa ikona, nazwa, szczegół."""
    color, icon = _STATE[state]
    return f":{color}[:material/{icon}:] **{label}** · {detail}"


def page_break_view(text: str, page_break: str) -> str:
    """Tekst do wyświetlenia: niewidoczny znak nowej strony jako czytelny separator."""
    return text.replace(page_break, "\n\n— nowa strona —\n\n")


def empty_state(key: str, icon: str, title: str, text: str) -> None:
    """Pusty stan: zamiast pustego miejsca – co tu się pojawi i skąd."""
    with st.container(key=f"card_{key}"):
        st.markdown(f"### :gray[:material/{icon}:]")
        st.markdown(f"**{title}**")
        st.markdown(text)


# Po ilu sekundach pod paskiem pojawia się podpowiedź, że analiza nadal trwa.
SLOW_HINT_S = 20
VERY_SLOW_HINT_S = 60


def run_with_progress(work, progress, hint, use_llm: bool,
                      slow_s: float = SLOW_HINT_S, very_slow_s: float = VERY_SLOW_HINT_S):
    """Uruchamia anonimizację w wątku, a pasek odświeża co 0,25 s.

    Wcześniej pasek ruszał się dopiero po skończeniu fragmentu przez model –
    przy pierwszym starcie (wczytanie modeli) albo w trybie oszczędzania
    energii stał nieruchomo przez kilkadziesiąt sekund i wyglądał na
    zawieszony. Teraz zawsze widać upływający czas, a po dłuższej chwili
    – wyjaśnienie, dlaczego to trwa.

    work: funkcja przyjmująca callback on_progress(done, total).
    progress / hint: elementy Streamlita (st.progress, st.empty) – w testach atrapy.
    """
    state = {"done": 0, "total": 0, "result": None, "error": None}

    def on_progress(done: int, total: int) -> None:   # wołane z wątku roboczego
        state["done"], state["total"] = done, total

    def target() -> None:
        try:
            state["result"] = work(on_progress)
        except BaseException as error:  # przekazujemy dalej w głównym wątku
            state["error"] = error

    started = time.perf_counter()
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    shown_hint = 0
    while worker.is_alive():
        elapsed = time.perf_counter() - started
        if state["total"]:
            # Regex i NER są szybkie – pasek: 10% za nie, reszta za fragmenty modelu.
            fraction = 0.1 + 0.9 * state["done"] / state["total"]
            label = f"Model językowy: fragment {min(state['done'] + 1, state['total'])} z {state['total']}"
        else:
            fraction = 0.05
            label = "Regex i NER" + (" – za chwilę model językowy" if use_llm else "")
        progress.progress(min(fraction, 0.99), text=f"{label} · {elapsed:.0f} s")
        if elapsed >= very_slow_s and shown_hint < 2:
            hint.caption(":material/hourglass_bottom: To trwa dłużej niż zwykle. Model językowy działa "
                         "wolniej na zasilaniu bateryjnym, w trybie oszczędzania energii albo gdy "
                         "z Ollamy korzysta jednocześnie inny program. Analiza nadal trwa.")
            shown_hint = 2
        elif elapsed >= slow_s and shown_hint < 1:
            hint.caption(":material/hourglass_top: Wciąż analizuję… Przy pierwszym uruchomieniu "
                         "wczytują się modele, a długie dokumenty mają więcej fragmentów.")
            shown_hint = 1
        worker.join(timeout=0.25)
    if state["error"] is not None:
        raise state["error"]
    progress.progress(1.0, text=f"Gotowe · {time.perf_counter() - started:.0f} s")
    return state["result"]
