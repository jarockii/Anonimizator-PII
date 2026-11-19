"""Zaczernianie danych wrażliwych w PDF (czarne paski, jak w aktach sądowych).

Częsty błąd w „redakcji” to narysowanie czarnego prostokąta NA tekście:
dane nadal są w pliku i wystarczy je zaznaczyć i skopiować. Tu tego nie ma,
bo mamy dwie ścieżki, w których prawdziwe wartości nie trafiają do pliku:

1. Tekst (wklejony, .txt, .docx...) -> redact_text_to_pdf():
   budujemy PDF od zera, a w miejscu danych wstawiamy czarne tło
   ze spacjami. Wartości nigdy nie są zapisywane w pliku.

2. Wgrany PDF -> redact_pdf():
   rysujemy paski na stronie, a potem „spłaszczamy” ją do obrazka
   (jak skan). W wyniku nie ma żadnej warstwy tekstowej, więc spod paska
   nie da się niczego skopiować.

   Dlaczego nie wbudowane apply_redactions() z PyMuPDF? Usuwa tekst
   poprawnie, ale przepisując linię przesuwa resztę tekstu o szerokość
   litery – kolejne słowo wchodzi pod pasek („netto” -> „etto”).
"""

import html
import io
import re
from dataclasses import dataclass, field

import pymupdf

from engine.file_reader import PAGE_BREAK, normalize_pdf_text

# O ile zmniejszamy pasek od góry. 0 = pełna wysokość linii: przy 10%
# nad paskiem wystawały kreski z „Ś”, „Ź”, „Ż” pisanych WIELKIMI literami
# (Karta Podpisów). Strona jest spłaszczana do obrazu, więc lekkie
# zachodzenie na linię powyżej niczego nie usuwa – to tylko wygląd.
TOP_SHRINK = 0.0
# Znaki interpunkcyjne odcinane z brzegów słów przy porównywaniu,
# żeby „Kowalski,” na stronie pasowało do „Kowalski” z sejfu.
WORD_EDGE_PUNCTUATION = ".,;:!?()[]{}\"'„”«»"
# Rozdzielczość spłaszczonych stron: 150 dpi to czytelny wydruk
# przy rozsądnym rozmiarze pliku.
FLATTEN_DPI = 150

PAGE_CSS = "p { margin: 0; font-family: sans-serif; font-size: 11pt; line-height: 1.4; }"
BAR_START = '<span style="background-color: black">'
BAR_END = "</span>"


@dataclass
class RedactionResult:
    pdf: bytes
    bars: int                                          # ile pasków narysowano
    missing: list[str] = field(default_factory=list)   # wartości nieznalezione
    images: int = 0                                    # ile obrazów usunięto
    background_pages: int = 0                          # strony-skany (tło pominięte)


def _clean_values(values: list[str]) -> list[str]:
    # Najdłuższe najpierw – tak samo jak przy podmianie tokenów.
    return sorted({v.strip() for v in values if v.strip()}, key=len, reverse=True)


def _html_lines_to_pdf(html_lines: list[str]) -> bytes:
    """Składa PDF A4 z gotowych linii HTML (każda linia = akapit)."""
    body = "".join(f"<p>{line or '&nbsp;'}</p>" for line in html_lines)
    story = pymupdf.Story(html=body, user_css=PAGE_CSS)

    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    page_rect = pymupdf.paper_rect("a4")
    content_rect = page_rect + (50, 50, -50, -50)  # marginesy 50 pt
    more_content = True
    while more_content:  # Story sam dzieli tekst na kolejne strony
        device = writer.begin_page(page_rect)
        more_content, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
    writer.close()
    return buffer.getvalue()


def text_to_pdf(text: str) -> bytes:
    """Zwykły PDF z tekstu, bez zaczerniania."""
    # html.escape chroni przed tym, żeby np. „<b>” w tekście stało się HTML-em.
    text = text.replace(PAGE_BREAK, "\n")
    return _html_lines_to_pdf([html.escape(line) for line in text.split("\n")])


def redact_text_to_pdf(text: str, values: list[str]) -> RedactionResult:
    """Tworzy PDF z tekstu, w którym dane wrażliwe są czarnymi paskami."""
    # Znak nowej strony (z PDF) zamieniamy na zwykły koniec linii.
    text = text.replace(PAGE_BREAK, "\n")
    unique_values = _clean_values(values)
    if not unique_values:
        return RedactionResult(text_to_pdf(text), 0)

    # Jeden regex ze wszystkimi wartościami: przechodzimy tekst raz,
    # a dłuższe wartości mają pierwszeństwo przed krótszymi.
    # IGNORECASE + dowolny odstęp: tak samo jak przy maskowaniu w sejfie,
    # więc „LIVERI SPÓŁKA” z Karty Podpisów też dostanie pasek.
    alternatives = (r"\s+".join(re.escape(word) for word in v.split()) for v in unique_values)
    pattern = re.compile("|".join(alternatives), re.IGNORECASE)
    found: set[str] = set()
    pieces: list[str] = []
    position = 0
    for match in pattern.finditer(text):
        pieces.append(html.escape(text[position:match.start()]))
        found.add(" ".join(match.group(0).split()).casefold())
        # Pasek = czarne tło na niełamliwych spacjach o długości wartości.
        # Wartość może obejmować kilka linii (np. „Marek\nLewandowski”),
        # wtedy każda linia dostaje własny pasek.
        bars = [BAR_START + "&nbsp;" * max(len(part), 1) + BAR_END
                for part in match.group(0).split("\n")]
        pieces.append("\n".join(bars))
        position = match.end()
    pieces.append(html.escape(text[position:]))

    html_lines = "".join(pieces).split("\n")
    bar_count = sum(line.count(BAR_START) for line in html_lines)
    missing = [v for v in unique_values if " ".join(v.split()).casefold() not in found]
    return RedactionResult(_html_lines_to_pdf(html_lines), bar_count, missing)


def _normalize_word(word: str) -> str:
    """Postać słowa do porównań: zwykłe myślniki, bez interpunkcji na brzegach,
    bez wielkości liter (casefold działa też dla „Ś” i „ś”)."""
    return normalize_pdf_text(word).strip(WORD_EDGE_PUNCTUATION).casefold()


@dataclass
class PageWord:
    key: str                # znormalizowane słowo
    rect: pymupdf.Rect      # położenie na stronie
    line: tuple[int, int]   # (blok, linia) – do łączenia słów w jeden pasek


def page_words(page: pymupdf.Page) -> list[PageWord]:
    """Lista słów strony w kolejności czytania (bez samej interpunkcji)."""
    words = []
    for x0, y0, x1, y1, text, block, line, _ in page.get_text("words", sort=True):
        key = _normalize_word(text)
        if key:
            words.append(PageWord(key, pymupdf.Rect(x0, y0, x1, y1), (block, line)))
    return words


def _word_matches(page_key: str, target_key: str, first: bool, last: bool) -> bool:
    """Czy słowo strony pasuje do słowa wartości.

    Pierwsze/ostatnie słowo wartości może być „przyklejone” do znaków,
    które nie są literą ani cyfrą: wartość „0.80 zł” ma pasować do
    „0.80 zł/km” na stronie, a „Kraków” do „(Kraków)”.
    """
    if page_key == target_key:
        return True
    n = len(target_key)
    if last and page_key.startswith(target_key) and not page_key[n].isalnum():
        return True
    if first and page_key.endswith(target_key) and not page_key[-n - 1].isalnum():
        return True
    return False


def find_value_rects(words: list[PageWord], value: str) -> list[pymupdf.Rect]:
    """Prostokąty, w których wartość występuje na stronie – CAŁYMI słowami.

    Wszystkie słowa wartości muszą stać obok siebie w tej samej kolejności
    (mogą być w dwóch liniach, np. adres złamany na końcu wiersza).
    Wcześniej używaliśmy page.search_for(), które szuka PODCIĄGÓW:
    „Jan” trafiało w środek „Janem”, a tryb awaryjny „słowo po słowie”
    zaczerniał każde „ul.” i „2009” na stronie – 844 paski na 6 stron.
    """
    target = [key for key in (_normalize_word(w) for w in value.split()) if key]
    if not target:
        return []
    rects: list[pymupdf.Rect] = []
    size = len(target)
    i = 0
    while i <= len(words) - size:
        window = words[i:i + size]
        if all(_word_matches(word.key, key, first=(k == 0), last=(k == size - 1))
               for k, (word, key) in enumerate(zip(window, target))):
            # Jeden pasek na każdą linię, którą zajmuje wartość.
            by_line: dict[tuple[int, int], pymupdf.Rect] = {}
            for word in window:
                if word.line in by_line:
                    by_line[word.line] |= word.rect   # suma prostokątów
                else:
                    by_line[word.line] = pymupdf.Rect(word.rect)
            rects.extend(by_line.values())
            i += size
        else:
            i += 1
    return rects


# Obraz zajmujący co najmniej tyle strony to TŁO, nie podpis ani pieczątka.
# Typowy przypadek: zeskanowane pismo – cała strona to jeden obraz
# (np. 2480×3508 px = A4 w 300 dpi), a tekst jest niewidoczną warstwą OCR.
# Zamazanie takiego obrazu dawało całkowicie czarną stronę.
BACKGROUND_IMAGE_SHARE = 0.5


def split_page_images(page: pymupdf.Page) -> tuple[list[pymupdf.Rect], list[pymupdf.Rect]]:
    """Dzieli obrazy strony na (małe obrazy do zamazania, obrazy tła)."""
    small: list[pymupdf.Rect] = []
    backgrounds: list[pymupdf.Rect] = []
    page_area = page.rect.get_area()
    for info in page.get_image_info(xrefs=True):
        # Przycinamy do strony – części obrazu poza nią i tak nie widać.
        rect = pymupdf.Rect(info["bbox"]) & page.rect
        if rect.is_empty:
            continue
        if rect.get_area() >= BACKGROUND_IMAGE_SHARE * page_area:
            backgrounds.append(rect)
        else:
            small.append(rect)
    return small, backgrounds


def remove_images(page: pymupdf.Page) -> int:
    """Wycina ze strony podpisy odręczne, pieczątki, kody QR (małe obrazy).

    Zwraca liczbę usuniętych obrazów. Na miejscu każdego zostaje czarny
    prostokąt, a sam obraz jest FIZYCZNIE usuwany z pliku – nie da się
    go wyciągnąć narzędziem do eksportu obrazków. Obrazy tła (skan całej
    strony) zostają – inaczej strona byłaby w całości czarna.
    """
    rects, backgrounds = split_page_images(page)
    for rect in rects:
        page.add_redact_annot(rect, fill=(0, 0, 0))
    if rects:
        page.apply_redactions(
            # REMOVE usuwa w całości każdy obraz, którego DOTKNIE adnotacja –
            # na skanie usunąłby więc cały skan. Gdy strona ma tło, czyścimy
            # tylko piksele pod prostokątem (PIXELS).
            images=(pymupdf.PDF_REDACT_IMAGE_PIXELS if backgrounds
                    else pymupdf.PDF_REDACT_IMAGE_REMOVE),
            # Linii i tekstu NIE ruszamy. Usuwanie tekstu przez
            # apply_redactions przesuwało resztę linii („netto” -> „etto”) –
            # tekst zaczerniamy osobno, a stronę na końcu spłaszczamy.
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_NONE,
        )
    return len(rects)


def burn_bars(pixmap: pymupdf.Pixmap, page: pymupdf.Page,
              rects: list[pymupdf.Rect], dpi: int) -> None:
    """Wypala czarne paski bezpośrednio w pikselach wyrenderowanej strony.

    Dlaczego nie page.draw_rect() przed renderowaniem? Pola formularza
    (np. pola podpisu w Karcie Podpisów Autenti) są rysowane NA WIERZCHU
    treści strony – ich białe tło zasłaniało nasze paski i imiona, e-maile,
    NIP i IP podpisujących zostawały widoczne. Paski malowane po
    wyrenderowaniu wszystkiego są zawsze na górze.

    Dlaczego nie prostokąt na gotowej stronie? Bo pod takim prostokątem
    nadal byłyby piksele z danymi – wystarczyłoby go usunąć w edytorze PDF.
    """
    # Współrzędne tekstu -> piksele: obrót strony (jeśli jest) i skala dpi.
    matrix = page.rotation_matrix * pymupdf.Matrix(dpi / 72, dpi / 72)
    for rect in rects:
        bar = pymupdf.Rect(rect.x0, rect.y0 + rect.height * TOP_SHRINK, rect.x1, rect.y1)
        pixel_rect = (bar * matrix).irect & pixmap.irect
        if not pixel_rect.is_empty:
            # Czarny = 0 w kanałach koloru; kanał alfa (jeśli jest) = 255,
            # inaczej „czarny” piksel byłby przezroczysty.
            black = (0,) * (pixmap.n - 1) + (255,) if pixmap.alpha else (0,) * pixmap.n
            pixmap.set_rect(pixel_rect, black)


def redact_pdf(
    pdf_bytes: bytes,
    values: list[str],
    dpi: int = FLATTEN_DPI,
    redact_images: bool = True,
) -> RedactionResult:
    """Zaczernia wartości w istniejącym PDF i spłaszcza strony do obrazków.

    redact_images=True dodatkowo wycina wszystkie obrazy osadzone w PDF
    (podpisy odręczne, pieczątki, kody QR, zdjęcia dowodu).
    """
    source = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    output = pymupdf.open()
    unique_values = _clean_values(values)
    found: set[str] = set()
    bars = 0
    images = 0
    background_pages = 0

    for page in source:
        if redact_images:
            if split_page_images(page)[1]:
                background_pages += 1
            images += remove_images(page)

        words = page_words(page)
        page_bars: list[pymupdf.Rect] = []
        for value in unique_values:
            rects = find_value_rects(words, value)
            page_bars.extend(rects)
            if rects:
                found.add(value)
                bars += len(rects)

        # Spłaszczenie: nowa strona zawiera TYLKO obrazek starej strony.
        # Tekst, metadane i ukryte warstwy oryginału nie przechodzą dalej.
        pixmap = page.get_pixmap(dpi=dpi)
        burn_bars(pixmap, page, page_bars, dpi)
        new_page = output.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, pixmap=pixmap)

    missing = [v for v in unique_values if v not in found]
    return RedactionResult(output.tobytes(garbage=3, deflate=True), bars, missing,
                           images, background_pages)


def render_preview(pdf_bytes: bytes, page_number: int = 0, dpi: int = 90) -> bytes:
    """Zwraca obrazek PNG wybranej strony – do podglądu w aplikacji."""
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    return document[page_number].get_pixmap(dpi=dpi).tobytes("png")


def page_count(pdf_bytes: bytes) -> int:
    return pymupdf.open(stream=pdf_bytes, filetype="pdf").page_count
