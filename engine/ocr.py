"""OCR tylko tam, gdzie jest potrzebny: na obrazach osadzonych w PDF.

Cały PDF przez OCR to strata czasu i RAM-u – zwykła umowa ma warstwę
tekstową, a obrazy to co najwyżej pieczątka, podpis albo zeskanowany
załącznik. Dlatego:

1. Bierzemy TYLKO obrazy strony (page.get_image_info) – strona bez obrazów
   w ogóle nie trafia do OCR.
2. Pomijamy obrazy, które nic nie wniosą:
   - za małe (ikony, kreski, punktory),
   - już pokryte warstwą tekstową (skan z OCR-em od skanera).
3. Renderujemy wyłącznie prostokąt obrazu (clip) w 300 dpi i dajemy go
   Tesseractowi (przez PyMuPDF – bez nowych bibliotek Pythona).
4. Ten sam obraz (np. logo w stopce każdej strony) rozpoznajemy RAZ –
   wynik jest zapamiętywany po identyfikatorze obrazu w pliku (xref).
5. Wynik całego pliku jest zapamiętywany (po skrócie SHA-256), więc odczyt
   tekstu i późniejsze zaczernianie nie robią OCR dwa razy.

Słowa z OCR mają współrzędne na stronie, więc redaktor może wypalić na nich
czarne paski tak samo jak na zwykłym tekście.
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pymupdf

# Język Tesseracta. „pol” zna polskie znaki (ą, ę, ł…), „eng” dobrze czyta
# e-maile, kody i identyfikatory – razem dają najlepszy wynik.
OCR_LANGUAGE = "pol+eng"
# 300 dpi to standard dla OCR. Mniej – Tesseract gubi małe znaki (pieczątki).
OCR_DPI = 300
# Bezpiecznik RAM: większy wycinek renderujemy w niższej rozdzielczości.
MAX_OCR_PIXELS = 12_000_000
# Obrazy mniejsze niż to (w punktach PDF, 72 pt = 2,54 cm) pomijamy.
# 30 x 12 pt to mniej więcej jedno krótkie słowo – mniejsze to ikony i kreski.
MIN_IMAGE_WIDTH = 30
MIN_IMAGE_HEIGHT = 12
# Jeśli słowa warstwy tekstowej pokrywają już tyle powierzchni obrazu,
# to obraz ma OCR (np. skan ze skanera biurowego) – nie powtarzamy pracy.
TEXT_LAYER_COVERAGE = 0.15
# Obraz zajmujący co najmniej połowę strony to skan (tak samo jak w redaktorze).
# Wystarczy tyle słów w nim, żeby uznać, że skaner zrobił już OCR.
BACKGROUND_SHARE = 0.5
MIN_SCAN_LAYER_WORDS = 2
# Ile ostatnich plików pamiętamy (wyniki zawierają dane wrażliwe – mało).
CACHE_FILES = 4


@dataclass
class OcrWord:
    text: str
    rect: pymupdf.Rect          # współrzędne na STRONIE (punkty PDF)
    line: tuple[int, int, int]  # (nr obrazu, blok, linia) – do łączenia w paski


@dataclass
class OcrReport:
    """Wynik OCR całego pliku + statystyki do pokazania w aplikacji."""
    words: list[list[OcrWord]] = field(default_factory=list)  # słowa per strona
    texts: list[str] = field(default_factory=list)            # tekst per strona
    images_total: int = 0      # wszystkie obrazy w pliku
    images_ocr: int = 0        # przepuszczone przez Tesseracta
    images_reused: int = 0     # ten sam obraz drugi raz – wynik z pamięci
    skipped_small: int = 0     # za małe
    skipped_text: int = 0      # mają już warstwę tekstową
    pending: int = 0           # wymagały OCR, ale Tesseract niedostępny
    errors: list[str] = field(default_factory=list)
    ocr_ms: float = 0.0


class OcrUnavailableError(RuntimeError):
    """Tesseract albo polski model językowy nie są zainstalowane."""


# Lokalny katalog z modelami (poza Gitem). PyMuPDF ma wbudowany silnik
# Tesseracta i potrzebuje tylko plików *.traineddata – można je tu wrzucić
# bez uprawnień administratora (instalator Tesseracta daje sam „eng”).
LOCAL_TESSDATA = Path(__file__).resolve().parent.parent / "tessdata"


def _has_languages(folder: Path) -> bool:
    return all((folder / f"{language}.traineddata").exists()
               for language in OCR_LANGUAGE.split("+"))


@lru_cache(maxsize=1)
def tessdata_path() -> str | None:
    """Katalog z modelami Tesseracta albo None, gdy OCR jest niedostępny.

    Kolejność: TESSDATA_PREFIX / instalacja Tesseracta (PyMuPDF szuka sam),
    potem lokalny katalog tessdata/ w projekcie. Wymagamy modeli wszystkich
    języków z OCR_LANGUAGE.
    """
    candidates = []
    try:
        system = pymupdf.get_tessdata()
        if system:
            candidates.append(Path(system))
    except (RuntimeError, OSError):
        pass
    candidates.append(LOCAL_TESSDATA)
    for folder in candidates:
        if _has_languages(folder):
            return str(folder)
    return None


def ocr_available() -> bool:
    return tessdata_path() is not None


# Słowo z OCR we współrzędnych WZGLĘDNYCH obrazu (0..1). Dzięki temu wynik
# dla danego obrazu można użyć ponownie, gdy stoi on w innym miejscu strony.
RelativeWord = tuple[float, float, float, float, str, int, int]


def run_tesseract(pixmap: pymupdf.Pixmap) -> list[RelativeWord]:
    """Rozpoznaje tekst na obrazku. Zwraca słowa ze współrzędnymi 0..1.

    PyMuPDF wywołuje Tesseracta i oddaje jednostronicowy PDF z warstwą
    tekstową – z niej bierzemy słowa z położeniem. (Funkcja jest osobno,
    żeby testy mogły ją podmienić bez instalowania Tesseracta.)
    """
    tessdata = tessdata_path()
    if tessdata is None:
        raise OcrUnavailableError("Tesseract (z modelem „pol”) nie jest zainstalowany.")
    pdf = pixmap.pdfocr_tobytes(language=OCR_LANGUAGE, tessdata=tessdata)
    page = pymupdf.open(stream=pdf, filetype="pdf")[0]
    width, height = page.rect.width, page.rect.height
    return [
        (x0 / width, y0 / height, x1 / width, y1 / height, text, block, line)
        for x0, y0, x1, y1, text, block, line, _ in page.get_text("words", sort=True)
    ]


def _has_text_layer(page: pymupdf.Page, rect: pymupdf.Rect) -> bool:
    """Czy obraz ma już tekst (np. warstwę OCR dodaną przez skaner)?

    - Obraz tła (skan całej strony): wystarczy kilka słów – skaner z OCR
      zawsze daje warstwę na całym skanie, nawet gdy strona jest prawie pusta.
    - Mały obraz (pieczątka, zdjęcie): słowa muszą pokrywać jego istotną
      część – samo „Podpis:” obok pieczątki nie oznacza, że ją odczytano.
    """
    words = page.get_text("words", clip=rect)
    if rect.get_area() >= BACKGROUND_SHARE * page.rect.get_area():
        return len(words) >= MIN_SCAN_LAYER_WORDS
    covered = sum((pymupdf.Rect(w[:4]) & rect).get_area() for w in words)
    return covered / max(rect.get_area(), 1.0) >= TEXT_LAYER_COVERAGE


def _render(page: pymupdf.Page, rect: pymupdf.Rect) -> pymupdf.Pixmap:
    """Renderuje TYLKO prostokąt obrazu.

    Musi być RGB: pdfocr_tobytes() na obrazie w skali szarości po cichu
    zwraca PDF bez tekstu (sprawdzone na PyMuPDF z MuPDF 1.28).
    """
    dpi = OCR_DPI
    pixels = rect.width * rect.height * (dpi / 72) ** 2
    if pixels > MAX_OCR_PIXELS:
        dpi = int(dpi * (MAX_OCR_PIXELS / pixels) ** 0.5)
    return page.get_pixmap(dpi=dpi, clip=rect, colorspace=pymupdf.csRGB, alpha=False)


def _place(words: list[RelativeWord], rect: pymupdf.Rect, image_no: int) -> list[OcrWord]:
    """Przelicza słowa ze współrzędnych obrazu na współrzędne strony."""
    placed = []
    for x0, y0, x1, y1, text, block, line in words:
        placed.append(OcrWord(
            text,
            pymupdf.Rect(rect.x0 + x0 * rect.width, rect.y0 + y0 * rect.height,
                         rect.x0 + x1 * rect.width, rect.y0 + y1 * rect.height),
            (image_no, block, line),
        ))
    return placed


def _words_to_text(words: list[OcrWord]) -> str:
    """Składa słowa w linie (jak tekst wyciągnięty ze zwykłego PDF-a)."""
    lines: dict[tuple[int, int, int], list[str]] = {}
    for word in words:
        lines.setdefault(word.line, []).append(word.text)
    return "\n".join(" ".join(parts) for parts in lines.values())


def ocr_document(document: pymupdf.Document) -> OcrReport:
    """OCR obrazów w otwartym dokumencie (bez pamięci podręcznej plików)."""
    report = OcrReport()
    start = time.perf_counter()
    by_xref: dict[int, list[RelativeWord]] = {}
    available = ocr_available()
    image_no = 0

    for page in document:
        page_words: list[OcrWord] = []
        for info in page.get_image_info(xrefs=True):
            rect = pymupdf.Rect(info["bbox"]) & page.rect
            if rect.is_empty:
                continue
            report.images_total += 1
            if rect.width < MIN_IMAGE_WIDTH or rect.height < MIN_IMAGE_HEIGHT:
                report.skipped_small += 1
                continue
            if _has_text_layer(page, rect):
                report.skipped_text += 1
                continue

            xref = info.get("xref", 0)
            if xref and xref in by_xref:
                relative = by_xref[xref]
                report.images_reused += 1
            elif not available:
                report.pending += 1
                continue
            else:
                try:
                    relative = run_tesseract(_render(page, rect))
                except Exception as error:  # jeden zły obraz nie psuje całego pliku
                    report.errors.append(f"strona {page.number + 1}: {error}")
                    continue
                report.images_ocr += 1
                if xref:
                    by_xref[xref] = relative

            image_no += 1
            page_words.extend(_place(relative, rect, image_no))

        report.words.append(page_words)
        report.texts.append(_words_to_text(page_words))

    report.ocr_ms = (time.perf_counter() - start) * 1000
    return report


_CACHE: "OrderedDict[str, OcrReport]" = OrderedDict()


def ocr_pdf(pdf_bytes: bytes) -> OcrReport:
    """OCR obrazów w PDF; wynik zapamiętany dla ostatnich CACHE_FILES plików."""
    key = hashlib.sha256(pdf_bytes).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    report = ocr_document(pymupdf.open(stream=pdf_bytes, filetype="pdf"))
    if report.pending:
        # Brak Tesseracta – nie zapamiętujemy, żeby po jego instalacji
        # ten sam plik został rozpoznany.
        return report
    _CACHE[key] = report
    while len(_CACHE) > CACHE_FILES:
        _CACHE.popitem(last=False)
    return report


def clear_cache() -> None:
    """Czyści zapamiętane wyniki (zawierają dane wrażliwe)."""
    _CACHE.clear()
