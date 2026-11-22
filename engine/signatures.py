"""Wykrywanie podpisów odręcznych, których nie usuwa wycinanie obrazów.

Redaktor (engine/redactor.py) już wycina z PDF osadzone obrazy – podpis
wklejony jako obrazek znika razem z nimi. Zostawały dwa przypadki:

A. Podpis WEKTOROWY – narysowany liniami i krzywymi (tablet, część platform
   e-podpisu, „podpis” z edytora PDF). To nie jest obraz, więc nic go nie
   wycinało. Rozpoznajemy go po kształcie ścieżek (find_vector_signatures):
   dużo krótkich odcinków i krzywych bez wypełnienia, w różnych kierunkach,
   a pióro wraca w lewo (pętle liter) – czego nie robią ramki tabel,
   podkreślenia ani wykresy liniowe. Koszt: kilka ms na stronę, bez modelu.

B. Podpis WEWNĄTRZ skanu – cała strona to jeden obraz, którego nie wolno
   wyciąć (strona byłaby czarna). Bez modelu i bez pobierania: przy słowie
   „podpis” (z warstwy tekstowej skanu albo z OCR) szukamy ciemnych pikseli,
   które nie są rozpoznanym tekstem ani linią do podpisu
   (find_scan_signatures). Działa tylko na stronach-skanach, na obrazie
   strony, który redaktor i tak renderuje.
"""

import re
from dataclasses import dataclass

import pymupdf

# --- A: podpisy wektorowe -----------------------------------------------------

# Ścieżki bliżej niż tyle punktów (ok. 4 mm) należą do tego samego podpisu –
# litery podpisu to osobne ścieżki z przerwami między słowami.
CLUSTER_GAP = 12
# Rozmiar podpisu: mniejsze to parafa/kropka, większe to wykres albo ramka.
MIN_SIGNATURE_SIZE = (20, 8)
MAX_SIGNATURE_SHARE = (0.6, 0.25)    # część szerokości / wysokości strony
# Ile odcinków musi mieć podpis i jaka ich część ma być ukośna
# (ramka tabeli i podkreślenie mają same poziome/pionowe linie).
MIN_SEGMENTS = 12
MIN_DIAGONAL_SHARE = 0.5
# Ile razy pióro „wraca w lewo”. Wykres liniowy idzie stale w prawo,
# a pismo odręczne robi pętle: „a”, „o”, „l”, powrót do kreski „t”.
MIN_LEFT_MOVES = 3
# Odcinek uznajemy za ukośny, gdy odchyla się od poziomu/pionu o > ~10°.
AXIS_TOLERANCE = 0.18
# Pióro (tablet, edytor PDF) zapisuje pociągnięcie jako JEDNĄ ścieżkę z wieloma
# odcinkami. Gęste, ale pojedyncze kreski (kreskowanie na rysunku technicznym)
# dawały bez tego warunku dziesiątki fałszywych „podpisów” na stronie.
MIN_LONGEST_STROKE = 6
# Pasek trochę większy niż podpis – grube pociągnięcia wystają poza ścieżkę.
SIGNATURE_MARGIN = 3
# Bezpiecznik czasu na stronach z tysiącami ścieżek (rysunki techniczne).
MAX_STROKES = 2000


@dataclass
class _Stroke:
    rect: pymupdf.Rect
    segments: list[tuple[pymupdf.Point, pymupdf.Point]]


def _strokes(page: pymupdf.Page) -> list[_Stroke]:
    """Ścieżki, które MOGĄ być fragmentem podpisu: obrysy bez wypełnienia
    (wypełnione kształty to logo, wykres kołowy, ikona – wycinek koła ma
    w PDF jawne linie do środka, więc nie zawsze jest „zamknięty”) i bez
    prostokątów (ramki, tła komórek). Podpis zapisany jako wypełniony
    kontur liter nie zostanie więc wykryty – to rzadkie, a fałszywe paski
    na każdym logo i wykresie byłyby częste."""
    strokes = []
    for drawing in page.get_drawings():
        if drawing.get("fill") is not None:
            continue
        items = drawing.get("items", [])
        if not items or any(item[0] in ("re", "qu") for item in items):
            continue
        segments = []
        for item in items:
            if item[0] == "l":
                segments.append((item[1], item[2]))
            elif item[0] == "c":   # krzywa Béziera: początek -> koniec
                segments.append((item[1], item[4]))
        if segments:
            strokes.append(_Stroke(pymupdf.Rect(drawing["rect"]), segments))
    return strokes


def _grow(rect: pymupdf.Rect) -> pymupdf.Rect:
    return rect + (-CLUSTER_GAP, -CLUSTER_GAP, CLUSTER_GAP, CLUSTER_GAP)


def _cluster(strokes: list[_Stroke]) -> list[list[_Stroke]]:
    """Łączy ścieżki leżące blisko siebie (podpis = wiele osobnych kresek).

    Dwie ścieżki są w jednej grupie, gdy dzieli je ≤ CLUSTER_GAP – także
    pośrednio, przez łańcuch sąsiadów (union-find). Sąsiadów szukamy tylko
    w tych samych oczkach siatki, więc koszt rośnie prawie liniowo: pętla
    „każda grupa z każdą” na stronie z 1900 kreskami trwała 1,6 s.
    """
    parent = list(range(len(strokes)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cell = CLUSTER_GAP * 4
    grid: dict[tuple[int, int], list[int]] = {}
    for index, stroke in enumerate(strokes):
        grown = _grow(stroke.rect)
        for gx in range(int(grown.x0 // cell), int(grown.x1 // cell) + 1):
            for gy in range(int(grown.y0 // cell), int(grown.y1 // cell) + 1):
                for other in grid.setdefault((gx, gy), []):
                    if grown.intersects(strokes[other].rect):
                        parent[root(other)] = root(index)
                grid[(gx, gy)].append(index)

    groups: dict[int, list[_Stroke]] = {}
    for index, stroke in enumerate(strokes):
        groups.setdefault(root(index), []).append(stroke)
    return list(groups.values())


def _looks_handwritten(members: list[_Stroke], page_rect: pymupdf.Rect) -> pymupdf.Rect | None:
    rect = pymupdf.Rect(members[0].rect)
    for stroke in members[1:]:
        rect |= stroke.rect
    if rect.width < MIN_SIGNATURE_SIZE[0] or rect.height < MIN_SIGNATURE_SIZE[1]:
        return None
    if (rect.width > MAX_SIGNATURE_SHARE[0] * page_rect.width
            or rect.height > MAX_SIGNATURE_SHARE[1] * page_rect.height):
        return None
    segments = [segment for stroke in members for segment in stroke.segments]
    if len(segments) < MIN_SEGMENTS:
        return None
    if max(len(stroke.segments) for stroke in members) < MIN_LONGEST_STROKE:
        return None
    diagonal = left_moves = 0
    for start, end in segments:
        dx, dy = end.x - start.x, end.y - start.y
        length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        if min(abs(dx), abs(dy)) / length > AXIS_TOLERANCE:
            diagonal += 1
        if dx < -0.5:
            left_moves += 1
    if diagonal / len(segments) < MIN_DIAGONAL_SHARE or left_moves < MIN_LEFT_MOVES:
        return None
    return rect + (-SIGNATURE_MARGIN, -SIGNATURE_MARGIN, SIGNATURE_MARGIN, SIGNATURE_MARGIN)


def find_vector_signatures(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Prostokąty podpisów narysowanych wektorowo (współrzędne strony)."""
    strokes = _strokes(page)
    if len(strokes) > MAX_STROKES:
        # Rysunek techniczny / mapa: tysiące ścieżek, a łączenie w grupy
        # rośnie kwadratowo. Takie strony nie są typowym miejscem podpisu.
        return []
    rects = []
    for members in _cluster(strokes):
        rect = _looks_handwritten(members, page.rect)
        if rect is not None:
            rects.append(rect & page.rect)
    return rects


# --- B: podpisy na skanach (atrament przy słowie „podpis”) --------------------

# Kotwica = samo słowo „podpis” w formie z formularza: „Podpis:”, „(podpis)”,
# „Podpisy”, „podpisu”, „czytelny podpis”, „signature”. NIE „podpisany”
# ani „podpisania” – to treść pisma („Ja, niżej podpisany…”), nie miejsce podpisu.
_ANCHOR = re.compile(r"^(podpis|podpisy|podpisu|podpisem|signature)$")
_ANCHOR_EDGES = ".,;:!?()[]\"'„”«»"
# Obszar szukania wokół kotwicy (punkty PDF): podpis bywa nad kropkami
# z „(podpis)” pod spodem, obok „Podpis:” albo pod „Podpis Najemcy”.
SEARCH_LEFT, SEARCH_RIGHT = 60, 230
SEARCH_UP, SEARCH_DOWN = 70, 70
# Piksel „atramentu”: ciemniejszy niż to (0 = czarny, 255 = biały). Skany mają
# szare tło i szum – 110 odcina tło, a zostawia pismo długopisem.
INK_LEVEL = 110
# Ile atramentu (poza rozpoznanym tekstem i liniami) musi zostać, żeby
# uznać obszar za podpis: odcina kropki, szum skanu i pojedyncze kreski.
MIN_INK_PIXELS = 120
# Margines wokół słów tekstu (pt) czyszczony przed liczeniem atramentu –
# litery wychodzą trochę poza prostokąt słowa.
WORD_MARGIN = 2
# Wiersz / kolumna obszaru zajęta atramentem w tej części to linia
# („…………”, „_____”, ramka tabeli), nie pismo.
LINE_SHARE = 0.5
# Plamka mniejsza niż tyle pikseli w obu wymiarach (przy 150 dpi ok. 3 pt)
# to kropka z „……”, przecinek albo szum skanu – nie pociągnięcie pióra.
MIN_STROKE_PIXELS = 7


def _is_anchor(word: str) -> bool:
    return bool(_ANCHOR.match(word.strip(_ANCHOR_EDGES).casefold()))


def _drop_specks(region):
    """Usuwa małe, oddzielne plamki (kropki, szum), zostawia pociągnięcia.

    Składowe spójne liczone bez scipy: przeszukiwanie po pikselach atramentu
    (w obszarze przy podpisie jest ich najwyżej kilka tysięcy).
    """
    import numpy as np

    height, width = region.shape
    seen = np.zeros_like(region)
    keep = np.zeros_like(region)
    for start in zip(*np.nonzero(region)):
        if seen[start]:
            continue
        stack, pixels = [start], []
        seen[start] = True
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if 0 <= ny < height and 0 <= nx < width and region[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        ys = [y for y, _ in pixels]
        xs = [x for _, x in pixels]
        if max(ys) - min(ys) + 1 >= MIN_STROKE_PIXELS or max(xs) - min(xs) + 1 >= MIN_STROKE_PIXELS:
            for pixel in pixels:
                keep[pixel] = True
    return keep


def find_scan_signatures(pixmap: pymupdf.Pixmap, page_rect: pymupdf.Rect,
                         words: list[tuple[str, pymupdf.Rect]]) -> list[pymupdf.IRect]:
    """Podpisy odręczne na obrazie strony-skanu, w pikselach tej pixmapy.

    words: słowa strony (warstwa tekstowa skanu i/lub OCR) z położeniem w pt.
    Na skanie nie ma osobnego obrazu podpisu ani ścieżek – jest tylko piksel.
    Dlatego: przy każdym słowie „podpis” bierzemy okolicę, czyścimy w niej
    piksele rozpoznanego tekstu i linie, a to, co zostało ciemnego, to pismo
    odręczne. Bez słowa „podpis” nic nie zaczerniamy (inaczej każda
    odręczna adnotacja i każde logo na skanie dostałoby pasek).
    """
    import numpy as np

    anchors = [rect for text, rect in words if _is_anchor(text)]
    if not anchors:
        return []
    scale_x, scale_y = pixmap.width / page_rect.width, pixmap.height / page_rect.height
    samples = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    gray = samples[:, :, : min(3, pixmap.n)].min(axis=2)   # najciemniejszy kanał = atrament w kolorze też
    ink = gray < INK_LEVEL

    def to_pixels(rect: pymupdf.Rect) -> tuple[int, int, int, int]:
        x0 = max(0, int((rect.x0 - page_rect.x0) * scale_x))
        y0 = max(0, int((rect.y0 - page_rect.y0) * scale_y))
        x1 = min(pixmap.width, int((rect.x1 - page_rect.x0) * scale_x) + 1)
        y1 = min(pixmap.height, int((rect.y1 - page_rect.y0) * scale_y) + 1)
        return x0, y0, x1, y1

    # Tekst rozpoznany na stronie to nie podpis – czyścimy go na kopii.
    ink = ink.copy()
    for _, rect in words:
        x0, y0, x1, y1 = to_pixels(rect + (-WORD_MARGIN, -WORD_MARGIN, WORD_MARGIN, WORD_MARGIN))
        ink[y0:y1, x0:x1] = False

    found: list[pymupdf.IRect] = []
    for anchor in anchors:
        area = pymupdf.Rect(anchor.x0 - SEARCH_LEFT, anchor.y0 - SEARCH_UP,
                            anchor.x1 + SEARCH_RIGHT, anchor.y1 + SEARCH_DOWN) & page_rect
        x0, y0, x1, y1 = to_pixels(area)
        region = ink[y0:y1, x0:x1].copy()
        if region.size == 0:
            continue
        # Linie do podpisu i ramki: wiersze/kolumny prawie całe w atramencie.
        region[region.mean(axis=1) > LINE_SHARE, :] = False
        region[:, region.mean(axis=0) > LINE_SHARE] = False
        if region.sum() < MIN_INK_PIXELS:
            continue
        region = _drop_specks(region)
        if region.sum() < MIN_INK_PIXELS:
            continue
        rows = np.flatnonzero(region.any(axis=1))
        cols = np.flatnonzero(region.any(axis=0))
        rect = pymupdf.IRect(x0 + cols[0] - 4, y0 + rows[0] - 4,
                             x0 + cols[-1] + 5, y0 + rows[-1] + 5) & pixmap.irect
        if not rect.is_empty and not any(rect in other for other in found):
            found.append(rect)
    return found
