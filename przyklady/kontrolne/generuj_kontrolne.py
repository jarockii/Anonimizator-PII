"""Zestaw KONTROLNY: dokumenty, których nie używano przy dobieraniu reguł,
progów ani przy treningu GLiNER-a (training/gliner_dane.py wyklucza ich słowa).

Dlaczego osobno: wynik na przyklady/1–8 jest optymistyczny – na tych samych
plikach porównywaliśmy kolejne wersje modelu i poprawiali reguły. Tu
sprawdzamy, jak aplikacja radzi sobie z czymś nowym. Plików kontrolnych NIE
poprawiamy pod wynik; nowe przypadki dokładamy jako kolejny plik.

Wszystkie dane są FIKCYJNE. Numery (PESEL, IBAN, NIP, dowód) mają tylko
poprawne sumy kontrolne, żeby Regex mógł je rozpoznać.

Uruchomienie (z katalogu projektu):
    python przyklady/kontrolne/generuj_kontrolne.py
"""

import io
import math
import random
from pathlib import Path

import pymupdf
from docx import Document
from docx.shared import Cm, Pt

OUT = Path(__file__).resolve().parent
A4 = pymupdf.paper_rect("a4")
CSS = ("* { font-family: sans-serif; font-size: 10.5pt; line-height: 1.35; } "
       "h2 { font-size: 14pt; } td, th { padding: 2px 6px; } "
       "table { border-collapse: collapse; } th { text-align: left; background: #eeeeee; }")

rng = random.Random(4711)


# --- Numery z poprawnymi sumami kontrolnymi --------------------------------------

def pesel(year: int, month: int, day: int, male: bool) -> str:
    month_code = month + (20 if year >= 2000 else 0)
    serial = f"{rng.randint(0, 99):02d}{rng.randrange(1 if male else 0, 10, 2)}"
    digits = f"{year % 100:02d}{month_code:02d}{day:02d}{rng.randint(0, 9)}{serial}"
    weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    return digits + str((10 - sum(int(d) * w for d, w in zip(digits, weights)) % 10) % 10)


def iban(bank: str) -> str:
    bban = bank + "".join(str(rng.randint(0, 9)) for _ in range(24 - len(bank)))
    check = 98 - int(bban + "252100") % 97
    s = f"PL{check:02d}{bban}"
    return " ".join([s[:4]] + [s[i:i + 4] for i in range(4, len(s), 4)])


def nip() -> str:
    weights = [6, 5, 7, 2, 3, 4, 5, 6, 7]
    while True:
        digits = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(8)]
        check = sum(d * w for d, w in zip(digits, weights)) % 11
        if check != 10:
            s = "".join(map(str, digits)) + str(check)
            return f"{s[:3]}-{s[3:6]}-{s[6:8]}-{s[8:]}"


def id_card() -> str:
    """Dowód osobisty: 3 litery + cyfra kontrolna + 5 cyfr (wagi 7,3,1,9,7,3,1,7,3)."""
    letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3))
    digits = [rng.randint(0, 9) for _ in range(5)]
    values = [ord(c) - 55 for c in letters]
    total = sum(v * w for v, w in zip(values, (7, 3, 1))) + sum(d * w for d, w in zip(digits, (7, 3, 1, 7, 3)))
    return f"{letters}{total % 10}{''.join(map(str, digits))}"


# --- Pomocnicze -------------------------------------------------------------------

def faint_watermark(page: pymupdf.Page, text: str) -> None:
    """Ukośny, półprzezroczysty napis przez środek strony (jak „KOPIA” w banku)."""
    writer = pymupdf.TextWriter(page.rect, opacity=0.18, color=(0.5, 0.5, 0.5))
    writer.append((150, 560), text, fontsize=110)
    writer.write_text(page, morph=(pymupdf.Point(150, 560), pymupdf.Matrix(-35)))


def handwriting(page: pymupdf.Page, x: float, y: float, width: float = 140) -> None:
    """Odręczny podpis: pętle liter i zamaszysta kreska."""
    steps = 140
    points = [pymupdf.Point(x + width * i / steps + 6 * math.cos(i / steps * 14 * math.pi),
                            y + 8 * math.sin(i / steps * 14 * math.pi) + 4 * math.sin(i / 23))
              for i in range(steps)]
    shape = page.new_shape()
    shape.draw_polyline(points)
    shape.draw_bezier((x - 5, y + 14), (x + 40, y - 22), (x + 100, y + 26), (x + width + 10, y - 4))
    shape.finish(color=(0.05, 0.1, 0.45), fill=None, width=1.3, closePath=False)
    shape.commit()


def to_scan(page: pymupdf.Page, dpi: int = 200) -> pymupdf.Pixmap:
    """Strona -> obraz w skali szarości z szarym tłem i szumem (jak skaner)."""
    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    samples = bytearray(pixmap.samples)
    for i, value in enumerate(samples):
        if value > 200:
            samples[i] = 230 + rng.randint(-9, 9)
        elif rng.random() < 0.015:
            samples[i] = min(255, value + 50)
    return pymupdf.Pixmap(pixmap.colorspace, pixmap.width, pixmap.height, bytes(samples), pixmap.alpha)


# --- K1: wyciąg bankowy ze znakiem wodnym ---------------------------------------

def k1_wyciag() -> pymupdf.Document:
    rachunek = iban("19401234")
    pracodawca_nip = nip()
    rows = [
        ("01.03.2026", "Saldo początkowe", "", "", "3 412,56"),
        ("02.03.2026", f"Wynagrodzenie za 02/2026 – HYDRO-TECH PSZCZYNA SP. Z O.O., NIP {pracodawca_nip}", "", "7 840,00", "11 252,56"),
        ("03.03.2026", "Przelew do: Oksana Melnyk – czynsz za marzec, lokal przy ul. Bielskiej 9/2", "2 300,00", "", "8 952,56"),
        ("05.03.2026", "BLIK na telefon +48 512 784 903", "150,00", "", "8 802,56"),
        ("06.03.2026", "Przelew od: Celina Krzemień – zwrot za bilety do Zakopanego", "", "268,00", "9 070,56"),
        ("09.03.2026", "Opłata za przedszkole nr 7 – Antoś Pstrąg, grupa Biedronki", "640,00", "", "8 430,56"),
        ("12.03.2026", "Płatność kartą – Apteka Pod Lipą, Pszczyna", "87,40", "", "8 343,16"),
        ("15.03.2026", "Wpłata gotówkowa – Zdzisław Wrona, tytuł: darowizna na komunię", "", "500,00", "8 843,16"),
        ("20.03.2026", "Spłata raty kredytu hipotecznego nr KH/2019/00731", "1 912,33", "", "6 930,83"),
        ("31.03.2026", "Saldo końcowe", "", "", "6 930,83"),
    ]
    table = "".join(f"<tr><td>{d}</td><td>{o}</td><td>{w}</td><td>{u}</td><td>{s}</td></tr>"
                    for d, o, w, u, s in rows)
    html = f"""
<h2>Bank Spółdzielczy Ziemi Pszczyńskiej</h2>
<p>WYCIĄG Z RACHUNKU nr 3/2026 · okres 01.03.2026 – 31.03.2026</p>
<p>Posiadacz rachunku: <b>Hieronim Pstrąg</b><br>
Adres: ul. Wrzosowa 14/3, 43-200 Pszczyna<br>
Numer rachunku: {rachunek}</p>
<table border="1">
<tr><th>Data</th><th>Opis operacji</th><th>Obciążenia</th><th>Uznania</th><th>Saldo</th></tr>
{table}
</table>
<p>Wyciąg wygenerowany elektronicznie, nie wymaga podpisu ani stempla.</p>
"""
    document = pymupdf.open()
    page = document.new_page(width=A4.width, height=A4.height)
    page.insert_htmlbox(A4 + (45, 45, -45, -45), html, css=CSS)
    faint_watermark(page, "KOPIA")
    return document


# --- K2: wniosek .docx ze zdjęciem dowodu --------------------------------------

def id_photo(name: str, surname: str, number: str, birth: str, pesel_no: str) -> bytes:
    """„Zdjęcie” dowodu osobistego jako obraz PNG (dane tylko na obrazie i w tabeli)."""
    document = pymupdf.open()
    page = document.new_page(width=320, height=200)
    page.draw_rect(pymupdf.Rect(4, 4, 316, 196), color=(0.3, 0.4, 0.6), fill=(0.9, 0.93, 0.97), width=2)
    page.draw_rect(pymupdf.Rect(16, 40, 96, 140), color=(0.5, 0.5, 0.5), fill=(0.8, 0.8, 0.8))
    lines = [("RZECZPOSPOLITA POLSKA · DOWÓD OSOBISTY", 9, 24, 16),
             (f"Nazwisko: {surname}", 10, 110, 56), (f"Imię: {name}", 10, 110, 74),
             (f"Data urodzenia: {birth}", 10, 110, 92), (f"PESEL: {pesel_no}", 10, 110, 110),
             (f"Nr dokumentu: {number}", 10, 110, 128)]
    for text, size, x, y in lines:
        page.insert_text((x, y), text, fontsize=size, fontname="helv")
    return page.get_pixmap(dpi=160).tobytes("png")


def k2_wniosek() -> bytes:
    name, surname = "Teodora", "Hryniewicz"
    pesel_no = pesel(1971, 11, 4, male=False)
    birth = f"{pesel_no[4:6]}.11.1971"
    card = id_card()
    document = Document()
    style = document.styles["Normal"]
    style.font.name, style.font.size = "Calibri", Pt(11)
    document.add_heading("Wniosek o wydanie karty parkingowej dla osoby niepełnosprawnej", level=1)
    document.add_paragraph("Do: Powiatowy Zespół do Spraw Orzekania o Niepełnosprawności w Żywcu")
    table = document.add_table(rows=5, cols=2)
    table.style = "Table Grid"
    for row, (label, value) in zip(table.rows, [
        ("Imię i nazwisko wnioskodawcy", f"{name} {surname}"),
        ("PESEL", pesel_no),
        ("Adres zamieszkania", "Juszczyna 212, 34-312 Juszczyna"),
        ("Telefon", "609 118 427"),
        ("E-mail", "t.hryniewicz71@skrzynka-kontrolna.pl"),
    ]):
        row.cells[0].text, row.cells[1].text = label, value
    document.add_paragraph()
    document.add_paragraph(
        "Uzasadnienie: od 2019 r. choruję na stwardnienie rozsiane, poruszam się o kulach. "
        "Posiadam orzeczenie o znacznym stopniu niepełnosprawności (symbol 05-R). "
        "Kartę potrzebuję do dojazdów na rehabilitację do szpitala w Żywcu.")
    document.add_paragraph(
        "W sprawie wniosku proszę kontaktować się także z moim synem, który jest "
        "jedynym fizjoterapeutą w przychodni w Juszczynie.")
    document.add_paragraph("Załącznik nr 1 – skan dowodu osobistego:")
    document.add_picture(io.BytesIO(id_photo(name, surname.upper(), card, birth, pesel_no)), width=Cm(9))
    document.add_paragraph("Oświadczam, że dane podane we wniosku są zgodne z prawdą.")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# --- K3: skan wypisu ze szpitala z podpisem lekarza ------------------------------

def k3_wypis() -> pymupdf.Document:
    pesel_no = pesel(1948, 2, 17, male=False)
    html = f"""
<h2>KARTA INFORMACYJNA LECZENIA SZPITALNEGO</h2>
<p>Szpital Powiatowy w Wadowicach · Oddział Chorób Wewnętrznych</p>
<p>Pacjentka: <b>Marianna Sołtys</b>, PESEL {pesel_no}<br>
Adres: Zawadka 47, 34-100 Wadowice<br>
Przyjęta: 02.02.2026 · Wypisana: 11.02.2026</p>
<p><b>Rozpoznanie:</b> zaostrzenie przewlekłej obturacyjnej choroby płuc, cukrzyca typu 2,
niewydolność serca NYHA II.</p>
<p><b>Zalecenia:</b> kontrola w poradni pulmonologicznej za 4 tygodnie, insulina według
schematu, ograniczenie soli. Pacjentka mieszka sama; opiekę sprawuje córka, sołtyska wsi Zawadka.</p>
<p>Lekarz prowadzący: lek. Ignacy Lipiec</p>
<p>Ordynator: dr n. med. Radosław Kępka</p>
<p style="margin-top: 40px">Podpis lekarza:</p>
"""
    source = pymupdf.open()
    page = source.new_page(width=A4.width, height=A4.height)
    page.insert_htmlbox(A4 + (55, 50, -55, -60), html, css=CSS)
    label = page.search_for("Podpis lekarza")[0]
    handwriting(page, label.x1 + 20, label.y1 - 4)
    scan = pymupdf.open()
    scan.new_page(width=A4.width, height=A4.height).insert_image(A4, pixmap=to_scan(page))
    return scan


# --- K4: protokół zebrania wspólnoty (potocznie, identyfikacja pośrednia) --------

def k4_protokol() -> pymupdf.Document:
    html = """
<h2>Protokół z zebrania Wspólnoty Mieszkaniowej „Jaskółcza 3”</h2>
<p>Zebranie odbyło się 14 stycznia 2026 r. w świetlicy osiedlowej. Obecnych 11 właścicieli lokali.
Zebranie prowadziła przewodnicząca zarządu, Grażyna Pleban (lokal 12).</p>
<p>1. Sprawa hałasu: pan z czwórki, ten co jeździ czerwoną taksówką na nocne kursy, ponownie
parkuje na trawniku. Numer rejestracyjny SPS 4K21 przekazano straży miejskiej.
Zdzichu z parteru zgłosił, że to już piąty raz w tym miesiącu.</p>
<p>2. Pani Wiesia Dziedzic poprosiła o rozłożenie zaległości na raty – choruje na depresję
i od jesieni przebywa na zwolnieniu lekarskim. Zarząd się zgodził.</p>
<p>3. Kontakt do administratora: pięćset trzydzieści jeden, dwieście dwadzieścia, osiemset
czternaście. Faktury proszę wysyłać na adres wspolnota.jaskolcza3 (at) poczta-kontrolna kropka pl.</p>
<p>4. Księga wieczysta nieruchomości: KA1P/00067213/4. Wspólnota nie zgadza się na sprzedaż
działki nr 418/7 sąsiadowi, właścicielowi jedynej apteki na osiedlu.</p>
<p>Protokołowała: Wiesława Dziedzic</p>
"""
    document = pymupdf.open()
    page = document.new_page(width=A4.width, height=A4.height)
    page.insert_htmlbox(A4 + (55, 50, -55, -60), html, css=CSS)
    return document


if __name__ == "__main__":
    k1_wyciag().save(OUT / "k1_wyciag_bankowy_znak_wodny.pdf", garbage=3, deflate=True)
    (OUT / "k2_wniosek_ze_skanem_dowodu.docx").write_bytes(k2_wniosek())
    k3_wypis().save(OUT / "k3_skan_wypisu_z_podpisem.pdf", garbage=3, deflate=True)
    k4_protokol().save(OUT / "k4_protokol_wspolnoty.pdf", garbage=3, deflate=True)
    print("Zapisano 4 dokumenty kontrolne w", OUT)
