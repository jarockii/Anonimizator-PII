"""Generator polskich danych treningowych dla GLiNER-a.

Pomiar (eval/szara_strefa.py) pokazał, że GLiNER po polsku myli się głównie przy
ODMIANIE nazwisk (celownik: „Brzęczyszczykiewiczowi”, „Szewczence”). Ten skrypt
tworzy sztuczne zdania z fikcyjnymi osobami, adresami, firmami i loginami we
wszystkich przypadkach, a do tego zdania z rolami („Najemca”, „szwagier”),
w których NIE MA osoby – żeby model nauczył się, że rola to nie imię.

Druga runda: przykłady negatywne (wszystkie etykiety puste). Po 1. treningu model
maskował same nazwy identyfikatorów („PESEL”), numery wewnętrzne („wew. 214”),
nazwy miast bez adresu, stanowiska („Oberarzt”) i sklepy pisane małą literą.

Trzecia runda: tytuł/stanowisko + osoba („dr Anna Nowak”, „Head of Sales: …”) –
same negatywy sprawiły, że model przestał widzieć osoby stojące po tytule.

Ważne: ŻADNE imię, nazwisko ani ulica z dokumentów testowych (eval/oznaczenia)
nie trafia do treningu. Inaczej model „nauczyłby się sprawdzianu na pamięć”
i wynik pomiaru byłby nieuczciwy.

Uruchom z katalogu projektu:
    python training/gliner_dane.py
Wynik: training/dane/gliner_train.jsonl i gliner_val.jsonl
Autor: Wiktor Jarocki
"""

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from scoring import load_annotations  # noqa: E402

WYJSCIE = ROOT / "training" / "dane"
ILE_TRENING = 2000
ILE_WALIDACJA = 250
ZIARNO = 42      # to samo ziarno = te same dane przy każdym uruchomieniu

# Przypadki: M(ianownik), D(opełniacz), C(elownik), B(iernik), N(arzędnik), Ms(miejscownik), W(ołacz)
PRZYPADKI = ["M", "D", "C", "B", "N", "Ms", "W"]

# ---------------------------------------------------------------------------
# 1. Imiona (wszystkie 7 przypadków)
# ---------------------------------------------------------------------------

IMIONA_M = {
    "Tomasz": "Tomasza Tomaszowi Tomasza Tomaszem Tomaszu Tomaszu",
    "Marek": "Marka Markowi Marka Markiem Marku Marku",
    "Paweł": "Pawła Pawłowi Pawła Pawłem Pawle Pawle",
    "Piotr": "Piotra Piotrowi Piotra Piotrem Piotrze Piotrze",
    "Michał": "Michała Michałowi Michała Michałem Michale Michale",
    "Łukasz": "Łukasza Łukaszowi Łukasza Łukaszem Łukaszu Łukaszu",
    "Adam": "Adama Adamowi Adama Adamem Adamie Adamie",
    "Rafał": "Rafała Rafałowi Rafała Rafałem Rafale Rafale",
    "Kamil": "Kamila Kamilowi Kamila Kamilem Kamilu Kamilu",
    "Jakub": "Jakuba Jakubowi Jakuba Jakubem Jakubie Jakubie",
    "Dariusz": "Dariusza Dariuszowi Dariusza Dariuszem Dariuszu Dariuszu",
    "Robert": "Roberta Robertowi Roberta Robertem Robercie Robercie",
    "Artur": "Artura Arturowi Artura Arturem Arturze Arturze",
    "Szymon": "Szymona Szymonowi Szymona Szymonem Szymonie Szymonie",
    "Filip": "Filipa Filipowi Filipa Filipem Filipie Filipie",
    "Patryk": "Patryka Patrykowi Patryka Patrykiem Patryku Patryku",
    "Stanisław": "Stanisława Stanisławowi Stanisława Stanisławem Stanisławie Stanisławie",
    "Mikołaj": "Mikołaja Mikołajowi Mikołaja Mikołajem Mikołaju Mikołaju",
    "Bogdan": "Bogdana Bogdanowi Bogdana Bogdanem Bogdanie Bogdanie",
    "Jerzy": "Jerzego Jerzemu Jerzego Jerzym Jerzym Jerzy",
}

IMIONA_K = {
    "Anna": "Anny Annie Annę Anną Annie Anno",
    "Katarzyna": "Katarzyny Katarzynie Katarzynę Katarzyną Katarzynie Katarzyno",
    "Magdalena": "Magdaleny Magdalenie Magdalenę Magdaleną Magdalenie Magdaleno",
    "Joanna": "Joanny Joannie Joannę Joanną Joannie Joanno",
    "Monika": "Moniki Monice Monikę Moniką Monice Moniko",
    "Ewa": "Ewy Ewie Ewę Ewą Ewie Ewo",
    "Agata": "Agaty Agacie Agatę Agatą Agacie Agato",
    "Natalia": "Natalii Natalii Natalię Natalią Natalii Natalio",
    "Paulina": "Pauliny Paulinie Paulinę Pauliną Paulinie Paulino",
    "Justyna": "Justyny Justynie Justynę Justyną Justynie Justyno",
    "Barbara": "Barbary Barbarze Barbarę Barbarą Barbarze Barbaro",
    "Iwona": "Iwony Iwonie Iwonę Iwoną Iwonie Iwono",
    "Beata": "Beaty Beacie Beatę Beatą Beacie Beato",
    "Dorota": "Doroty Dorocie Dorotę Dorotą Dorocie Doroto",
    "Elżbieta": "Elżbiety Elżbiecie Elżbietę Elżbietą Elżbiecie Elżbieto",
    "Weronika": "Weroniki Weronice Weronikę Weroniką Weronice Weroniko",
    "Julia": "Julii Julii Julię Julią Julii Julio",
    "Zofia": "Zofii Zofii Zofię Zofią Zofii Zofio",
    "Alicja": "Alicji Alicji Alicję Alicją Alicji Alicjo",
    "Renata": "Renaty Renacie Renatę Renatą Renacie Renato",
}

# ---------------------------------------------------------------------------
# 2. Nazwiska – trzy wzorce odmiany
# ---------------------------------------------------------------------------

# a) przymiotnikowe: -ski / -cki / -dzki (odmieniają się i u mężczyzn, i u kobiet)
RDZENIE_SKI = ["Lewandowsk", "Dąbrowsk", "Zielińsk", "Szymańsk", "Kozłowsk", "Jankowsk",
               "Michalsk", "Pawłowsk", "Sikorsk", "Baranowsk", "Rutkowsk", "Malinowsk",
               "Ostrowsk", "Tomaszewsk", "Jasińsk", "Krajewsk", "Sadowsk", "Głowack",
               "Wysock", "Czarneck", "Sobieck", "Borowsk", "Górsk", "Kalinowsk"]
KONCOWKI_SKI_M = ["i", "iego", "iemu", "iego", "im", "im", "i"]
KONCOWKI_SKI_K = ["a", "iej", "iej", "ą", "ą", "iej", "a"]

# b) rzeczownikowe męskie (kobiety z takim nazwiskiem go NIE odmieniają: „Annie Nowak”)
NAZWISKA_RZECZ = {
    "Nowak": "Nowaka Nowakowi Nowaka Nowakiem Nowaku Nowaku",
    "Kaczmarek": "Kaczmarka Kaczmarkowi Kaczmarka Kaczmarkiem Kaczmarku Kaczmarku",
    "Mazur": "Mazura Mazurowi Mazura Mazurem Mazurze Mazurze",
    "Krawczyk": "Krawczyka Krawczykowi Krawczyka Krawczykiem Krawczyku Krawczyku",
    "Kubiak": "Kubiaka Kubiakowi Kubiaka Kubiakiem Kubiaku Kubiaku",
    "Wójcik": "Wójcika Wójcikowi Wójcika Wójcikiem Wójciku Wójciku",
    "Pietrzak": "Pietrzaka Pietrzakowi Pietrzaka Pietrzakiem Pietrzaku Pietrzaku",
    "Michalak": "Michalaka Michalakowi Michalaka Michalakiem Michalaku Michalaku",
    "Jankiewicz": "Jankiewicza Jankiewiczowi Jankiewicza Jankiewiczem Jankiewiczu Jankiewiczu",
    "Bartkiewicz": "Bartkiewicza Bartkiewiczowi Bartkiewicza Bartkiewiczem Bartkiewiczu Bartkiewiczu",
    # nazwiska, które są zwykłymi słowami – trudne dla modelu
    "Kot": "Kota Kotowi Kota Kotem Kocie Kocie",
    "Baran": "Barana Baranowi Barana Baranem Baranie Baranie",
    "Kruk": "Kruka Krukowi Kruka Krukiem Kruku Kruku",
    "Gołąb": "Gołębia Gołębiowi Gołębia Gołębiem Gołębiu Gołębiu",
    "Kozioł": "Kozła Kozłowi Kozła Kozłem Koźle Koźle",
    "Sikora": "Sikory Sikorze Sikorę Sikorą Sikorze Sikoro",
}

# c) ukraińskie na -ko (u mężczyzn odmiana jak „Szewczenko → Szewczence”)
NAZWISKA_KO = {
    "Bondarenko": "Bondarenki Bondarence Bondarenkę Bondarenką Bondarence Bondarenko",
    "Tkaczenko": "Tkaczenki Tkaczence Tkaczenkę Tkaczenką Tkaczence Tkaczenko",
    "Kowalenko": "Kowalenki Kowalence Kowalenkę Kowalenką Kowalence Kowalenko",
    "Ponomarenko": "Ponomarenki Ponomarence Ponomarenkę Ponomarenką Ponomarence Ponomarenko",
    "Łysenko": "Łysenki Łysence Łysenkę Łysenką Łysence Łysenko",
}


def formy(mianownik, reszta):
    """'Marek', 'Marka Markowi …' -> {'M': 'Marek', 'D': 'Marka', …}"""
    return dict(zip(PRZYPADKI, [mianownik] + reszta.split()))


def nazwisko_form(nazwisko, plec):
    """Formy nazwiska w 7 przypadkach dla danej płci ('m' albo 'k')."""
    for rdzen in RDZENIE_SKI:
        if nazwisko.startswith(rdzen) and nazwisko[len(rdzen):] in ("i", "a"):
            konc = KONCOWKI_SKI_M if plec == "m" else KONCOWKI_SKI_K
            return dict(zip(PRZYPADKI, [rdzen + k for k in konc]))
    slownik = NAZWISKA_RZECZ if nazwisko in NAZWISKA_RZECZ else NAZWISKA_KO
    if plec == "k":                       # kobieta: „Anna Nowak”, „Annie Nowak”
        return {p: nazwisko for p in PRZYPADKI}
    return formy(nazwisko, slownik[nazwisko])


# ---------------------------------------------------------------------------
# 3. Adresy, firmy, loginy
# ---------------------------------------------------------------------------

# ulica: (mianownik, miejscownik) – „ul. Kwiatowa 5”, „mieszka na Kwiatowej 5”
ULICE = [("Kwiatowa", "Kwiatowej"), ("Polna", "Polnej"), ("Leśna", "Leśnej"),
         ("Ogrodowa", "Ogrodowej"), ("Szkolna", "Szkolnej"), ("Krótka", "Krótkiej"),
         ("Akacjowa", "Akacjowej"), ("Brzozowa", "Brzozowej"), ("Klonowa", "Klonowej"),
         ("Zielona", "Zielonej"), ("Spacerowa", "Spacerowej"), ("Wiśniowa", "Wiśniowej"),
         ("Jasna", "Jasnej"), ("Kolejowa", "Kolejowej"), ("Wąska", "Wąskiej")]
# ulice „z nazwiska” – ta sama forma w każdym kontekście; NIE są osobami
ULICE_NAZWISKA = ["Mickiewicza", "Kościuszki", "Reymonta", "Sienkiewicza", "Konopnickiej",
                  "Prusa", "Żeromskiego", "Moniuszki"]
MIASTA = [("20-345", "Lublin"), ("25-512", "Kielce"), ("10-217", "Olsztyn"),
          ("35-060", "Rzeszów"), ("45-040", "Opole"), ("09-400", "Płock"),
          ("82-300", "Elbląg"), ("22-400", "Zamość"), ("08-110", "Siedlce"),
          ("30-150", "Kraków"), ("60-688", "Poznań"), ("50-120", "Wrocław")]
WSIE = ["Nowa Wola", "Dąbrówka Mała", "Zalesie Górne", "Kępa Wielka", "Borki Stare"]

FIRMY = ["Polmar Logistics", "Ekobud sp. z o.o.", "Transmar S.A.", "Optiprint",
         "Mazurskie Ogrody", "Kwadrat Serwis", "Nordbit sp. z o.o.", "Agro-Mix",
         "Biuro Rachunkowe Bilans", "Zielona Dolina", "Termex Instalacje", "Kancelaria Lexus Prawo"]

# role i relacje – w zdaniach z nimi NIE ma osoby
ROLE = ["Najemca", "Wynajmujący", "Poręczyciel", "Leasingobiorca", "Zleceniobiorca",
        "Pożyczkodawca", "Kupujący", "Sprzedawca", "Pracodawca", "Pracownik",
        "Pełnomocnik", "Prezes Zarządu", "Kierownik zmiany", "Ubezpieczony"]


def bez_ogonkow(tekst):
    return tekst.translate(str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ"))


# ---------------------------------------------------------------------------
# 4. Wykluczenie wszystkiego, co jest w dokumentach testowych
# ---------------------------------------------------------------------------

def slowa_z_testow():
    """Wszystkie słowa z oznaczeń w eval/ (małymi literami)."""
    slowa = set()
    for adnotacja in load_annotations(ROOT / "eval" / "oznaczenia", ROOT):
        for item in adnotacja.items:
            for fragment in item.fragments:
                slowa.update(w.casefold() for w in re.findall(r"\w+", fragment))
    return slowa


def wyczysc_listy():
    """Usuwa z list każde imię/nazwisko/ulicę, której jakakolwiek forma jest w testach."""
    zakazane = slowa_z_testow()

    def czyste(*formy_slowa):
        return not any(f.casefold() in zakazane for f in formy_slowa)

    for slownik in (IMIONA_M, IMIONA_K, NAZWISKA_RZECZ, NAZWISKA_KO):
        for klucz in list(slownik):
            if not czyste(klucz, *slownik[klucz].split()):
                del slownik[klucz]
    RDZENIE_SKI[:] = [r for r in RDZENIE_SKI
                      if czyste(*(r + k for k in KONCOWKI_SKI_M + KONCOWKI_SKI_K))]
    ULICE[:] = [u for u in ULICE if czyste(*u)]
    ULICE_NAZWISKA[:] = [u for u in ULICE_NAZWISKA if czyste(u)]
    # przykłady negatywne – te same zasady: żadne słowo z testów
    MIASTA_NEG[:] = [m for m in MIASTA_NEG if czyste(*m)]
    SKLEPY[:] = [s for s in SKLEPY if czyste(*s)]
    for lista in (IDENTYFIKATORY, STANOWISKA, TYTULY):
        lista[:] = [x for x in lista if czyste(*re.findall(r"\w+", x))]
    return zakazane


# ---------------------------------------------------------------------------
# 5. Losowanie osób, adresów…
# ---------------------------------------------------------------------------

def losuj_osobe(los):
    plec = los.choice("mk")
    imiona = IMIONA_M if plec == "m" else IMIONA_K
    imie = los.choice(list(imiona))
    pula = ([r + ("i" if plec == "m" else "a") for r in RDZENIE_SKI]
            + list(NAZWISKA_RZECZ) + list(NAZWISKA_KO))
    nazwisko = los.choice(pula)
    return {"plec": plec, "imie": formy(imie, imiona[imie]),
            "nazwisko": nazwisko_form(nazwisko, plec),
            "nazwisko_m": nazwisko}


def zapis_osoby(osoba, przypadek, los):
    """Jak osoba pojawia się w tekście: pełne imię i nazwisko, samo nazwisko, inicjał…"""
    imie = osoba["imie"][przypadek]
    nazw = osoba["nazwisko"][przypadek]
    wariant = los.random()
    if wariant < 0.55:
        return f"{imie} {nazw}"
    if wariant < 0.75:
        return nazw
    if wariant < 0.87 and przypadek == "M":
        return f"{imie[0]}. {nazw}"
    return imie


def losuj_adres(los, forma="M"):
    """Zwraca tekst adresu. forma 'M': „ul. Kwiatowa 5…”, 'Ms': „Kwiatowej 5…” (po „na”/„przy ul.”)."""
    nr = los.randint(1, 140)
    lokal = f"/{los.randint(1, 60)}" if los.random() < 0.5 else ""
    kod, miasto = los.choice(MIASTA)
    if forma == "M" and los.random() < 0.12:                  # adres wiejski
        wies = los.choice(WSIE)
        return f"{wies} {nr}, {kod} {wies}"
    if los.random() < 0.2:
        ulica = los.choice(ULICE_NAZWISKA)
        baza = f"{ulica} {nr}{lokal}"
        przedrostek = "ul. " if forma == "M" else ""
    else:
        m, ms = los.choice(ULICE)
        baza = f"{m if forma == 'M' else ms} {nr}"
        if lokal and los.random() < 0.5:
            baza += f" m. {lokal[1:]}"
        else:
            baza += lokal
        przedrostek = "ul. " if forma == "M" and los.random() < 0.8 else ""
    if los.random() < 0.5:
        return f"{przedrostek}{baza}, {kod} {miasto}"
    return f"{przedrostek}{baza}"


def losuj_login(osoba, los):
    i = bez_ogonkow(osoba["imie"]["M"]).lower()
    n = bez_ogonkow(osoba["nazwisko_m"]).lower()
    wzory = [f"{i[0]}{n}", f"{i}.{n}", f"{n}{los.randint(1, 99)}", f"{i}_{n[:3]}{los.randint(70, 99)}"]
    return los.choice(wzory)


# ---------------------------------------------------------------------------
# 6. Szablony zdań. {X} = osoba w danym przypadku, {a} = końcówka czasownika (""/„a”)
# ---------------------------------------------------------------------------

SZABLONY_OSOBA = {
    "M": ["{X} podpisał{a} umowę w dniu wczorajszym.", "W spotkaniu uczestniczył{a} {X}.",
          "{X} zgłosił{a} awarię w poniedziałek.", "Wniosek złożył{a} {X}.",
          "Za realizację odpowiada {X}.", "Protokół sporządził{a} {X}."],
    "D": ["Pismo od {X} wpłynęło wczoraj.", "Brakuje podpisu {X} na drugiej stronie.",
          "Dane {X} zostały zaktualizowane.", "Nie mogę się dodzwonić do {X}.",
          "Na prośbę {X} przesunięto termin.", "Czekamy na odpowiedź {X}."],
    "C": ["Przekazałem dokumenty {X}.", "Wypłacono {X} zaliczkę.", "Dziękuję {X} za pomoc.",
          "Zlecenie przydzielono {X}.", "Zarząd przyznał {X} premię.",
          "Proszę przesłać {X} kopię faktury.", "Klucze wydano {X} w recepcji."],
    "B": ["Proszę zaprosić {X} na rozmowę.", "Widziałem {X} w biurze.",
          "Upoważniam {X} do odbioru przesyłki.", "Polecam {X} jako wykonawcę.",
          "Wezwano {X} do uzupełnienia braków."],
    "N": ["Rozmawiałem wczoraj z {X}.", "Umowę zawarto z {X}.", "Spotkanie z {X} przesunięto.",
          "Kontakt z {X} jest utrudniony.", "Zgodnie z ustaleniami z {X} termin mija w piątek."],
    "Ms": ["Rozmawialiśmy o {X} na zebraniu.", "Wspomniano o {X} w protokole.",
           "W notatce o {X} brakuje daty.", "Przy {X} siedziała jeszcze jedna osoba."],
}
SZABLONY_WOLACZ = ["{X}, oddzwoń proszę.", "Cześć {X}, masz chwilę?", "Dzień dobry, {X}!",
                   "{X}, przypominam o spotkaniu."]
SZABLONY_ADRES_M = ["Adres zamieszkania: {A}.", "Adres do korespondencji: {A}.",
                    "Lokal położony pod adresem {A}.", "Przesyłkę proszę nadać na adres {A}."]
SZABLONY_ADRES_MS = ["Mieszka na {A}, drugie piętro.", "Spotkanie odbędzie się na {A}.",
                     "Biuro mieści się przy ul. {A}."]
SZABLONY_FIRMA = ["Fakturę wystawiła firma {F}.", "Umowę zawarto ze spółką {F}.",
                  "Zatrudniony w {F} od 2019 roku.", "Wykonawcą prac jest {F}.",
                  "Reklamację skierowano do {F}."]
SZABLONY_LOGIN = ["Login do systemu: {L}.", "Konto użytkownika {L} zostało zablokowane.",
                  "Zaloguj się jako {L} i zmień hasło.", "Nick na forum: {L}."]
SZABLONY_ROLA = ["{R} zobowiązuje się do zapłaty w terminie 14 dni.",
                 "{R} odpowiada za szkody powstałe z jego winy.",
                 "Podpis: {R}.", "{R} potwierdza odbiór lokalu.",
                 "Zgodnie z § 4 {R} ponosi koszty eksploatacji."]
# relacje i zawody bez imienia – gotowe zdania (odmiana się zgadza)
ZDANIA_RELACJA = ["Mój szwagier przyjedzie jutro.", "Wczoraj dzwoniła mama.",
                  "Pozdrów ode mnie teściową.", "Brat mówi, że wszystko w porządku.",
                  "Byłem z sąsiadem na zakupach.", "Kuzynka przyjedzie w sobotę.",
                  "Mąż odbierze dzieci ze szkoły.", "Żona jest w pracy do 18.",
                  "Hydraulik przyjdzie we wtorek.", "Lekarz zalecił odpoczynek.",
                  "Kierowca czekał pod budynkiem.", "Księgowa prosi o faktury do piątku.",
                  "Oddałem samochód szwagrowi.", "Zadzwoń do mamy wieczorem.",
                  "Babcia przesyła pozdrowienia.", "Kolega z pracy pomógł mi z przeprowadzką."]
# rola + osoba w jednym zdaniu – oznaczamy TYLKO osobę
SZABLONY_ROLA_OSOBA = ["{R}, {X}, zobowiązuje się do zapłaty czynszu.",
                       "{R} – {X} – potwierdza odbiór kluczy.",
                       "Jako {r} występuje {X}."]

SZABLONY_ZWYKLE = ["Termin płatności upływa 15 marca.", "Załącznik nr 2 stanowi integralną część umowy.",
                   "Proszę o potwierdzenie otrzymania wiadomości.", "Spotkanie zostało przełożone.",
                   "Dokumenty należy złożyć w sekretariacie.", "Wszystkie kwoty podano w złotych."]


# ---------------------------------------------------------------------------
# 6b. Przykłady negatywne – słowa, które WYGLĄDAJĄ na dane, a nimi nie są.
# Wszystkie etykiety puste. Miasta, stanowiska i sklepy celowo INNE niż w testach.
# ---------------------------------------------------------------------------

# nazwy identyfikatorów (bez wartości obok)
IDENTYFIKATORY = ["PESEL", "NIP", "REGON", "KRS", "nr dowodu", "numer dowodu osobistego",
                  "nr paszportu", "nr rachunku", "IBAN", "numer legitymacji", "NIP-UE"]
SZABLONY_IDENTYFIKATOR = ["Proszę podać {I} w formularzu.", "W polu {I} wpisz dane z dokumentu.",
                          "Brakuje numeru {I} we wniosku.", "{I}: ………………",
                          "Czy {I} jest wymagany przy rejestracji?",
                          "Pole „{I}” zostało puste.", "{I} należy wpisać bez spacji i myślników.",
                          "Bez {I} nie wyślemy deklaracji.", "Rubryka {I} – do uzupełnienia."]

# numery wewnętrzne: telefon wewnętrzny, pokój, piętro, stanowisko
NUMERY_WEWNETRZNE = ["wew. {n}", "wewn. {n}", "pokój {p}", "pok. {p}", "sala {n}",
                     "piętro {k}", "okienko nr {k}", "stanowisko {k}", "budynek {B}",
                     # grafik pracy: zmiany i dyżury z numerem to nie adres
                     "zmiana {k}", "zmiana poranna {k}", "II zmiana, linia {k}", "dyżur nr {k}",
                     "dyżur weekendowy {k}", "grafik zmian {k}"]
SZABLONY_WEWNETRZNY = ["Proszę dzwonić na {W}.", "Sekretariat: {W}.", "Spotkanie w sali obok, {W}.",
                       "Formularze wydaje {W}.", "Kontakt w sprawie faktur – {W}.",
                       "Szkolenie odbędzie się w {W}.", "Pytania kieruj na {W} po 10:00."]

# miasta: (mianownik, dopełniacz, miejscownik) – samo miasto w zdaniu to NIE adres
MIASTA_NEG = [("Kielce", "Kielc", "Kielcach"), ("Białystok", "Białegostoku", "Białymstoku"),
              ("Gniezno", "Gniezna", "Gnieźnie"), ("Toruń", "Torunia", "Toruniu"),
              ("Tarnów", "Tarnowa", "Tarnowie"), ("Koszalin", "Koszalina", "Koszalinie"),
              ("Suwałki", "Suwałk", "Suwałkach"), ("Legnica", "Legnicy", "Legnicy"),
              ("Sandomierz", "Sandomierza", "Sandomierzu"), ("Zakopane", "Zakopanego", "Zakopanem"),
              ("Bydgoszcz", "Bydgoszczy", "Bydgoszczy"), ("Łomża", "Łomży", "Łomży"),
              ("Przemyśl", "Przemyśla", "Przemyślu"), ("Kalisz", "Kalisza", "Kaliszu")]
SZABLONY_MIASTO = ["Pracuję w {Ms}.", "Jest lekarzem z {D}.", "Wróciłem wczoraj z {D}.",
                   "Konferencja w {Ms} była udana.", "{M} to piękne miasto.",
                   "Pociąg do {D} ma opóźnienie.", "Mamy oddział w {Ms} i dwa w okolicy.",
                   "Firma przeniosła produkcję do {D}.", "Studiowała w {Ms}.",
                   "Kierowca jedzie przez {M} do granicy."]

# stanowiska i tytuły – po polsku i obco
STANOWISKA = ["lekarz kierujący", "lekarz prowadzący", "kierownik działu", "starszy specjalista",
              "dyrektor handlowy", "pielęgniarka oddziałowa", "radca prawny", "główny księgowy",
              "Oberarzt", "Abteilungsleiter", "Head of Sales", "Senior Developer",
              "Account Manager", "Chief Financial Officer", "Geschäftsführer", "Team Lead"]
SZABLONY_STANOWISKO = ["Skierowanie wystawia {S}.", "Stanowisko: {S}.", "Szukamy osoby na stanowisko {S}.",
                       "Podpisuje {S} lub jego zastępca.", "Funkcja: {S}, pełny etat.",
                       "Decyzję zatwierdza {S}.", "Rekrutujemy: {S} (umowa o pracę)."]

# sklepy i marki pisane małą literą w mowie potocznej (mianownik, dopełniacz, miejscownik)
SKLEPY = [("biedronka", "biedronki", "biedronce"), ("lidl", "lidla", "lidlu"),
          ("stokrotka", "stokrotki", "stokrotce"), ("rossmann", "rossmanna", "rossmannie"),
          ("orlen", "orlenu", "orlenie"), ("netto", "netto", "netto"),
          ("castorama", "castoramy", "castoramie"), ("dino", "dino", "dino")]
SZABLONY_SKLEP = ["spotkajmy się koło {D}.", "kupiłem chleb w {Ms}.", "idę do {D} po mleko.",
                  "byłam dziś w {Ms}, tłumy.", "zatankuj na {Ms} przy wyjeździe.",
                  "czekam pod {D} od dziesięciu minut."]

# ---------------------------------------------------------------------------
# 6c. Tytuł/stanowisko + osoba – przykłady POZYTYWNE (runda 3).
# Same negatywy z 6b nauczyły model, że „po tytule nie ma osoby”. Tu pokazujemy,
# że tytuł i stanowisko NIE są encją, ale imię i nazwisko obok nich – tak.
# ---------------------------------------------------------------------------

# skróty tytułów (nie odmieniają się)
TYTULY = ["dr", "dr n. med.", "dr hab.", "prof.", "lek. med.", "lek. dent.", "mgr", "mgr inż.",
          "inż.", "adw.", "r.pr.", "Dr.", "Prof. Dr.", "Dipl.-Ing."]
# {T} = tytuł, {X} = osoba w danym przypadku (zawsze nazwisko, zwykle z imieniem)
SZABLONY_TYTUL_OSOBA = {
    "M": ["Zaświadczenie wystawił{a} {T} {X}.", "Opinię sporządził{a} {T} {X}.",
          "Podpis: {T} {X}", "Konsultował{a} {T} {X}.", "Prelegent: {T} {X}"],
    "D": ["Skierowanie od {T} {X}.", "Na wniosek {T} {X} wydłużono termin.",
          "Pacjent pozostaje pod opieką {T} {X}."],
    "N": ["Wizyta kontrolna z {T} {X} za miesiąc.", "Rozmawiałem z {T} {X} o wynikach."],
}
SZABLONY_STANOWISKO_OSOBA = ["{S}: {X}", "{S} – {X}", "{X}, {S}", "{X} ({s})",
                             "Kontakt: {X}, {s}."]


# ---------------------------------------------------------------------------
# 7. Składanie przykładów
# ---------------------------------------------------------------------------

def zdanie_negatywne(los):
    """Zdanie z czymś, co przypomina daną osobową, ale nią nie jest. Encji brak."""
    rodzaj = los.random()
    if rodzaj < 0.25:
        return los.choice(SZABLONY_IDENTYFIKATOR).replace("{I}", los.choice(IDENTYFIKATORY))
    if rodzaj < 0.45:
        w = (los.choice(NUMERY_WEWNETRZNE)
             .replace("{n}", str(los.randint(100, 499)))
             .replace("{p}", f"{los.randint(0, 6)}.{los.randint(1, 40):02d}")
             .replace("{k}", str(los.randint(1, 12)))
             .replace("{B}", los.choice("ABCDE")))
        return los.choice(SZABLONY_WEWNETRZNY).replace("{W}", w)
    if rodzaj < 0.70:
        m, d, ms = los.choice(MIASTA_NEG)
        return los.choice(SZABLONY_MIASTO).replace("{M}", m).replace("{D}", d).replace("{Ms}", ms)
    if rodzaj < 0.90:
        return los.choice(SZABLONY_STANOWISKO).replace("{S}", los.choice(STANOWISKA))
    _, d, ms = los.choice(SKLEPY)
    return los.choice(SZABLONY_SKLEP).replace("{D}", d).replace("{Ms}", ms)


def zdanie_tytul_osoba(los, czat):
    """Tytuł albo stanowisko obok osoby – oznaczamy TYLKO imię i nazwisko."""
    osoba = losuj_osobe(los)
    if los.random() < 0.6:
        przypadek = los.choice(list(SZABLONY_TYTUL_OSOBA))
        imie, nazw = osoba["imie"][przypadek], osoba["nazwisko"][przypadek]
        x = f"{imie} {nazw}" if los.random() < 0.8 else nazw
        tekst = los.choice(SZABLONY_TYTUL_OSOBA[przypadek]).replace("{T}", los.choice(TYTULY))
    else:
        x = f"{osoba['imie']['M']} {osoba['nazwisko']['M']}"
        s = los.choice(STANOWISKA)
        tekst = los.choice(SZABLONY_STANOWISKO_OSOBA).replace("{S}", s[0].upper() + s[1:]).replace("{s}", s)
    if czat:
        x = x.lower()                         # czat: imiona małymi literami
    a = "" if osoba["plec"] == "m" else "a"
    return tekst.replace("{X}", x).replace("{a}", a), {"person": [x]}


def zdanie_osoba(los, czat):
    osoba = losuj_osobe(los)
    if los.random() < 0.08:
        tekst = los.choice(SZABLONY_WOLACZ)
        x = osoba["imie"]["W"]
    else:
        przypadek = los.choice(list(SZABLONY_OSOBA))
        tekst = los.choice(SZABLONY_OSOBA[przypadek])
        x = zapis_osoby(osoba, przypadek, los)
    if czat:
        x = x.lower()                         # czat: imiona małymi literami
    a = "" if osoba["plec"] == "m" else "a"
    return tekst.replace("{X}", x).replace("{a}", a), {"person": [x]}, osoba


def zdanie(los, czat):
    """Jedno losowe zdanie + jego encje."""
    if los.random() < 0.15:                   # przykład negatywny (ok. 15% zdań)
        return zdanie_negatywne(los), {}
    if los.random() < 0.10:                   # tytuł/stanowisko + osoba (ok. 8,5% zdań)
        return zdanie_tytul_osoba(los, czat)
    rodzaj = los.random()
    if rodzaj < 0.45:
        tekst, enc, osoba = zdanie_osoba(los, czat)
        return tekst, enc
    if rodzaj < 0.60:
        if los.random() < 0.6:
            a = losuj_adres(los, "M")
            return los.choice(SZABLONY_ADRES_M).replace("{A}", a), {"address": [a]}
        a = losuj_adres(los, "Ms")
        szablon = los.choice(SZABLONY_ADRES_MS)
        if szablon.endswith("ul. {A}.") and a.startswith("ul. "):
            a = a[4:]
        return szablon.replace("{A}", a), {"address": [a]}
    if rodzaj < 0.68:
        f = los.choice(FIRMY)
        return los.choice(SZABLONY_FIRMA).replace("{F}", f), {"company": [f]}
    if rodzaj < 0.73:
        l = losuj_login(losuj_osobe(los), los)
        return los.choice(SZABLONY_LOGIN).replace("{L}", l), {"username": [l]}
    if rodzaj < 0.83:
        r = los.choice(ROLE)
        return los.choice(SZABLONY_ROLA).replace("{R}", r), {}
    if rodzaj < 0.91:
        return los.choice(ZDANIA_RELACJA), {}
    if rodzaj < 0.96:
        osoba = losuj_osobe(los)
        x = zapis_osoby(osoba, "M", los)
        r = los.choice(ROLE)
        szablon = los.choice(SZABLONY_ROLA_OSOBA)
        return szablon.replace("{R}", r).replace("{r}", r.lower()).replace("{X}", x), {"person": [x]}
    return los.choice(SZABLONY_ZWYKLE), {}


def przyklad(los):
    """Krótki tekst z 1–4 zdań. Wszystkie 4 etykiety zawsze obecne (pusta lista = „tu tego nie ma”)."""
    czat = los.random() < 0.2
    zdania, encje = [], {"person": [], "company": [], "address": [], "username": []}
    for _ in range(los.randint(1, 4)):
        tekst, enc = zdanie(los, czat)
        zdania.append(tekst)
        for etykieta, wartosci in enc.items():
            for w in wartosci:
                if w not in encje[etykieta]:
                    encje[etykieta].append(w)
    tekst = " ".join(zdania)
    if czat:
        tekst = tekst[0].lower() + tekst[1:]
    return {"input": tekst, "output": {"entities": encje}}


def sprawdz(rekord):
    """Każda encja musi być DOKŁADNYM fragmentem tekstu (wymóg trenera GLiNER2)."""
    for etykieta, wartosci in rekord["output"]["entities"].items():
        for w in wartosci:
            if w not in rekord["input"]:
                raise ValueError(f"„{w}” ({etykieta}) nie występuje w: {rekord['input']}")


def zapisz(rekordy, plik):
    plik.parent.mkdir(parents=True, exist_ok=True)
    with plik.open("w", encoding="utf-8") as f:
        for r in rekordy:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    zakazane = wyczysc_listy()
    print(f"Wykluczono słowa z testów ({len(zakazane)} słów). Zostało: "
          f"{len(IMIONA_M)} imion męskich, {len(IMIONA_K)} żeńskich, "
          f"{len(RDZENIE_SKI) + len(NAZWISKA_RZECZ) + len(NAZWISKA_KO)} nazwisk, "
          f"{len(ULICE) + len(ULICE_NAZWISKA)} ulic, {len(MIASTA_NEG)} miast "
          f"i {len(SKLEPY)} sklepów do przykładów negatywnych.")

    los = random.Random(ZIARNO)
    rekordy, widziane = [], set()
    while len(rekordy) < ILE_TRENING + ILE_WALIDACJA:
        r = przyklad(los)
        if r["input"] in widziane:          # bez powtórek
            continue
        sprawdz(r)
        widziane.add(r["input"])
        rekordy.append(r)

    trening, walidacja = rekordy[:ILE_TRENING], rekordy[ILE_TRENING:]
    zapisz(trening, WYJSCIE / "gliner_train.jsonl")
    zapisz(walidacja, WYJSCIE / "gliner_val.jsonl")

    ile = {e: sum(len(r["output"]["entities"][e]) for r in trening)
           for e in ("person", "company", "address", "username")}
    puste = sum(1 for r in trening if not any(r["output"]["entities"].values()))
    print(f"Zapisano {len(trening)} przykładów treningowych i {len(walidacja)} walidacyjnych.")
    print(f"Encje w treningu: {ile}. Przykłady bez żadnej encji: {puste}.")
    print("\nPrzykłady:")
    for r in trening[:5]:
        print(" ", json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
