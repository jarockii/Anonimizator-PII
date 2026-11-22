"""Odczyt treści dokumentu Word (.docx) – CAŁEJ, nie tylko akapitów ciała.

python-docx daje wygodne `document.paragraphs` i `table.cell().text`, ale
pomija sporo miejsc, w których w prawdziwych pismach stoją dane:

- nagłówki i stopki (kancelaria, „Sporządził: …”, numer sprawy),
- przypisy dolne i końcowe („Pełnomocnictwo dla …”),
- pola tekstowe (pieczątka wstawiona jako kształt),
- kontrolki treści (formularze: „Najemca: [pole]”) – w akapicie i jako blok,
- tabele zagnieżdżone w komórkach.

Dlatego czytamy XML sami, w kolejności dokumentu. Pole tekstowe Word
zapisuje DWA razy (DrawingML w mc:Choice + kopia VML w mc:Fallback dla
starych wersji) – kopię pomijamy, inaczej każdy tekst byłby podwojony.

Kolejność w wyniku: nagłówki, treść, przypisy dolne, przypisy końcowe, stopki.
Tekst z obrazów (OCR) trafia zaraz po akapicie, w którym obraz stoi.
"""

import io
from dataclasses import dataclass

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import parse_xml
from docx.oxml.ns import qn

W_P, W_TBL, W_TR, W_TC = qn("w:p"), qn("w:tbl"), qn("w:tr"), qn("w:tc")
W_T, W_TAB, W_BR, W_CR = qn("w:t"), qn("w:tab"), qn("w:br"), qn("w:cr")
W_NO_BREAK_HYPHEN = qn("w:noBreakHyphen")
W_TXBX = qn("w:txbxContent")
MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
# Opakowania, w których mogą stać akapity, wiersze tabel albo przebiegi tekstu.
_WRAPPERS = {qn("w:sdt"), qn("w:sdtContent"), qn("w:customXml"), qn("w:smartTag"),
             qn("w:ins"), qn("w:moveTo"), qn("w:hyperlink"), qn("w:fldSimple"),
             "{http://schemas.openxmlformats.org/markup-compatibility/2006}AlternateContent",
             "{http://schemas.openxmlformats.org/markup-compatibility/2006}Choice"}
# Obraz: <a:blip r:embed> (DrawingML) albo <v:imagedata r:id> (VML).
_IMAGE_REFS = ((qn("a:blip"), qn("r:embed")),
               ("{urn:schemas-microsoft-com:vml}imagedata", qn("r:id")))
# Przypisy-separatory (kreska nad przypisami) – nie są treścią.
_NOTE_SEPARATORS = {"separator", "continuationSeparator", "continuationNotice"}


@dataclass
class Story:
    """Jedna „opowieść” dokumentu: treść, nagłówek, stopka, przypisy…"""
    kind: str        # "naglowek" | "tresc" | "przypisy" | "przypisy_koncowe" | "stopka"
    element: object  # korzeń XML z akapitami i tabelami
    part: object     # część pakietu – rozwiązuje identyfikatory obrazów (r:embed)


def _walk(element, skip_textboxes: bool):
    """Elementy w kolejności dokumentu, bez kopii zapasowych (mc:Fallback)
    i – opcjonalnie – bez wnętrza pól tekstowych."""
    for child in element:
        if child.tag == MC_FALLBACK or (skip_textboxes and child.tag == W_TXBX):
            continue
        yield child
        yield from _walk(child, skip_textboxes)


def paragraph_text(paragraph) -> str:
    """Widoczny tekst akapitu: także w kontrolkach treści, hiperłączach i polach.
    Bez tekstu usuniętego w śledzeniu zmian (w:delText) i bez pól tekstowych
    (te czytamy osobno, jako własne akapity)."""
    parts = []
    for node in _walk(paragraph, skip_textboxes=True):
        if node.tag == W_T:
            parts.append(node.text or "")
        elif node.tag == W_TAB:
            parts.append("\t")
        elif node.tag in (W_BR, W_CR):
            parts.append("\n")
        elif node.tag == W_NO_BREAK_HYPHEN:
            parts.append("-")
    return "".join(parts)


def image_parts(element, part, skip_textboxes: bool = False) -> list:
    """Obrazy wstawione w danym fragmencie XML, w kolejności, jako części pakietu."""
    found = []
    for node in _walk(element, skip_textboxes):
        for tag, attribute in _IMAGE_REFS:
            if node.tag == tag:
                rel_id = node.get(attribute)
                target = part.related_parts.get(rel_id) if rel_id else None
                if target is not None and hasattr(target, "blob"):
                    found.append(target)
    return found


def _block_children(element):
    """Bezpośrednie akapity/tabele/wiersze/komórki – przez opakowania (kontrolki itp.)."""
    for child in element:
        if child.tag in _WRAPPERS:
            yield from _block_children(child)
        else:
            yield child


def block_lines(element, part, image_texts: dict[str, str]) -> list[str]:
    """Linie tekstu z akapitów i tabel danego kontenera (ciało, komórka, pole tekstowe…)."""
    lines: list[str] = []
    for child in _block_children(element):
        if child.tag == W_P:
            lines.append(paragraph_text(child))
            # Pola tekstowe w akapicie (bez kopii VML) – ich akapity zaraz pod nim.
            for node in _walk(child, skip_textboxes=False):
                if node.tag == W_TXBX and not _inside_textbox(node, child):
                    lines.extend(block_lines(node, part, image_texts))
            for image in image_parts(child, part, skip_textboxes=True):
                text = image_texts.get(str(image.partname), "")
                if text.strip():
                    lines.append(text.strip())
        elif child.tag == W_TBL:
            for row in _block_children(child):
                if row.tag != W_TR:
                    continue
                cells = []
                for cell in _block_children(row):
                    if cell.tag == W_TC:
                        cell_lines = block_lines(cell, part, image_texts)
                        # Wiersz tabeli = jedna linia: złamania wewnątrz komórki
                        # (kilka akapitów, tekst z obrazu) zamieniamy na spacje.
                        cells.append(" ".join(" ".join(line.split("\n")).strip()
                                              for line in cell_lines if line.strip()))
                lines.append(" | ".join(cells))
    return lines


def _inside_textbox(node, paragraph) -> bool:
    """Czy pole tekstowe jest zagnieżdżone w INNYM polu tekstowym tego akapitu
    (wtedy przeczyta je rekurencja tamtego pola, nie my)."""
    parent = node.getparent()
    while parent is not None and parent is not paragraph:
        if parent.tag == W_TXBX:
            return True
        parent = parent.getparent()
    return False


def _related(document, reltype: str) -> list:
    parts = {}
    for rel in document.part.rels.values():
        if rel.reltype == reltype and not rel.is_external:
            parts[str(rel.target_part.partname)] = rel.target_part
    return [parts[name] for name in sorted(parts)]


def _element(part):
    element = getattr(part, "element", None)
    return element if element is not None else parse_xml(part.blob)


def _notes(part, kind: str) -> list[Story]:
    stories = []
    for note in _element(part):
        if note.get(qn("w:type")) not in _NOTE_SEPARATORS:
            stories.append(Story(kind, note, part))
    return stories


def stories(document) -> list[Story]:
    """Wszystkie części z treścią, w kolejności, w jakiej trafią do tekstu."""
    found = [Story("naglowek", _element(part), part) for part in _related(document, RT.HEADER)]
    found.append(Story("tresc", document.element.body, document.part))
    for part in _related(document, RT.FOOTNOTES):
        found += _notes(part, "przypisy")
    for part in _related(document, RT.ENDNOTES):
        found += _notes(part, "przypisy_koncowe")
    found += [Story("stopka", _element(part), part) for part in _related(document, RT.FOOTER)]
    return found


def open_document(data: bytes):
    return Document(io.BytesIO(data))


def document_text(document, image_texts: dict[str, str] | None = None) -> str:
    """Pełny tekst dokumentu (bez pustych linii na brzegach)."""
    image_texts = image_texts or {}
    lines: list[str] = []
    for story in stories(document):
        lines.extend(block_lines(story.element, story.part, image_texts))
    return "\n".join(lines).strip()
