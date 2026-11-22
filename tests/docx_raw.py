"""Budowanie plików .docx z surowego XML – do testów odczytu Worda.

python-docx nie umie utworzyć pól tekstowych, przypisów ani kontrolek treści,
a prawdziwe pliki z Worda je mają. Tu składamy minimalny pakiet ręcznie,
z takimi samymi strukturami, jakie zapisuje Word (np. pole tekstowe jako
mc:AlternateContent z kopią zapasową VML).
"""

import io
import zipfile

NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
      'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
      'xmlns:v="urn:schemas-microsoft-com:vml" '
      'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" mc:Ignorable="w14"')

REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def p(text: str) -> str:
    return f'<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'


def textbox(text: str) -> str:
    """Akapit z polem tekstowym – jak Word 2010+: wersja DrawingML + zapasowa VML."""
    content = f"<w:txbxContent>{p(text)}</w:txbxContent>"
    return (
        "<w:p><w:r><mc:AlternateContent>"
        '<mc:Choice Requires="wps"><w:drawing><wp:anchor><a:graphic>'
        '<a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape">'
        f"<wps:wsp><wps:txbx>{content}</wps:txbx></wps:wsp>"
        "</a:graphicData></a:graphic></wp:anchor></w:drawing></mc:Choice>"
        f"<mc:Fallback><w:pict><v:shape><v:textbox>{content}</v:textbox></v:shape></w:pict></mc:Fallback>"
        "</mc:AlternateContent></w:r></w:p>"
    )


def block_sdt(inner: str) -> str:
    """Kontrolka treści na poziomie akapitów (np. pole formularza „Najemca”)."""
    return f"<w:sdt><w:sdtPr/><w:sdtContent>{inner}</w:sdtContent></w:sdt>"


def inline_sdt_paragraph(before: str, value: str, after: str) -> str:
    """Kontrolka treści w środku zdania: „Najemca: [Imię i nazwisko], PESEL …”."""
    return (f'<w:p><w:r><w:t xml:space="preserve">{before}</w:t></w:r>'
            f'<w:sdt><w:sdtPr/><w:sdtContent><w:r><w:t>{value}</w:t></w:r></w:sdtContent></w:sdt>'
            f'<w:r><w:t xml:space="preserve">{after}</w:t></w:r></w:p>')


def footnote_ref_paragraph(text: str, note_id: int = 1) -> str:
    return (f'<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r>'
            f'<w:r><w:footnoteReference w:id="{note_id}"/></w:r></w:p>')


def table(rows: list[list[str]]) -> str:
    """Tabela; komórka może być gotowym XML (zaczyna się od „<”) – np. inna tabela."""
    xml = "<w:tbl>"
    for row in rows:
        xml += "<w:tr>"
        for cell in row:
            body = cell if cell.startswith("<") else p(cell)
            # Komórka musi kończyć się akapitem (wymóg formatu).
            xml += f"<w:tc>{body}{'' if body.endswith('</w:p>') else '<w:p/>'}</w:tc>"
        xml += "</w:tr>"
    return xml + "</w:tbl>"


def build(body: str, header: str | None = None, footer: str | None = None,
          footnotes: list[str] | None = None, endnotes: list[str] | None = None) -> bytes:
    """Składa pakiet .docx z podanych fragmentów XML."""
    rels, overrides, sect = [], [], ""
    files = {}
    if header is not None:
        files["word/header1.xml"] = f"<w:hdr {NS}>{header}</w:hdr>"
        rels.append(f'<Relationship Id="rIdH" Type="{REL}/header" Target="header1.xml"/>')
        overrides.append('<Override PartName="/word/header1.xml" ContentType="application/'
                         'vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>')
        sect += '<w:headerReference w:type="default" r:id="rIdH"/>'
    if footer is not None:
        files["word/footer1.xml"] = f"<w:ftr {NS}>{footer}</w:ftr>"
        rels.append(f'<Relationship Id="rIdF" Type="{REL}/footer" Target="footer1.xml"/>')
        overrides.append('<Override PartName="/word/footer1.xml" ContentType="application/'
                         'vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>')
        sect += '<w:footerReference w:type="default" r:id="rIdF"/>'
    for kind, notes in (("footnote", footnotes), ("endnote", endnotes)):
        if notes is None:
            continue
        # Word zawsze zapisuje dwa przypisy-separatory (id -1 i 0) – nie są treścią.
        xml = (f'<w:{kind} w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:{kind}>'
               f'<w:{kind} w:type="continuationSeparator" w:id="0"><w:p><w:r>'
               f"<w:continuationSeparator/></w:r></w:p></w:{kind}>")
        xml += "".join(f'<w:{kind} w:id="{i}">{p(text)}</w:{kind}>' for i, text in enumerate(notes, 1))
        files[f"word/{kind}s.xml"] = f"<w:{kind}s {NS}>{xml}</w:{kind}s>"
        rels.append(f'<Relationship Id="rId{kind}" Type="{REL}/{kind}s" Target="{kind}s.xml"/>')
        overrides.append(f'<Override PartName="/word/{kind}s.xml" ContentType="application/'
                         f'vnd.openxmlformats-officedocument.wordprocessingml.{kind}s+xml"/>')
    files["word/document.xml"] = (f"<w:document {NS}><w:body>{body}"
                                  f"<w:sectPr>{sect}</w:sectPr></w:body></w:document>")
    files["word/_rels/document.xml.rels"] = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels) + "</Relationships>")
    files["_rels/.rels"] = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{REL}/officeDocument" Target="word/document.xml"/>'
        "</Relationships>")
    files["[Content_Types].xml"] = (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        + "".join(overrides) + "</Types>")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + content)
    return buffer.getvalue()


def sample() -> bytes:
    """Wszystkie trudne miejsca naraz – każde z inną fikcyjną daną."""
    nested = table([["Pełnomocnik", "Bogumiła Czajka"]])
    body = (
        p("UMOWA NAJMU LOKALU")
        + block_sdt(p("Wynajmujący: Kazimierz Wilk"))
        + inline_sdt_paragraph("Najemca: ", "Oliwia Szpak", ", PESEL 85042311448")
        + footnote_ref_paragraph("Umowę zawarto na podstawie pełnomocnictwa.")
        + textbox("Pieczątka: Biuro Rachunkowe Dąb, tel. 600 700 800")
        + table([["Strona", "Dane"], ["Poręczyciel", nested]])
        + p("Koniec umowy.")
    )
    return build(
        body,
        header=p("Kancelaria Notarialna Jan Bocian, ul. Ogrodowa 5, Kalisz"),
        footer=p("Sporządził: Witold Kret, tel. 512 300 400"),
        footnotes=["Pełnomocnictwo dla Haliny Sowy z 3 marca 2025 r."],
        endnotes=["Odpis wydano Teofilowi Gilowi."],
    )
