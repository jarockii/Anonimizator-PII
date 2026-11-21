# Dodaje katalog główny projektu do sys.path, żeby `import engine` działał
# niezależnie od tego, skąd uruchomisz pytest.
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import llm_masker  # noqa: E402


@pytest.fixture(autouse=True)
def no_real_ollama(request, monkeypatch):
    """Testy nie mogą po cichu rozmawiać z prawdziwą Ollamą.

    Domyślnie każde zapytanie kończy się błędem połączenia (tak jak przy
    wyłączonej Ollamie). Test, który potrzebuje odpowiedzi modelu, podmienia
    requests.post sam – jego monkeypatch wygrywa. Testy oznaczone „slow”
    mogą używać prawdziwych modeli.
    """
    if request.node.get_closest_marker("slow"):
        return

    def blocked(*args, **kwargs):
        raise requests.ConnectionError("Zapytania do Ollamy są w testach zablokowane")

    monkeypatch.setattr(llm_masker.requests, "post", blocked)
