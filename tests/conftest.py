# Dodaje katalog główny projektu do sys.path, żeby `import engine` działał
# niezależnie od tego, skąd uruchomisz pytest.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
