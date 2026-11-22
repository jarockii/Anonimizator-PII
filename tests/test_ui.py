"""Komponenty interfejsu, które da się sprawdzić bez przeglądarki (ui.py)."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ui  # noqa: E402


class FakeProgress:
    def __init__(self):
        self.updates = []

    def progress(self, value, text=""):
        self.updates.append((value, text))


class FakeHint:
    def __init__(self):
        self.captions = []

    def caption(self, text):
        self.captions.append(text)


def test_progress_shows_elapsed_time_and_chunks():
    def work(on_progress):
        on_progress(0, 3)
        time.sleep(0.3)
        on_progress(1, 3)
        time.sleep(0.3)
        on_progress(3, 3)
        return "wynik"

    progress, hint = FakeProgress(), FakeHint()
    assert ui.run_with_progress(work, progress, hint, use_llm=True) == "wynik"
    texts = [text for _, text in progress.updates]
    assert any("fragment 1 z 3" in text for text in texts)
    assert any("fragment 2 z 3" in text for text in texts)
    assert all(" s" in text for text in texts)              # zawsze widać upływający czas
    assert texts[-1].startswith("Gotowe")
    values = [value for value, _ in progress.updates]
    assert values == sorted(values) and values[-1] == 1.0    # pasek tylko rośnie
    assert hint.captions == []                               # szybko – bez podpowiedzi


def test_slow_analysis_shows_hints_once_each():
    def slow_work(on_progress):
        time.sleep(0.9)
        return "ok"

    hint = FakeHint()
    ui.run_with_progress(slow_work, FakeProgress(), hint, use_llm=True, slow_s=0.2, very_slow_s=0.6)
    assert len(hint.captions) == 2
    assert "Wciąż analizuję" in hint.captions[0]
    assert "dłużej niż zwykle" in hint.captions[1]


def test_error_in_work_is_raised_in_caller():
    def broken(on_progress):
        raise RuntimeError("Ollama padła")

    with pytest.raises(RuntimeError, match="Ollama padła"):
        ui.run_with_progress(broken, FakeProgress(), FakeHint(), use_llm=True)


def test_dark_and_light_palettes_have_the_same_tokens():
    assert ui.THEMES["light"].keys() == ui.THEMES["dark"].keys()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_text_colors_meet_wcag_aa(theme):
    def luminance(hex_color):
        channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    def contrast(a, b):
        high, low = sorted([luminance(a), luminance(b)], reverse=True)
        return (high + 0.05) / (low + 0.05)

    palette = ui.THEMES[theme]
    for token in ("text", "text-secondary", "error", "success", "warning", "info", "primary"):
        assert contrast(palette[token], palette["surface"]) >= 4.5, (theme, token)
    assert contrast("#FFFFFF", palette["danger-fill"]) >= 4.5, theme      # przycisk „Wyczyść sejf”
    assert contrast(palette["error"], palette["error-bg"]) >= 4.5, theme
