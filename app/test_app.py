"""Smoke test: the dashboard renders every tab without raising."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).with_name("streamlit_app.py"))


def test_app_renders_without_exceptions():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception

    at.button[0].click().run()
    assert not at.exception

    at.radio[0].set_value("Out-of-sample").run()
    assert not at.exception
