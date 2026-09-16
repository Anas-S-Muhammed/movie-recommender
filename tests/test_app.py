from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_app_loads_and_generates_recommendations() -> None:
    app = AppTest.from_file("app.py", default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "🎬 ReelMatch"
    assert [metric.value for metric in app.metric] == ["100,000", "1,682", "943"]

    app.multiselect[0].set_value([50])
    app.button[0].click().run(timeout=30)

    assert not app.exception
    assert len(app.dataframe) == 1
    assert "Recommended for you" in [heading.value for heading in app.subheader]
