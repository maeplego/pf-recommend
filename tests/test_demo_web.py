from pathlib import Path


def test_demo_web_does_not_assign_innerhtml():
    """Model names come from training artifacts; keep them out of HTML parse."""
    source = Path("apps/demo-web/public/app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in source
    assert "textContent" in source
