import pytest

import highway


def test_package_imports() -> None:
    assert highway.__name__ == "highway"


def test_main_prints_greeting(capsys: pytest.CaptureFixture[str]) -> None:
    highway.main()

    assert "highway" in capsys.readouterr().out
