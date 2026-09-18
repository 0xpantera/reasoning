from pathlib import Path

import pytest

import reasoning.artifacts.qwen3 as qwen3_artifacts


def test_download_qwen3_tokenizer_only(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_download(primary, out_dir, backup):
        calls.append((primary, out_dir, backup))
        return Path(out_dir) / "tokenizer-base.json"

    monkeypatch.setattr(qwen3_artifacts, "download_file", fake_download)
    result = qwen3_artifacts.download_qwen3_small(
        kind="base", tokenizer_only=True, out_dir=tmp_path
    )

    assert result == {"tokenizer": tmp_path / "tokenizer-base.json"}
    assert len(calls) == 1
    assert calls[0][0].endswith("/tokenizer-base.json")
    assert calls[0][2].endswith("/tokenizer-base.json")


def test_download_qwen3_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="base.*reasoning"):
        qwen3_artifacts.download_qwen3_small(kind="unknown")
