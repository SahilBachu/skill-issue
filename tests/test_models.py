"""Release-zip model downloads: hash pinning, safe extraction, fallback to the base model."""

import hashlib
import io
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from skillissue import models


def _zip(entries: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in entries.items():
            z.writestr(name, text)
    return buf.getvalue()


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture
def served(tmp_path: Path):
    root = tmp_path / "www"
    root.mkdir()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _publish(root: Path, base: str, name: str, data: bytes, monkeypatch, sha: str | None) -> str:
    (root / name).write_bytes(data)
    url = f"{base}/{name}"
    pins = {url: {"sha256": sha, "fallback": "org/base-model"}} if sha else {}
    monkeypatch.setattr(models, "RELEASE_MODELS", pins)
    return url


def test_release_zip_downloads_and_unpacks(env, served, monkeypatch):
    root, base = served
    data = _zip({"gate-v1/config.json": "{}", "gate-v1/skill_issue_gate.json": '{"calibration": {}}'})
    url = _publish(root, base, "gate-v1.zip", data, monkeypatch, hashlib.sha256(data).hexdigest())
    path = models.ensure_model(url)
    assert path.name == "release--gate-v1"
    assert (path / "config.json").is_file() and (path / "skill_issue_gate.json").is_file()
    assert models.is_downloaded(url)
    assert models.ensure_model(url) == path  # cached, no second download


def test_hash_mismatch_is_refused(env, served, monkeypatch):
    root, base = served
    url = _publish(root, base, "bad.zip", _zip({"x/config.json": "{}"}), monkeypatch, "0" * 64)
    with pytest.raises(ValueError, match="hash mismatch"):
        models.ensure_model(url)
    assert not models.is_downloaded(url)


def test_path_traversal_is_refused(env, served, monkeypatch):
    root, base = served
    data = _zip({"../evil.txt": "x"})
    url = _publish(root, base, "evil.zip", data, monkeypatch, hashlib.sha256(data).hexdigest())
    with pytest.raises(ValueError, match="unsafe path"):
        models.ensure_model(url)
    assert not (Path(models.paths.models_dir()).parent / "evil.txt").exists()


def test_unreachable_release_falls_back_to_base(env, monkeypatch):
    url = "http://127.0.0.1:9/gone.zip"
    monkeypatch.setattr(models, "RELEASE_MODELS", {url: {"sha256": "0" * 64, "fallback": "org/base-model"}})
    calls = []

    def fake_ensure(ref, allow_patterns=None, quiet=False):
        calls.append(ref)
        if ref == url:
            raise OSError("connection refused")
        return Path("/models") / ref

    monkeypatch.setattr(models, "ensure_model", fake_ensure)
    path, used = models.ensure_gate_model(url)
    assert used == "org/base-model" and calls == [url, "org/base-model"]


def test_defaults_point_at_pinned_releases():
    for ref in models.DEFAULT_MODELS.values():
        assert models.is_url(ref)
        pin = models.RELEASE_MODELS[ref]
        assert len(pin["sha256"]) == 64 and pin["fallback"]
