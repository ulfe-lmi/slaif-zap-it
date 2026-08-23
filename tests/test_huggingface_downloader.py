"""Tests for huggingface_downloader.py.

History note (documented repair, objective 000-a): these tests were originally
written against the pre-``2eae55a`` downloader interface, which exposed
``resolve_output_dir`` plus ``repos``/``no-symlinks``/``resume`` CLI arguments.
Commit ``2eae55a`` ("Final changes to work with BLIP3 on Vega") intentionally
rewrote the module to a fixed repo list with ``resolve_out`` and a simplified
argument surface, but the tests were not updated, so collection failed. The
current module represents the supported CLI behavior, so the tests are aligned
to the *actual* exports while preserving each test's original intent: token
preference for ``ensure_login``, WORKDIR-based default directory resolution,
and end-to-end wiring of ``main`` into ``huggingface_hub.snapshot_download``.
No network access occurs because the test harness provides a fake
``huggingface_hub`` when the real package is absent.
"""

from types import SimpleNamespace

from huggingface_downloader import DEFAULT_REPOS, ensure_login, main, resolve_out


def test_ensure_login_prefers_argument(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        "huggingface_downloader.login",
        lambda token=None, add_to_git_credential=False: calls.setdefault("token", token),
    )

    ensure_login("hf_test", non_interactive=True)
    assert calls["token"] == "hf_test"


def test_resolve_out_uses_workdir_default(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKDIR", str(tmp_path))
    out = resolve_out(None)
    assert out.parent == tmp_path


def test_resolve_out_creates_explicit_directory(tmp_path):
    target = tmp_path / "models"
    out = resolve_out(str(target))
    assert out == target.resolve()
    assert target.is_dir()


def test_main_invokes_snapshot(monkeypatch, tmp_path):
    records = {"downloads": []}

    class DummyParser:
        def __init__(self, *a, **k):
            self.namespace = SimpleNamespace(
                output=str(tmp_path),
                token="hf_token",
                non_interactive=True,
                with_blip3_processors=False,
            )

        def add_argument(self, *a, **k):
            return None

        def parse_args(self, argv=None):
            return self.namespace

    monkeypatch.setattr("huggingface_downloader.argparse.ArgumentParser", DummyParser)
    monkeypatch.setattr(
        "huggingface_downloader.login",
        lambda token=None, add_to_git_credential=False: records.setdefault("token", token),
    )
    monkeypatch.setattr(
        "huggingface_downloader.snapshot_download",
        lambda **kwargs: records["downloads"].append(kwargs),
    )

    rc = main([])
    assert rc == 0
    assert records["token"] == "hf_token"
    assert [d["repo_id"] for d in records["downloads"]] == list(DEFAULT_REPOS)
    for download, repo in zip(records["downloads"], DEFAULT_REPOS):
        expected_dir = tmp_path / repo.split("/")[-1]
        assert download["local_dir"] == str(expected_dir)
