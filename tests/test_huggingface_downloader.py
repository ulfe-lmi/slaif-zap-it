from types import SimpleNamespace

from huggingface_downloader import ensure_login, main, resolve_output_dir


def test_ensure_login_prefers_argument(monkeypatch):
    calls = {}

    monkeypatch.setattr("huggingface_downloader.login", lambda token=None, add_to_git_credential=False: calls.setdefault("token", token))

    ensure_login("hf_test")
    assert calls["token"] == "hf_test"


def test_resolve_output_dir_uses_default(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKDIR", str(tmp_path))
    out = resolve_output_dir(None)
    assert out.parent == tmp_path


def test_main_invokes_snapshot(monkeypatch, tmp_path):
    records = {"downloads": []}

    class DummyParser:
        def __init__(self, *a, **k):
            self.namespace = SimpleNamespace(
                repos=["org/model"],
                output=str(tmp_path),
                token="hf_token",
                non_interactive=True,
                no_symlinks=True,
                resume=True,
            )

        def add_argument(self, *a, **k):
            return None

        def add_mutually_exclusive_group(self, **k):
            return self

        def parse_args(self, argv=None):
            return self.namespace

    monkeypatch.setattr("huggingface_downloader.argparse.ArgumentParser", DummyParser)
    monkeypatch.setattr("huggingface_downloader.login", lambda token=None, add_to_git_credential=False: records.setdefault("token", token))
    monkeypatch.setattr(
        "huggingface_downloader.snapshot_download",
        lambda **kwargs: records["downloads"].append(kwargs),
    )

    rc = main([])
    assert rc == 0
    assert records["token"] == "hf_token"
    assert records["downloads"][0]["repo_id"] == "org/model"
