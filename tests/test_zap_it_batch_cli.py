import runpy
from types import SimpleNamespace


def test_cli_accepts_video_input(monkeypatch, tmp_path):

    video = tmp_path / "movie.mp4"
    video.write_bytes(b"fake")

    config = {"video": {"sam": {}}, "mask_generator": {}, "alpha": 0.5}

    monkeypatch.setattr("src.config.load_config", lambda path, verbosity_level="some": (config, 1))
    monkeypatch.setattr("multiprocessing.set_start_method", lambda *a, **k: None)

    namespace = SimpleNamespace(
        input_image_dir=None,
        input_video=str(video),
        config="config.yaml",
        output_image_dir=None,
        output_video_dir=str(tmp_path / "videos"),
        recursive=False,
        verbose="some",
        randomize=False,
        ngpu=1,
        dryrun=False,
    )

    class DummyParser:
        def __init__(self, *a, **k):
            return

        def add_argument(self, *a, **k):
            return None

        def add_mutually_exclusive_group(self, **k):
            return self

        def parse_args(self):
            return namespace

        def error(self, message):  # pragma: no cover - not exercised here
            raise SystemExit(message)

    monkeypatch.setattr("argparse.ArgumentParser", DummyParser)

    runpy.run_path("zap-it-batch.py", init_globals={"__name__": "__main__"})
