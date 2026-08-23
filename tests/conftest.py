import os
import socket
import sys
import types
from pathlib import Path

import json
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True, scope="session")
def _block_network():
    """Suite-level guard: fail fast if any test opens a network socket.

    The CPU suite must stay offline (no model downloads, no remote calls).
    Set ``ZAP_IT_TESTS_ALLOW_SOCKETS=1`` to opt out while debugging.
    """
    if os.environ.get("ZAP_IT_TESTS_ALLOW_SOCKETS") == "1":
        yield
        return

    real_socket = socket.socket

    class _BlockedSocket(real_socket):
        def __init__(self, *args, **kwargs):
            raise AssertionError("network sockets are disabled during the ZAP-IT CPU test suite")

    socket.socket = _BlockedSocket  # type: ignore[misc]
    try:
        yield
    finally:
        socket.socket = real_socket


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


if "torch" not in sys.modules:
    torch_mod = types.ModuleType("torch")

    def _device(name="cpu"):
        return f"torch-device:{name}"

    torch_mod.device = _device
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    torch_mod.no_grad = lambda: _NoGrad()
    torch_mod.from_numpy = lambda arr: np.array(arr, copy=False)
    torch_mod.stack = lambda seq, dim=0: np.stack(list(seq), axis=dim)

    def _tensor(data, dtype=None):
        return np.array(data, dtype=dtype)

    torch_mod.tensor = _tensor
    torch_mod.matmul = lambda a, b: np.matmul(a, b)
    torch_mod.float32 = np.float32
    torch_mod.int64 = np.int64
    sys.modules["torch"] = torch_mod

if "PIL" not in sys.modules:
    try:
        # Prefer the REAL Pillow when installed so image encoding/decoding is
        # genuinely exercised (mirrors the PyYAML policy below and lets the
        # objective 001-a identity-PNG tests inspect decoded pixels); only
        # stub when absent.
        import PIL  # noqa: F401
        import PIL.Image  # noqa: F401
        import PIL.ImageOps  # noqa: F401
    except ImportError:
        pil_mod = types.ModuleType("PIL")
        image_mod = types.ModuleType("PIL.Image")
        image_ops_mod = types.ModuleType("PIL.ImageOps")

        class _FakeImage:
            def __init__(self, array):
                self._array = np.array(array, copy=True)
                if self._array.ndim == 2:
                    self._array = np.expand_dims(self._array, -1)
                self.size = (self._array.shape[1], self._array.shape[0])

            def convert(self, mode):
                return self

            def resize(self, size, resample=None):
                w, h = size
                new_arr = np.zeros((h, w, self._array.shape[2]), dtype=self._array.dtype)
                return _FakeImage(new_arr)

            def save(self, path, *_args, **_kwargs):
                payload = getattr(self, "_stub_bytes", b"fake-image")
                if hasattr(path, "write"):
                    path.write(payload)
                else:
                    Path(path).write_bytes(payload)

            def __array__(self, dtype=None, copy=None):
                return np.array(self._array, dtype=dtype, copy=copy)

        def _fromarray(array, mode=None):
            image = _FakeImage(array)
            if mode is not None:
                image._stub_bytes = b"fake-image"
            return image

        def _open(path):
            return _FakeImage(np.zeros((1, 1, 3), dtype=np.uint8))

        def _new(mode, size, color=(0, 0, 0)):
            w, h = size
            arr = np.zeros((h, w, 3), dtype=np.uint8)
            return _FakeImage(arr)

        class _Resampling:
            LANCZOS = 0

        image_mod.fromarray = _fromarray
        image_mod.open = _open
        image_mod.new = _new
        image_mod.Image = _FakeImage
        image_mod.Resampling = _Resampling

        image_ops_mod.exif_transpose = lambda img: img

        pil_mod.Image = image_mod
        pil_mod.ImageOps = image_ops_mod

        sys.modules["PIL"] = pil_mod
        sys.modules["PIL.Image"] = image_mod
        sys.modules["PIL.ImageOps"] = image_ops_mod

if "detectron2" not in sys.modules:
    detectron2_mod = types.ModuleType("detectron2")
    data_mod = types.ModuleType("detectron2.data")

    class _Metadata:
        def __init__(self):
            self.thing_classes = []

    data_mod.Metadata = _Metadata

    structures_mod = types.ModuleType("detectron2.structures")

    class _BitMasks:
        def __init__(self, tensor):
            self.tensor = np.array(tensor)

    class _Instances:
        def __init__(self, shape):
            self._shape = shape
            self.pred_masks = None
            self.pred_classes = None

    structures_mod.BitMasks = _BitMasks
    structures_mod.Instances = _Instances

    utils_mod = types.ModuleType("detectron2.utils")
    visualizer_mod = types.ModuleType("detectron2.utils.visualizer")

    class _Visualizer:
        def __init__(self, image, metadata=None, instance_mode=None):
            self.image = np.array(image)
            self.metadata = metadata
            self.instance_mode = instance_mode

        def draw_instance_predictions(self, instances):
            if instances.pred_masks is None:
                result = np.array(self.image)
            else:
                overlay = np.array(self.image)
                overlay[:] = np.clip(overlay + 1, 0, 255)
                result = overlay

            class _Result:
                def __init__(self, img):
                    self._img = img

                def get_image(self):
                    return np.array(self._img)

            return _Result(result)

    visualizer_mod.Visualizer = _Visualizer
    visualizer_mod.ColorMode = types.SimpleNamespace(SEGMENTATION="seg", IMAGE="image")

    sys.modules["detectron2"] = detectron2_mod
    sys.modules["detectron2.data"] = data_mod
    sys.modules["detectron2.structures"] = structures_mod
    sys.modules["detectron2.utils"] = utils_mod
    sys.modules["detectron2.utils.visualizer"] = visualizer_mod

if "huggingface_hub" not in sys.modules:
    hf_mod = types.ModuleType("huggingface_hub")
    hf_state = {}

    def _login(token=None, add_to_git_credential=False):
        hf_state["last_login"] = (token, add_to_git_credential)

    def _snapshot_download(*args, **kwargs):
        hf_state["last_download"] = {"args": args, "kwargs": kwargs}
        return "/tmp/fake-download"

    hf_mod.login = _login
    hf_mod.snapshot_download = _snapshot_download
    hf_mod._state = hf_state
    sys.modules["huggingface_hub"] = hf_mod

if "cv2" not in sys.modules:
    cv2_mod = types.ModuleType("cv2")
    cv2_mod.Canny = lambda image, threshold1, threshold2, apertureSize=3: np.zeros_like(
        image, dtype=np.uint8
    )
    cv2_mod.HoughLinesP = lambda *args, **kwargs: None
    cv2_mod.imwrite = lambda path, data: True
    cv2_mod.line = lambda *args, **kwargs: None
    cv2_mod.circle = lambda *args, **kwargs: None
    cv2_mod.LINE_AA = 0
    sys.modules["cv2"] = cv2_mod

if "yaml" not in sys.modules:
    try:
        # Prefer the REAL PyYAML when installed so YAML behavior is genuinely
        # exercised (see tests/test_real_yaml_config.py); only stub when absent.
        import yaml  # noqa: F401
    except ImportError:
        yaml_mod = types.ModuleType("yaml")

        def _safe_load(stream):
            text = stream if isinstance(stream, str) else stream.read()
            text = text.strip()
            if not text:
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                data: dict[str, object] = {}
                stack = [data]
                indents = [0]
                for raw_line in text.splitlines():
                    if not raw_line.strip() or raw_line.strip().startswith("#"):
                        continue
                    indent = len(raw_line) - len(raw_line.lstrip())
                    line = raw_line.strip()
                    while indent < indents[-1]:
                        stack.pop()
                        indents.pop()
                    if line.endswith(":"):
                        key = line[:-1].strip()
                        new_dict: dict[str, object] = {}
                        stack[-1][key] = new_dict
                        stack.append(new_dict)
                        indents.append(indent + 2)
                    else:
                        if ":" in line:
                            key, value = [part.strip() for part in line.split(":", 1)]
                            target = stack[-1]
                            if value.startswith("[") and value.endswith("]"):
                                items = [
                                    item.strip().strip('"')
                                    for item in value[1:-1].split(",")
                                    if item.strip()
                                ]
                                target[key] = items
                            elif value == "{}":
                                target[key] = {}
                            else:
                                lower = value.lower()
                                if lower in {"true", "false"}:
                                    target[key] = lower == "true"
                                else:
                                    try:
                                        target[key] = int(value)
                                    except ValueError:
                                        try:
                                            target[key] = float(value)
                                        except ValueError:
                                            target[key] = value.strip('"')
                return data

        def _safe_dump(data, stream=None, **_kwargs):
            text = json.dumps(data)
            if stream is None:
                return text
            stream.write(text)

        yaml_mod.safe_load = _safe_load
        yaml_mod.safe_dump = _safe_dump
        sys.modules["yaml"] = yaml_mod
