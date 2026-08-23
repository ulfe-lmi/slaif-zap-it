"""Strict multipart/form-data parsing for the ``/v1/completions`` contract.

The parser streams the request body through :mod:`python_multipart` with
per-part and total byte caps enforced while bytes arrive (limit+1 pattern),
so oversized uploads are rejected before decoding. Exactly one ``image`` part
and exactly one ``config`` part are required; scalar fields are validated
against the frozen v1 contract (strict integer verbosity, lowercase
``json|zip``, fixed model identifier, ``stream=false`` only). Unknown field
names are rejected; no client-controlled string is ever used as a path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from python_multipart.exceptions import FormParserError
from python_multipart.multipart import MultipartParser, parse_options_header

from .errors import ServiceError
from .settings import ServiceSettings

__all__ = ["ParsedMultipartRequest", "parse_strict_multipart"]

_KNOWN_PARTS = ("image", "config", "verbosity", "response_format", "model", "stream")
_SCALAR_FIELD_CAP_BYTES = 4096
_MAX_PART_COUNT = 16
_VALID_VERBOSITY = frozenset({"0", "1", "2", "3"})
_VALID_RESPONSE_FORMATS = frozenset({"json", "zip"})

_SCALAR_DECODE_CODES = {
    "verbosity": "unsupported_verbosity",
    "response_format": "unsupported_format",
    "model": "unsupported_model",
    "stream": "stream_unsupported",
}


@dataclass(frozen=True)
class ParsedMultipartRequest:
    """Fully validated multipart request payload."""

    image_bytes: bytes
    config_bytes: bytes
    verbosity: int
    response_format: str
    model: Optional[str]
    stream: bool


class _PartCollector:
    """Callback-driven accumulator enforcing cardinality and byte caps."""

    def __init__(self, settings: ServiceSettings) -> None:
        self.settings = settings
        self.parts: Dict[str, bytearray] = {}
        self.part_count = 0
        self.saw_end = False
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._headers: List[Tuple[str, str]] = []
        self._current_name: Optional[str] = None
        self._current_cap = 0

    def _reset_part_state(self) -> None:
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._headers = []
        self._current_name = None
        self._current_cap = 0

    def on_part_begin(self) -> None:
        if self.part_count >= _MAX_PART_COUNT:
            raise ServiceError(
                "multipart request contains too many parts", code="invalid_multipart"
            )
        self.part_count += 1
        self._reset_part_state()

    def on_header_begin(self) -> None:
        return None

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def on_header_end(self) -> None:
        name = self._header_field.decode("latin-1").strip().lower()
        value = self._header_value.decode("latin-1").strip()
        self._headers.append((name, value))
        self._header_field = bytearray()
        self._header_value = bytearray()

    def on_headers_finished(self) -> None:
        disposition = ""
        for header_name, header_value in self._headers:
            if header_name == "content-disposition":
                disposition = header_value
                break
        if not disposition:
            raise ServiceError("multipart part lacks Content-Disposition", code="invalid_multipart")
        raw_name = self._field_name_from_disposition(disposition)
        if not raw_name:
            raise ServiceError("multipart part lacks a field name", code="invalid_multipart")
        if raw_name in self.parts:
            raise ServiceError(
                f"multipart field {raw_name!r} was provided more than once",
                code="duplicate_part",
            )
        if raw_name not in _KNOWN_PARTS:
            raise ServiceError(
                f"multipart field {raw_name!r} is not supported by this endpoint",
                code="unsupported_field",
            )
        if raw_name == "image":
            cap = self.settings.max_image_upload_bytes
        elif raw_name == "config":
            cap = self.settings.max_config_upload_bytes
        else:
            cap = _SCALAR_FIELD_CAP_BYTES
        self._current_name = raw_name
        self._current_cap = cap
        self.parts[raw_name] = bytearray()

    @staticmethod
    def _field_name_from_disposition(disposition: str) -> Optional[str]:
        _, options = parse_options_header(disposition)
        raw_name = _option_value(options, "name")
        if raw_name is None:
            return None
        return raw_name.decode("latin-1")

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_name is None:
            raise ServiceError("multipart data outside of a named part", code="invalid_multipart")
        length = end - start
        buffer = self.parts[self._current_name]
        if len(buffer) + length > self._current_cap:
            raise ServiceError(
                f"multipart field {self._current_name!r} exceeds its upload size limit",
                code="payload_too_large",
            )
        buffer += data[start:end]

    def on_part_end(self) -> None:
        self._reset_part_state()

    def on_end(self) -> None:
        self.saw_end = True


def _option_value(options: dict, wanted: str) -> Optional[bytes]:
    """Fetch an option from ``parse_options_header`` output (bytes/str keys)."""
    for key, value in options.items():
        key_text = key.decode("latin-1") if isinstance(key, bytes) else str(key)
        if key_text.lower() == wanted:
            return value if isinstance(value, bytes) else str(value).encode("latin-1")
    return None


def _boundary_from_content_type(content_type: str) -> bytes:
    media_type, options = parse_options_header(content_type)
    base = (
        media_type.decode("latin-1").lower()
        if isinstance(media_type, bytes)
        else str(media_type).lower()
    )
    if base != "multipart/form-data":
        raise ServiceError("Content-Type must be multipart/form-data", code="invalid_multipart")
    boundary = _option_value(options, "boundary")
    if boundary:
        return boundary
    raise ServiceError("multipart boundary missing from Content-Type", code="invalid_multipart")


def _scalar_text(part: Optional[bytearray], field: str) -> str:
    try:
        return part.decode("utf-8").strip() if part else ""
    except UnicodeDecodeError as exc:
        raise ServiceError(
            f"multipart field {field!r} must be UTF-8 text",
            code=_SCALAR_DECODE_CODES.get(field, "invalid_multipart"),
        ) from exc


def parse_strict_multipart(
    content_type: str,
    body_chunks: Iterable[bytes],
    settings: ServiceSettings,
) -> ParsedMultipartRequest:
    """Parse and validate one multipart request body.

    ``body_chunks`` is an iterable of raw body chunks; the async handler
    materializes its stream so this function stays synchronous and pure.
    """
    boundary = _boundary_from_content_type(content_type)
    collector = _PartCollector(settings)
    callbacks = {
        "on_part_begin": collector.on_part_begin,
        "on_part_data": collector.on_part_data,
        "on_part_end": collector.on_part_end,
        "on_header_begin": collector.on_header_begin,
        "on_header_field": collector.on_header_field,
        "on_header_value": collector.on_header_value,
        "on_header_end": collector.on_header_end,
        "on_headers_finished": collector.on_headers_finished,
        "on_end": collector.on_end,
    }
    parser = MultipartParser(boundary, callbacks)
    total = 0
    try:
        for chunk in body_chunks:
            if not chunk:
                continue
            total += len(chunk)
            if total > settings.max_request_bytes:
                raise ServiceError(
                    "request body exceeds the maximum allowed size",
                    code="payload_too_large",
                )
            parser.write(chunk)
        parser.finalize()
    except FormParserError as exc:
        raise ServiceError("malformed multipart request body", code="invalid_multipart") from exc

    if "image" not in collector.parts:
        raise ServiceError("required multipart field 'image' is missing", code="missing_part")
    if "config" not in collector.parts:
        raise ServiceError("required multipart field 'config' is missing", code="missing_part")

    image_bytes = bytes(collector.parts["image"])
    config_bytes = bytes(collector.parts["config"])

    verbosity_text = _scalar_text(collector.parts.get("verbosity"), "verbosity")
    if verbosity_text and verbosity_text not in _VALID_VERBOSITY:
        raise ServiceError(
            "verbosity must be one of the integers 0, 1, 2 or 3",
            code="unsupported_verbosity",
        )
    verbosity = int(verbosity_text) if verbosity_text else 0

    format_text = _scalar_text(collector.parts.get("response_format"), "response_format")
    if format_text and format_text not in _VALID_RESPONSE_FORMATS:
        raise ServiceError("response_format must be 'json' or 'zip'", code="unsupported_format")
    response_format = format_text or "json"

    model_text = _scalar_text(collector.parts.get("model"), "model")
    if model_text and model_text != settings.model_id:
        raise ServiceError(
            f"unsupported model {model_text!r}; this service runs {settings.model_id!r}",
            code="unsupported_model",
        )

    stream_text = _scalar_text(collector.parts.get("stream"), "stream")
    stream = False
    if stream_text:
        if stream_text.lower() == "false":
            stream = False
        else:
            raise ServiceError(
                "streaming responses are not supported; stream must be false or omitted",
                code="stream_unsupported",
            )

    return ParsedMultipartRequest(
        image_bytes=image_bytes,
        config_bytes=config_bytes,
        verbosity=verbosity,
        response_format=response_format,
        model=model_text or None,
        stream=stream,
    )
