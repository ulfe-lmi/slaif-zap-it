# OpenAI Responses-compatible facade

ZAP-IT exposes a deliberately narrow, stateless, non-streaming
`POST /v1/responses` adapter. It runs the same typed in-memory SAM2, filtering,
CLIP, routing, optional BLIP3 and final-ordering pipeline as
`POST /v1/completions`, but projects the result into a public JSON text item.
It is a future gateway surface, not a general OpenAI model or public/WAN
deployment.

## Supported request

Requests are exactly `Content-Type: application/json` with no query parameters:

```json
{
  "model": "zap-it-1",
  "input": [{
    "role": "user",
    "content": [
      {"type": "input_image", "detail": "auto",
       "image_url": "data:image/png;base64,<strict-base64>"},
      {"type": "input_file", "filename": "task.yaml",
       "file_data": "data:application/yaml;base64,<strict-base64>"}
    ]
  }]
}
```

The two content parts may be in either order. There is exactly one JPEG, PNG or
WebP image and one UTF-8 YAML file. The image declaration MIME must match the
decoded image format. The YAML MIME must be one of
`application/yaml`, `application/x-yaml`, `text/yaml`, `text/x-yaml` or
`text/plain`. `filename` is metadata only: it is an ASCII `.yaml`/`.yml`
basename of bounded length and is never used in a path, prompt, ID, log or
artifact name. Data URLs are inline strict base64 only; URLs, `file_id`,
`file_url`, percent encoding, paths and empty data are rejected.

The only optional controls are `store: false`, `stream: false`,
`background: false`, and either no `tools` or exactly
`[{"type":"image_generation"}]`. The tool does not invoke an image model; it
requests the deterministic final annotated PNG described below. Every other
field, message, source, tool, conversation, previous response and state
operation is rejected. There are no retrieve, update, delete or list routes.

The body cap is derived from the configured decoded image/config upload caps by
exact base64 expansion plus a fixed bounded JSON-envelope allowance. Decoded
dimensions, pixels, YAML structure, model resources, object count, response
bytes, timeout and shared one-request admission use the existing service
policies. Authentication is the existing fixed deployment bearer; strict
loopback development retains the key-optional behavior.

Cardinality is classified before image or YAML decoding. An empty or file-only
content list returns `missing_image`; an image-only list returns `missing_config`;
the second image or YAML file returns `duplicate_image` or `duplicate_config`.
Unsupported content types retain the explicit `unsupported_field` rejection.
The accepted shape remains exactly two content parts: one image and one YAML
file.

## Successful output

The completed response follows the current official Responses shape: `id` is
`resp_...`, the assistant item is `msg_...`, and an optional image item is
`ig_...`. Timestamps are bounded Unix seconds and are protocol metadata. The
response omits `usage`; SAM2/CLIP/BLIP3 work is not LLM token accounting.

The assistant `output_text` is canonical JSON serialized with sorted keys,
compact separators, UTF-8 characters and `allow_nan=false`. Its exact
`schema_version` is `zap-it.public.v1`. It contains the effective config digest,
original image dimensions and class map, SAM2 requested/effective/source
values and counts, complete stage candidate counts, effective CLIP/BLIP3 view
and routing policy, bounded prompt metadata, final ordered object records, and
sanitized deterministic warnings. Public object records reuse the private L2
record builder and omit `mask_rle`.

The projection excludes request/message/tool IDs, timestamps, runtime/GPU
provenance, operator limits and paths, masks and identity PNGs, raw candidates,
geometry rejection lists, candidate-view records, debug metadata, ZIP members,
contact sheets and private visualization streams. Equal decoded image,
effective YAML, fixed model outputs and renderer version yield equal projection
text. Outer protocol IDs and timestamps intentionally do not.

When the image tool is present, the successful response echoes exactly
`tools: [{"type":"image_generation"}]`, sets `tool_choice: "auto"`, retains
`parallel_tool_calls: false`, and appends exactly one completed
`image_generation_call`. Its `result` is standard base64 PNG (not a data URL),
created by the existing `render_annotated_labelled` renderer with
`alpha=0.5` and `show_confidence=false`, then encoded by the shared PNG
encoder. Without the tool declaration, successful responses retain
`tools: []`, `tool_choice: "none"`, and `parallel_tool_calls: false` and have
no image item. The maintained response schema rejects mismatched tool/output
combinations. Raw, encoded and complete response budgets are checked; an
essential PNG that does not fit returns the typed `response_too_large` error
rather than being silently omitted.

Errors use the bounded OpenAI-shaped form:

```json
{"error":{"message":"sanitized message","type":"invalid_request_error",
"param":"input[0].content[0].image_url","code":"invalid_image"}}
```

The safe request ID is in `x-request-id`, not the error body. Client/input and
capacity failures use `invalid_request_error`, authentication uses
`authentication_error`, and readiness/busy/timeout/inference failures use
`server_error`; existing HTTP statuses, typed codes and safe retry headers are
preserved. Error messages never contain request bodies, YAML, base64, secrets,
answers, prompts, paths, stack traces or model internals.

## Official SDK qualification

The development extra pins `openai==3.7.0`. Runtime code has no SDK import.
The bounded operator check builds a small RGB PNG and safe YAML in memory,
constructs inline data URLs, calls the local service with the official client,
uses `response.output_text`, iterates typed `response.output` items and strictly
decodes the PNG result:

```bash
.venv/bin/python scripts/qualify_responses.py \
  --host "$SLAIF_ZAP_IT_HOST" --port "$SLAIF_ZAP_IT_PORT" \
  --evidence-root "$SLAIF_ZAP_IT_TMP_ROOT"
```

The script retains only bounded mode-0600 summary/hash evidence below a
mode-0700 RAM-backed evidence directory. An optional PNG is written only to an
explicit operator path. It prints statuses, counts, sizes, hashes and timing,
never the bearer or request/output content.

`/v1/completions` remains the native private multipart research/debug surface.
Its identity masks, RLE, candidate views, contact sheets, ZIPs and other L3
diagnostics are not passed through this facade. `slaif-api-gateway` is unchanged
and does not yet provide the canonical Responses multimodal/image-generation
path, so gateway end-to-end qualification and any public deployment are later
separate work.
