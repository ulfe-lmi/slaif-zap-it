# Proposed SLAIF gateway integration

This document defines a proposed future dependency contract; it is not a claim
that integration is implemented in either repository. The future
`slaif-api-gateway` backend call uses the narrow Responses facade. The current
gateway repository does not yet support this multimodal/image-generation path
and is unchanged by this repository's work.

## Fixed request mapping

- The future backend request is JSON `POST /v1/responses` with the fixed model
  `zap-it-1`.
- It uses the normal Responses input-item/content-part structure with exactly
  one inline strict-base64 `input_image` data URL and exactly one inline
  strict-base64 YAML `input_file` data URL. The file name is bounded metadata,
  not a path or a source of authority.
- The request is stateless and non-streaming: `store=false`, `stream=false`,
  and `background=false`. Unsupported state, URLs, file IDs, any other tools,
  extra content and retries remain explicit errors.
- The only optional declaration is
  `tools: [{"type": "image_generation"}]`. The service's output is assistant
  `output_text` containing the deterministic public projection and, when that
  declaration is requested, exactly one standard
  `image_generation_call.result` PNG. This is an annotated ZAP-IT result, not
  generative image output.

The gateway must not route public or general requests through the native
`/v1/completions` endpoint. The gateway must not proxy native `verbosity`,
`response_format`, or invent a ZAP-specific output type/bypass. JSON/ZIP debug
artifacts remain private and are not gateway output. The gateway uses the
public JSON projection above and does not map it to OpenAI text-completion
semantics.

## Authorization and topology

Client authorization terminates at the gateway. The gateway creates a distinct,
high-entropy backend bearer secret and supplies it only through
`SLAIF_ZAP_IT_API_KEY`. Client `Authorization` is never forwarded, persisted or
logged. The backend timeout remains bounded and retries are not automatic.

The first supported topology for this future dependency is co-located loopback
only. Cross-host, LAN, TLS/mTLS and public exposure require a separate
architecture and authorization review. Per-user authorization, quota, rate,
billing, TLS termination and other gateway functions are not invented inside
ZAP-IT.

## Current state and later qualification

The current `slaif-api-gateway` repository lacks the canonical Responses
multimodal/image-generation path described here and was not changed by this
PR. A later cross-repository deliverable must qualify the full official SDK ->
gateway -> ZAP-IT path, including request/error mapping, fixed-model and
bearer-boundary behavior, timeout and privacy controls, and the exact standard
Responses output. That qualification is not claimed to pass here.

The later gateway evidence must account for ZAP-IT's work truthfully as
non-token processing. It must not fabricate token counts or OpenAI text-
completion semantics where the service exposes no such usage.
