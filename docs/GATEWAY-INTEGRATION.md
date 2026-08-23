# Future SLAIF gateway integration contract

This document is a follow-up contract, not an implementation. Objective 006-a
does not modify ulfe-lmi/slaif-api-gateway, add an adapter, or run gateway
E2E. A future gateway PR must be separately reviewed and must report this
contract as zap-it-1 backed by the co-located loopback ZAP-IT service.

## Fixed request mapping

- Native module ID and model mapping are fixed to backend zap-it-1.
- The backend request is multipart POST /v1/completions.
- Exactly one bounded JPEG/PNG/WebP image and one bounded UTF-8 API-safe
  YAML/config file are accepted.
- verbosity is 0..3, response_format is json|zip, and stream=false.
- URLs, file IDs, multiple images/configs, retries, arbitrary backend/model/
  revision/path/device fields and caller-selected service settings are rejected.
- The gateway applies a bounded timeout and sends no automatic retries.

## Authorization and topology

Client authorization terminates at the gateway. The gateway creates a distinct,
high-entropy backend bearer secret and supplies it only through
SLAIF_ZAP_IT_API_KEY. Client Authorization is never forwarded, persisted or
logged. The first supported topology is co-located loopback only; cross-host,
LAN, TLS/mTLS and public exposure require a separate architecture and
authorization review.

## Required future tests and accounting

The gateway PR must test request cardinality, YAML policy parity, fixed model
mapping, bearer-boundary behavior, response/error mapping, JSON/ZIP artifact
limits, timeout behavior, privacy-safe logs and backend cleanup. It must map
the service's zero token usage honestly to non-token accounting and must not
invent token counts or OpenAI text-completion semantics. Gateway E2E is
NOT INCLUDED — separate repository/order required for this candidate.
