"""Pure, request-local permissive routing for complete CLIP score vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

ROUTING_PRIMARY_REASONS = (
    "target_top_1",
    "target_in_top_k",
    "target_within_score_margin",
    "target_exceeded_minimum_score",
    "explicitly_uncertain",
    "clear_negative",
)


@dataclass(frozen=True)
class ClipRoutingDecision:
    """One deterministic routing decision and its complete audit record."""

    record: Mapping[str, Any]
    routed: bool


def _source_candidate_id(mask: Mapping[str, Any], ordinal: int) -> int:
    value = mask.get("_source_index")
    return int(value) + 1 if type(value) is int and value >= 0 else ordinal + 1


def _ordered_scores(mask: Mapping[str, Any]) -> dict[str, float]:
    scores = mask.get("clip_scores")
    if isinstance(scores, Mapping) and scores:
        return {str(label): float(score) for label, score in scores.items()}
    label = mask.get("clip_label")
    if label is None:
        return {}
    return {str(label): float(mask.get("clip_score", 0.0))}


def route_clip_candidate(
    mask: Mapping[str, Any],
    routing_config: Mapping[str, Any],
    *,
    ordinal: int = 0,
) -> ClipRoutingDecision:
    """Evaluate one candidate with OR semantics and stable tie-breaking."""
    route = routing_config.get("route_to_blip3", routing_config)
    target_labels = tuple(str(label) for label in route.get("labels", ()))
    scores = _ordered_scores(mask)
    config_order = tuple(scores)
    if not config_order:
        source_id = _source_candidate_id(mask, ordinal)
        return ClipRoutingDecision(
            {
                "source_candidate_id": source_id,
                "filtered_index": int(mask.get("_filtered_index", ordinal)),
                "clip_scores": {},
                "winner": None,
                "winning_label": None,
                "chosen_target": None,
                "target_rank": None,
                "chosen_target_rank": None,
                "target_score": None,
                "best_score": None,
                "best_score_delta": None,
                "route_to_blip3": False,
                "matched_conditions": [],
                "primary_reason": "clear_negative",
                "cap_outcome": "not_applicable",
                "crop_metadata": dict(mask.get("_clip_crop_metadata", {})),
            },
            False,
        )

    order_index = {label: index for index, label in enumerate(config_order)}
    ranked = sorted(config_order, key=lambda label: (-scores[label], order_index[label]))
    winner = ranked[0]
    available_targets = [label for label in target_labels if label in scores]
    chosen_target = (
        max(
            available_targets,
            key=lambda label: (scores[label], -target_labels.index(label)),
        )
        if available_targets
        else None
    )
    best_score = scores[winner]
    target_score = scores[chosen_target] if chosen_target is not None else None
    target_rank = ranked.index(chosen_target) + 1 if chosen_target in ranked else None

    matched: list[str] = []
    if chosen_target is not None and chosen_target == winner:
        matched.append("target_top_1")
    top_k = route.get("top_k")
    if chosen_target is not None and top_k is not None and target_rank <= int(top_k):
        matched.append("target_in_top_k")
    margin = route.get("score_margin_from_best")
    delta = None if target_score is None else best_score - target_score
    if chosen_target is not None and margin is not None and delta <= float(margin) + 1e-12:
        matched.append("target_within_score_margin")
    minimum = route.get("minimum_target_score")
    if chosen_target is not None and minimum is not None and target_score >= float(minimum):
        matched.append("target_exceeded_minimum_score")
    if winner in tuple(route.get("uncertain_labels", ())):
        matched.append("explicitly_uncertain")
    routed = bool(matched)
    primary = next(
        (reason for reason in ROUTING_PRIMARY_REASONS if reason in matched), "clear_negative"
    )
    source_id = _source_candidate_id(mask, ordinal)
    record = {
        "source_candidate_id": source_id,
        "filtered_index": int(mask.get("_filtered_index", ordinal)),
        "clip_scores": dict(scores),
        "winner": winner,
        "winning_label": winner,
        "chosen_target": chosen_target,
        "target_rank": target_rank,
        "chosen_target_rank": target_rank,
        "target_score": target_score,
        "best_score": best_score,
        "best_score_delta": delta,
        "route_to_blip3": routed,
        "matched_conditions": matched,
        "primary_reason": primary,
        "cap_outcome": "not_applicable",
        "crop_metadata": dict(mask.get("_clip_crop_metadata", {})),
    }
    return ClipRoutingDecision(record, routed)


def apply_clip_routing(
    masks: Sequence[dict[str, Any]], routing_config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Annotate all candidates and return the routed subset plus counts."""
    route = routing_config.get("route_to_blip3", routing_config)
    decisions = [
        route_clip_candidate(mask, route, ordinal=ordinal) for ordinal, mask in enumerate(masks)
    ]
    initially_routed = [decision for decision in decisions if decision.routed]
    max_candidates = route.get("max_candidates")
    retained = initially_routed
    if max_candidates is not None:
        retained = sorted(
            initially_routed,
            key=lambda decision: (
                -(
                    float(decision.record["target_score"])
                    if decision.record["target_score"] is not None
                    else float("-inf")
                ),
                int(decision.record["source_candidate_id"]),
            ),
        )[: int(max_candidates)]
    retained_ids = {id(decision) for decision in retained}
    routed_masks: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for mask, decision in zip(masks, decisions):
        record = dict(decision.record)
        if decision.routed and max_candidates is not None:
            if id(decision) in retained_ids:
                record["cap_outcome"] = "retained"
                routed_masks.append(mask)
            else:
                record["cap_outcome"] = "capped_out"
                record["initially_routed"] = True
                record["route_to_blip3"] = False
                record["primary_reason_before_cap"] = record["primary_reason"]
                record["matched_conditions_before_cap"] = list(record["matched_conditions"])
        elif decision.routed:
            record["cap_outcome"] = "not_applied"
            routed_masks.append(mask)
        else:
            record["cap_outcome"] = "not_routed"
        mask["clip_routing"] = record
        mask["_route_to_blip3"] = decision.routed and (
            max_candidates is None or id(decision) in retained_ids
        )
        if decision.routed and record["cap_outcome"] == "capped_out":
            record["primary_reason"] = "max_candidate_limit"
        diagnostics.append(record)
    return (
        routed_masks,
        diagnostics,
        {
            "initially_routed": len(initially_routed),
            "routed_after_cap": len(routed_masks),
        },
    )


__all__ = [
    "ROUTING_PRIMARY_REASONS",
    "ClipRoutingDecision",
    "apply_clip_routing",
    "route_clip_candidate",
]
