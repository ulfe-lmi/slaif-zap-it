"""CLIP-based classifier module with unified interface."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


class _DryRunClipFilter:
    """Simulates CLIP behaviour by emitting deterministic labels."""

    def __init__(
        self,
        clip_config: Dict[str, Any] | None = None,
        *,
        canonical_labels: bool = False,
        verbosity=1,
        log_print_func=None,
    ):
        self.verbosity = verbosity
        self.log_print = log_print_func if log_print_func else (lambda *a, **k: None)
        self._canonical_labels = bool(canonical_labels)
        self.class_map: Dict[str, List[str]] = {}
        self._set_labels(clip_config or {})

    def _set_labels(self, clip_config: Dict[str, Any]) -> bool:
        canonical = bool(clip_config.get("_canonical_labels", self._canonical_labels))
        class_map = _class_map_from(clip_config, canonical_labels=canonical)
        changed = canonical != self._canonical_labels or class_map != self.class_map
        self._canonical_labels = canonical
        self.class_map = class_map
        return changed

    def update_labels(self, clip_config: Dict[str, Any]) -> bool:
        """Refresh request-local canonical labels without loading a model."""
        return self._set_labels(clip_config or {})

    def filter_masks(
        self,
        masks,
        _image_np,
        _out_dir,
        _fname_stem,
        artifact_sink=None,
        safe_artifact_names=False,
    ):
        del safe_artifact_names
        if self._canonical_labels and self.class_map:
            labels = tuple(self.class_map)
            for idx, mask in enumerate(masks):
                scores = {
                    label: round(
                        max(-1.0, min(1.0, 0.80 - 0.03 * label_index - 0.01 * idx)),
                        4,
                    )
                    for label_index, label in enumerate(labels)
                }
                winner = max(
                    scores,
                    key=lambda label: (scores[label], -labels.index(label)),
                )
                mask["clip_scores"] = scores
                mask["clip_label"] = winner
                mask["clip_score"] = scores[winner]
                mask["clip_prompt"] = self.class_map[winner][0]
            self.log_print(
                f"[_DryRunClipFilter] scored {len(masks)} masks over {len(labels)} labels",
                2,
                self.verbosity,
            )
            return masks

        for idx, mask in enumerate(masks, start=1):
            label = f"dryrun region {idx}"
            mask["clip_label"] = label
            mask["clip_score"] = 1.0
        self.log_print(
            f"[_DryRunClipFilter] assigned {len(masks)} dry-run labels", 2, self.verbosity
        )
        return masks


def _class_map_from(
    clip_config: Dict[str, Any], *, canonical_labels: bool = False
) -> Dict[str, List[str]]:
    """Parse canonical one-prompt labels and explicit trusted legacy labels."""
    class_map: Dict[str, List[str]] = {}
    labels_cfg = clip_config.get("labels", None)
    if isinstance(labels_cfg, dict):
        for cname, val in labels_cfg.items():
            if isinstance(val, str) and canonical_labels:
                # Canonical API values are one complete prompt.  Commas and
                # newlines are prompt content, never an implicit list split.
                class_map[str(cname)] = [val]
            elif isinstance(val, str):
                flat = val.replace("\n", ",")
                class_map[str(cname)] = [p.strip() for p in flat.split(",") if p.strip()]
            elif isinstance(val, (list, tuple)):
                class_map[str(cname)] = [str(prompt) for prompt in val if isinstance(prompt, str)]
    for key, val in clip_config.items():
        if isinstance(key, str) and key.lower().startswith("label "):
            cname = key.split("label ", 1)[1].strip()
            flat = str(val).replace("\n", ",")
            prompts = [p.strip() for p in flat.split(",") if p.strip()]
            class_map[cname] = prompts
    return class_map


class _ClipFilter:
    """Lightweight wrapper around a CLIP model for zero-shot classification."""

    def __init__(
        self,
        clip_config: Dict[str, Any],
        device="cuda",
        verbosity=1,
        log_print_func=None,
        local_files_only: bool = False,
    ):
        self.verbosity = verbosity
        self.device = device
        self.debug = clip_config.get("debug") is True
        self.log_print = log_print_func if log_print_func else (lambda *a, **k: None)
        if "padding" in clip_config:
            self.log_print(
                "[_ClipFilter] clip.padding is deprecated and ignored; "
                "candidate_views.clip controls mask-isolated context",
                1,
                verbosity,
            )

        self._canonical_labels = bool(clip_config.get("_canonical_labels", False))
        self.class_map: Dict[str, List[str]] = _class_map_from(
            clip_config, canonical_labels=self._canonical_labels
        )
        self._rebuild_prompt_index()

        # Local imports avoid touching transformers/torch when running in dry-run mode.
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch

        self.log_print("[_ClipFilter] loading clip-vit-base-patch32", 1, self.verbosity)
        model_name = clip_config.get("model_name", "openai/clip-vit-base-patch32")
        revision = clip_config.get("revision")
        load_kwargs = {"revision": str(revision)} if revision else {}
        if local_files_only:
            load_kwargs["local_files_only"] = True
        self.processor = CLIPProcessor.from_pretrained(model_name, **load_kwargs)
        self.model = CLIPModel.from_pretrained(model_name, **load_kwargs).to(device)
        if str(clip_config.get("dtype", "auto")).lower() == "float16" and str(device).startswith(
            "cuda"
        ):
            self.model = self.model.half()
        self.model.eval()
        try:
            self.model_dtype = next(self.model.parameters()).dtype
        except (AttributeError, StopIteration, TypeError):
            self.model_dtype = None

        if self.all_prompts:
            self._encode_text_prompts()
        else:
            self.text_embeds = None

    def _rebuild_prompt_index(self) -> None:
        """Recompute the flat prompt/class index from ``class_map``."""
        self.class_idx: List[str] = []
        self.all_prompts: List[str] = []
        for cname, p_list in self.class_map.items():
            for prompt in p_list:
                self.all_prompts.append(prompt)
                self.class_idx.append(cname)

    def _encode_text_prompts(self) -> None:
        torch = self._torch
        with torch.no_grad():
            text_inputs = self._move_inputs(
                self.processor(text=self.all_prompts, return_tensors="pt", padding=True)
            )
            text_emb = self.model.get_text_features(**text_inputs)
            self.text_embeds = text_emb / text_emb.norm(dim=-1, keepdim=True)

    def _move_inputs(self, inputs):
        """Move processor tensors and match the pinned model's float dtype."""
        moved = {}
        for key, value in inputs.items():
            if self._torch.is_tensor(value):
                if value.is_floating_point() and self.model_dtype is not None:
                    moved[key] = value.to(self.device, dtype=self.model_dtype)
                else:
                    moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def update_labels(self, clip_config: Dict[str, Any]) -> bool:
        """Re-encode prompts when a request supplies different labels.

        The resident service reuses one CLIP model across requests; the model
        weights stay untouched and only the cheap text-projection is
        recomputed when the effective class map changes. Returns whether an
        update was applied.
        """
        new_class_map = _class_map_from(
            clip_config or {}, canonical_labels=getattr(self, "_canonical_labels", False)
        )
        if new_class_map == self.class_map:
            return False
        self.class_map = new_class_map
        self._rebuild_prompt_index()
        if self.all_prompts:
            self._encode_text_prompts()
        else:
            self.text_embeds = None
        return True

    def classify_single(self, patch: np.ndarray, mask_idx: int):
        """Return the historical winning-label tuple for trusted callers."""
        label, score, prompt, _scores = self.classify_single_scores(patch, mask_idx)
        return (label, score, prompt)

    def classify_single_scores(self, patch: np.ndarray, mask_idx: int):
        """Classify one literal crop and return the complete label score vector."""
        import time

        torch = self._torch
        t0 = time.time()

        if self.text_embeds is None or self.text_embeds.numel() == 0:
            return (None, 0.0, "no prompt", {})

        with torch.no_grad():
            inp = self._move_inputs(self.processor(images=patch, return_tensors="pt"))
            emb = self.model.get_image_features(**inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            sim = torch.matmul(emb, self.text_embeds.T)
            sim_row = sim[0]
            values = [
                max(-1.0, min(1.0, float(sim_row[index]))) for index in range(len(self.all_prompts))
            ]
            label_scores: Dict[str, float] = {}
            label_prompts: Dict[str, str] = {}
            for index, label in enumerate(self.class_idx):
                label = label.strip('"')
                score = values[index]
                if label not in label_scores or score > label_scores[label]:
                    label_scores[label] = score
                    label_prompts[label] = self.all_prompts[index]
            ordered_labels = (
                list(self.class_map)
                if hasattr(self, "class_map")
                else list(dict.fromkeys(label.strip('"') for label in self.class_idx))
            )
            ordered_scores = {
                label: label_scores[label] for label in ordered_labels if label in label_scores
            }
            best_label = max(
                ordered_scores,
                key=lambda label: (ordered_scores[label], -list(ordered_scores).index(label)),
            )
            best_score = ordered_scores[best_label]
            best_prompt = label_prompts[best_label]

        t1 = time.time()
        self.log_print(
            f"[_ClipFilter] mask={mask_idx}, best_label='{best_label}', score={best_score:.4f}, time={t1 - t0:.2f}s",
            2,
            self.verbosity,
        )
        return (best_label, best_score, best_prompt, ordered_scores)

    def filter_masks(
        self,
        masks,
        image_np,
        out_dir,
        fname_stem,
        artifact_sink=None,
        safe_artifact_names=False,
        candidate_view_config=None,
        candidate_view_inputs=None,
        debug=None,
    ):
        from src.core.mask_views import CandidateViewConfig, build_mask_views, build_raw_clip_crop

        if self.text_embeds is None or self.text_embeds.numel() == 0 or not masks:
            return masks

        view_config = (
            candidate_view_config
            if isinstance(candidate_view_config, CandidateViewConfig)
            else CandidateViewConfig.from_mapping(
                {"mode": "raw_bbox_crop"}
                if candidate_view_config is None
                else candidate_view_config,
                stage="clip",
            )
        )
        debug_enabled = self.debug if debug is None else debug is True
        for i, m in enumerate(masks):
            seg = m["segmentation"]
            source_index = m.get("_source_index")
            source_candidate_id = (
                int(source_index) + 1 if type(source_index) is int and source_index >= 0 else i + 1
            )
            if view_config.mode == "raw_bbox_crop":
                view = build_raw_clip_crop(
                    image_np,
                    seg,
                    source_candidate_id,
                    view_config,
                    filtered_index=int(m.get("_filtered_index", i)),
                    debug=debug_enabled,
                )
                patch = view.rgb
                view_metadata = view.metadata_dict()
            else:
                view = build_mask_views(
                    image_np,
                    seg,
                    source_candidate_id,
                    view_config,
                    stage="clip",
                )
                patch = view.context_rgb
                view_metadata = view.metadata_dict()
            # Keep request-local crop provenance available to the router and
            # L3 diagnostics without exposing it as a model-control field.
            m["_clip_crop_metadata"] = dict(view_metadata)
            score_method = getattr(self, "classify_single_scores", None)
            if callable(score_method) and hasattr(self, "class_idx"):
                best_lbl, best_sc, best_prompt, score_vector = score_method(patch, i)
            else:
                best_lbl, best_sc, best_prompt = self.classify_single(patch, i)
                score_vector = {str(best_lbl): float(best_sc)} if best_lbl is not None else {}
            m["clip_label"] = best_lbl
            m["clip_score"] = best_sc
            m["clip_scores"] = dict(score_vector)
            m["clip_prompt"] = best_prompt

            if debug_enabled and best_prompt is not None:
                if safe_artifact_names:
                    patch_file = f"clip-candidate-view-CANDIDATE-{source_candidate_id:04d}.png"
                else:
                    legacy_stem = str(fname_stem).replace("\\", "/").rsplit("/", 1)[-1]
                    patch_file = (
                        f"{legacy_stem[:96] or 'image'}-clip-candidate-view-"
                        f"CANDIDATE-{source_candidate_id:04d}.png"
                    )
                if artifact_sink is not None:
                    artifact_sink.store_image(patch_file, patch, fmt="png")
                else:
                    patch_path = os.path.join(out_dir, patch_file)
                    Image.fromarray(patch).save(patch_path, "PNG")
                if candidate_view_inputs is not None:
                    record = dict(view_metadata)
                    record.update(
                        {
                            "stage": "clip",
                            "source_candidate_id": source_candidate_id,
                            "filtered_index": int(m.get("_filtered_index", i)),
                            "artifact_name": patch_file,
                            "artifact_status": (
                                artifact_sink.artifact_status(patch_file)
                                if artifact_sink is not None
                                and hasattr(artifact_sink, "artifact_status")
                                else "emitted"
                            ),
                        }
                    )
                    # The old compatibility view uses different bbox field
                    # names; retain its accepted record shape while raw crops
                    # expose the standardized metadata above.
                    candidate_view_inputs.append(record)
                self.log_print(
                    f"[_ClipFilter debug] => wrote debug patch: {patch_file}", 2, self.verbosity
                )

        return masks


def initialize(
    config: Dict[str, Any],
    *,
    dryrun: bool = False,
    device="cuda",
    verbosity: int = 1,
    log_print_func=None,
    local_files_only: bool = False,
) -> Dict[str, Any]:
    """Create a CLIP filter or a dry-run stub."""

    log = log_print_func or (lambda *a, **k: None)

    if dryrun:
        log("[classifier.clip] Initializing dry-run CLIP filter", 1, verbosity)
        return {
            "clip_filter": _DryRunClipFilter(
                config,
                canonical_labels=bool(config.get("_canonical_labels", False)),
                verbosity=verbosity,
                log_print_func=log,
            )
        }

    log("[classifier.clip] Initializing CLIP filter", 1, verbosity)
    clip_filter = _ClipFilter(
        config,
        device=device,
        verbosity=verbosity,
        log_print_func=log,
        local_files_only=local_files_only,
    )
    return {"clip_filter": clip_filter}


def run(
    state: Dict[str, Any] | None,
    params: Dict[str, Any],
    images,
    *,
    verbosity: int = 1,
    log_print_func=None,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """Run CLIP classification using the unified module interface."""
    import time

    started = time.perf_counter()
    log = log_print_func or (lambda *a, **k: None)
    if state is None:
        state = {}

    dryrun_mode = bool(params.get("dryrun", False))

    clip_filter = state.get("clip_filter")
    if clip_filter is None:
        clip_cfg = dict(params.get("config", {}) or {})
        if params.get("canonical_labels"):
            clip_cfg["_canonical_labels"] = True
        device = params.get("device", "cuda")
        init_state = initialize(
            clip_cfg, dryrun=dryrun_mode, device=device, verbosity=verbosity, log_print_func=log
        )
        state.update(init_state)
        clip_filter = state.get("clip_filter")
    else:
        # Resident runtime: keep the loaded model, re-encode prompts only when
        # this request supplies a different effective label map.
        update_labels = getattr(clip_filter, "update_labels", None)
        if callable(update_labels):
            update_config = dict(params.get("config", {}) or {})
            if params.get("canonical_labels"):
                if hasattr(clip_filter, "_canonical_labels"):
                    clip_filter._canonical_labels = True
                update_config["_canonical_labels"] = True
            update_labels(update_config)

    image_np = images[0] if isinstance(images, (list, tuple)) else images

    masks = params.get("masks")
    if masks is None:
        raise ValueError("CLIP classifier requires 'masks' in params")

    out_dir = params.get("out_dir")
    fname_stem = params.get("fname_stem", "image")
    artifact_sink = params.get("artifact_sink")
    safe_artifact_names = bool(params.get("safe_artifact_names", False))
    view_kwargs = {}
    if isinstance(clip_filter, _ClipFilter):
        view_kwargs = {
            "candidate_view_config": params.get("candidate_view_config"),
            "candidate_view_inputs": params.get("candidate_view_inputs"),
            "debug": params.get("config", {}).get("debug", clip_filter.debug),
        }

    if artifact_sink is not None:
        processed_masks = clip_filter.filter_masks(
            masks,
            image_np,
            out_dir,
            fname_stem,
            artifact_sink=artifact_sink,
            safe_artifact_names=safe_artifact_names,
            **view_kwargs,
        )
    else:
        processed_masks = clip_filter.filter_masks(
            masks,
            image_np,
            out_dir,
            fname_stem,
            safe_artifact_names=safe_artifact_names,
            **view_kwargs,
        )
    meta = {
        "num_masks": len(processed_masks) if processed_masks is not None else 0,
        "scoring_time_ms": (time.perf_counter() - started) * 1000.0,
        "crop_time_ms": 0.0,
    }
    return state, processed_masks, meta


__all__ = ["initialize", "run"]
