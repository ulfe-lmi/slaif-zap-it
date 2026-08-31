# Configuration reference

ZAP-IT pipelines are driven by YAML configuration files. Tracked examples live
in `configs/`. The trusted batch loader preserves legacy flexibility; the HTTP
service applies a strict allowlist and rejects unknown or unsafe fields.

At a glance the batch runner understands these top-level sections:

- `preprocessing`
- `mask_generator`
- `postsam2processing`
- `clip`
- `clip_routing`
- `blip3`
- `candidate_views`
- `geometry`
- `visualization`
- `images`
- `video`
- `export_yolo_det`

The trusted algorithm sections work for image directories and video files. The
service additionally requires the canonical `clip_routing` section when CLIP
and BLIP3 are both enabled; batch-only `geometry`, paths, and output controls
are not service capabilities. When a video is processed, extracted frames and JSON manifests are written to an
`output/<video-stem>/` folder next to the source file.

## `preprocessing`

Optional region-of-interest cropping and resizing. All keys are optional.

| Key      | Type          | Description |
| -------- | ------------- | ----------- |
| `roi`    | string/`false` | Rectangle described as `"x,y,w,h"`. When set to `false` the full image is used. The loader clamps coordinates to the image bounds. |
| `resize` | number        | Uniform scale factor applied after cropping. `1.0` (or omission) keeps the native size. Values &lt; 1.0 downscale, values &gt; 1.0 upscale. |
| `debug`  | bool          | If `true`, a JPEG snapshot of the ROI is saved alongside the outputs. |

## `mask_generator`

The service resolves a fresh request-local `SAM2AutomaticMaskGenerator` around
the resident pinned model. It accepts 14 total safe generator scalars,
including `use_m2m`, and only the following strict scalars;
booleans are not integers, numeric strings and nulls are invalid, and values
are never coerced or silently clamped:

| Key | Type | Public range |
|---|---|---|
| `points_per_side`, `points_per_batch` | integer | 1..1024 |
| `pred_iou_thresh`, `stability_score_thresh`, `box_nms_thresh`, `crop_nms_thresh`, `crop_overlap_ratio` | number | 0..1 |
| `stability_score_offset` | number | 0..10 |
| `mask_threshold` | number | -32..32 |
| `crop_n_layers` | integer | 0..8 |
| `crop_n_points_downscale_factor` | integer | 1..32 |
| `min_mask_region_area` | integer | 0..64,000,000 |
| `use_m2m`, `multimask_output`, `debug` | boolean | `true` or `false` |

The exact server defaults are `points_per_side: 8`, `points_per_batch: 8`,
`pred_iou_thresh: 0.5`, `stability_score_thresh: 0.5`,
`stability_score_offset: 1.0`, `mask_threshold: 0.0`, `box_nms_thresh: 0.7`,
`crop_n_layers: 0`, `crop_nms_thresh: 0.7`, `crop_overlap_ratio: 512 / 1500`,
`crop_n_points_downscale_factor: 1`, `min_mask_region_area: 0`,
`use_m2m: false`, and `multimask_output: true`.

The case-sensitive profiles `fast`, `balanced`, and `quality` override only
their documented fields; explicit request values take precedence over profile
values, which take precedence over defaults. The response records that source
independently for every scalar. Every configured crop layer must retain at
least one point per side after downscaling.

Operator caps are startup-only: `SLAIF_ZAP_IT_SAM2_MAX_POINTS_PER_SIDE=64`,
`SLAIF_ZAP_IT_SAM2_MAX_POINTS_PER_BATCH=64`,
`SLAIF_ZAP_IT_SAM2_MAX_CROP_N_LAYERS=2`,
`SLAIF_ZAP_IT_SAM2_MAX_ESTIMATED_PROMPTS=8192`,
`SLAIF_ZAP_IT_SAM2_MAX_ESTIMATED_MASK_PREDICTIONS=24576`, and
`SLAIF_ZAP_IT_SAM2_MAX_MIN_MASK_REGION_AREA=1000000`. A request over a field
cap or estimate cap returns non-retryable `resource_limit` (HTTP 413) with
sanitized requested/effective values, deterministic estimates, causes, public
limits and same-validator alternatives. Optional artifact capacity is handled
after inference and is not a configuration rejection.
Estimated prompts are the sum of `4**layer * int(points_per_side /
downscale_factor**layer)**2` over layers `0..crop_n_layers`; estimated mask
predictions multiply that sum by 3 for `multimask_output: true`, otherwise 1.
An accepted estimate at or above 80% of its cap adds a deterministic warning.

`profile` and `debug` are service controls and are not sent to SAM2. The
service fixes model identity/revision, checkpoint/config and cache locations,
logical `cuda:0`, dtype, residency, `point_grids: null`,
`output_mode: binary_mask`, and arbitrary constructor kwargs. These values
cannot be selected by uploaded YAML. The trusted batch path retains its
legacy model initialization and configured generator behavior.

At service verbosity 3, `mask_generator.debug: true` selects the bounded raw
candidate diagnostic. It is the only trigger; page layout, palette, crop,
label, destination and artifact names are fixed. Sheets are three by four
320x240 content tiles with a 28-pixel label bar, at most eight sheets and 96
represented candidates. IDs are one-based source-order IDs (`_source_index +
1`) and may have gaps for empty proposals. Each tile uses clamped
`ceil(10%)` context padding with a four-pixel minimum, RGB bilinear/mask
nearest-neighbor letterboxing and 45% exact-mask alpha. Labels show finite IoU
and stability to three decimals, otherwise `n/a`.

The fixed service names are `sam2-candidates-page-0001.png` through
`sam2-candidates-page-0008.png`, `sam2-union-coverage.png`,
`sam2-overlap-heatmap.png`, and `sam2-uncovered-pixels.png`. Union is black
uncovered/white covered; overlap is black at zero with a fixed ramp scaled by
the observed maximum; uncovered is the source-resolution binary inverse of
union. Candidate crops may be enlarged for contact-sheet readability;
diagnostics include all non-empty candidates, never upscale, and use
nearest-neighbor downscaling to at most 2,000,000 pixels. The L3
`service.sam2.raw_visualization` child reports exact source accounting,
overlap histogram overflow, represented IDs and explicit truncation. Below L3
the validator strips debug and leaves the response unchanged apart from its
existing bounded warning. The trusted CLI continues to emit its historical
rectangular JPEG patches.

## `postsam2processing`

Filters applied after SAM2 but before CLIP. Canonical area, inclusive bbox,
aspect-ratio, and border thresholds are optional; null disables a rule and no
implicit geometry limit is invented. Legacy aliases remain available with a
warning for trusted/API compatibility.

- `min_area`/`max_area`: optional mask-area bounds, 0..64,000,000.
- `min_width`/`max_width` and `min_height`/`max_height`: optional inclusive
  bbox bounds, 0..32,768.
- `min_aspect_ratio`/`max_aspect_ratio`: optional width/height bounds, 0..1000.
- `allow_border_touching`: strict boolean, default true.
- `debug`: when `true`, the final mask patches that survive filtering are saved
  as JPEGs.

At service verbosity 3, `post_filter_diagnostics` evaluates every candidate,
including empty masks. Rejection precedence is empty, min/max area, min/max
width, min/max height, min/max aspect ratio, then border touching; equality is
retained. Every rejection carries source ID, nullable inclusive bbox, area,
dimensions, reason, configured limit field/value, and a bounded record count.

## `candidate_views`

This dedicated algorithmic section controls the source-pixel boundary shared by
the CLIP and BLIP3 adapters. Omission of the section or either child selects
these effective values:

```yaml
candidate_views:
  clip:
    mode: raw_bbox_crop
    context_fraction: 0.10
    min_context_pixels: 0
    max_context_pixels: 64
  blip3:
    mode: single_dilated_blur
    context_fraction: 0.20
    min_context_pixels: 0
    max_context_pixels: 64
    crop_extent_multiplier: 2.0
    blur_sigma_fraction: 0.15
    contour_enabled: true
    contour_fraction: 0.02
    contour_min_pixels: 1
    contour_max_pixels: 3
    contour_rgb: [255, 224, 0]
```

CLIP accepts only the four raw-crop fields `mode: raw_bbox_crop`,
`context_fraction`, `min_context_pixels`, and `max_context_pixels`. BLIP3 accepts only
`single_dilated_blur`, `context_fraction` 0..0.5, `min_context_pixels` 0..256,
`max_context_pixels` 0..512 (not below the minimum),
`crop_extent_multiplier` 1..2, `blur_sigma_fraction` 0..0.5,
`contour_enabled` as a strict boolean, `contour_fraction` 0..0.25,
`contour_min_pixels` and `contour_max_pixels` 1..3 (maximum not below minimum),
and `contour_rgb` as exactly three strict integers 0..255. The old BLIP3
`mask_dilated`, `outside_fill`, `context_intensity`, and `contour_width` fields
are explicitly unsupported. Null, bool-as-number, non-finite, unknown,
out-of-range and inverted bounds are rejected without clamping.

For a raw mask with inclusive width `W`, height `H`, and `L = max(W, H)`, BLIP3
uses `raw_context_radius = ceil(context_fraction * L)` and
`effective_context_radius = min(max(raw_context_radius, min_context_pixels),
max_context_pixels)`. Support `D` is an exact squared-Euclidean disk dilation
of the boolean mask. The nominal source crop is
`ceil(crop_extent_multiplier * W)` by `ceil(crop_extent_multiplier * H)`,
centered on the inclusive bbox (lower coordinate on a half-pixel tie), then
independently clamped without shifting. Raw/support bboxes use inclusive
`xyxy`; the array-slice crop uses half-open `xyxy`. The crop must contain every
pixel of `D` plus contour or that candidate is rejected with
`crop_cannot_contain_support_and_contour`. The source crop is blurred with
Pillow `ImageFilter.GaussianBlur`, sigma
`min(max(blur_sigma_fraction * L, 2), 20)`, then support D is restored from
source bytes and only the exterior contour is painted with the configured RGB
color. The complete one-image result
is resized bilinearly with a 256-pixel target short side and 768-pixel maximum
long side. Disconnected components and holes are mask-derived, not rectangular
fills.

At L3, `clip.debug` emits the exact CLIP processor RGB input as
`clip-candidate-view-CANDIDATE-####.png`. A debug BLIP3 rule emits the exact
sole QA input as `blip3-verification-CANDIDATE-####-QUESTION-####.png`.
Candidate and question IDs are one-based; `filtered_index` is zero-based and is
assigned immediately after the SAM2 area/bbox filter. Names never contain
prompts, labels, rule names, answers, frame names or client paths. Effective
values and application status are present in every service response, while
bounded model-input records are L3-only.

## `clip` (optional)

Enables the CLIP-based zero-shot classifier. When the section is absent the
CLIP stage is skipped entirely.

- `debug` (bool): if set, every effective candidate view is written as a
  lossless PNG. The service uses the fixed tokenized name above; trusted legacy
  output prefixes it with a sanitized frame stem.
- `labels` (mapping): safe identifiers to one complete natural-language prompt
  string. Commas and line breaks are prompt content, not list syntax.

The trusted loader also supports flattened keys such as `"label goat":
"prompt1, prompt2"` for legacy CLI compatibility. The service uses exactly one
natural-language value per safe identifier, preserves every label's cosine
score in configuration order, and stores the deterministic winner in
`clip_label`/`clip_score`. When canonical `clip_routing` is present, its OR
rules—not the winner alone—decide BLIP3 admission.

## `blip3` (optional)

Configures the BLIP-3 VQA verifier that refines CLIP labels. Legacy CLI configs
may still contain model-level options, but the service accepts only nested
request rules. The operator pins the model, revision and FP16 dtype, loads it
from the local cache, and selects residency automatically from physical total
GPU memory. Below 24,576 MiB the live-qualified service swaps SAM2+CLIP and
BLIP3 at the BLIP3 stage boundary on the historical 11 GB RTX 2080 Ti. At or
above 24,576 MiB it keeps all three pinned holders resident on the assigned RTX
3090. Objective 009 provides real all-resident matrix evidence for all four
supported profiles. Both modes expose only logical `cuda:0` after an explicit
operator index and UUID pin. This is bounded local research evidence, not an
SLA, accuracy claim, commercial-license clearance, or external deployment;
geometry/panoptic and deployment/release gates remain separate. Nested mappings
define rules:

- In the service, each key matches one `clip_routing` target label. The router
  selects it before BLIP3, even when another CLIP label is top-1; `any,<score>`
  remains a trusted-CLI compatibility form only.

Each rule supports these fields:

- `question`: prompt passed to BLIP-3.
- `trueresult` / `falseresult`: exact normalized answer tokens in the service;
  legacy CLI may use substring compatibility.
- `newcategory` / `falsecategory`: terminal labels selected by true/false or
  unmatched answers.
- `debug`: when `true`, service verbosity 3 returns the exact single-image
  mask-isolated verification image as a lossless PNG named
  `blip3-verification-CANDIDATE-####-QUESTION-####.png`.
  The image is not a semantic-accuracy guarantee, and structured answers are
  returned independently; raw answers are never logged or used as metric labels.
  At lower service verbosity the nested flag is set to `false` and one
  aggregate warning is returned.

BLIP3 receives no second image, pane, divider, fill, or untouched rectangular
context. The compositor creates one crop from exact mask support and exterior
contour, restores source pixels under support, and Gaussian-blurs every other
local scene pixel before the sole bilinear resize. The fixed instruction is
exactly `The unblurred region inside the yellow boundary is the selected
candidate. The blurred surroundings are context only. Answer exactly Yes or No.`
and follows the delimited client question. `clip.padding` is unsupported by the
service; trusted legacy configs continue to use the separate CLIP compatibility
view.

The exact disk dilation uses a two-pass squared Euclidean distance transform
over only the target-bbox window expanded by the effective radius. It retains a
constant number of arrays proportional to that local window, has no
radius-sized image cache, and keeps the public maximum radius at 512. Resource
admission is also two-phase: CLIP debug artifacts are admitted before CLIP,
then actual post-CLIP labels/scores determine single-image BLIP3 debug admission
before any QA call. A candidate-local containment rejection consumes no model
call or debug artifact and does not mutate its label, score or answer. The
bounded L3 `blip3_candidate_views` list has one record per applicable candidate;
debug records remain one-for-one with QA artifacts. This is pixel-boundary
evidence, not a semantic-accuracy guarantee.

The service allows at most 32 nested rules/questions and fixes generation to at
most 32 new tokens per question. It rejects `model_name`, `revision`, `dtype`,
tokenizer/processor controls, paths, URLs, cache/download settings, devices,
commands and remote-code controls anywhere in an upload.

For canonical service rules, an answer normalized exactly to `trueresult`
selects `newcategory`; an exact `falseresult` selects `falsecategory`; and an
unmatched answer conservatively selects `falsecategory`. Trusted legacy rules
retain substring matching and their default `negative` fallback.

## `geometry` (legacy-only, not a service stage)

The Canny/Hough helpers remain available for trusted legacy integrations, but
the canonical core does not execute them and the API rejects a top-level
`geometry` field with `unsupported_field` before inference. They write TSV and
debug files through their legacy adapter. Future service activation requires a
separate governed scientific-stage order and an in-memory refactor.

## `diagnostic_artifacts`

This optional section narrows which eligible L3 diagnostic bytes are delivered;
it never turns on a stage debug flag and never selects a path or destination.
The strict normalized shape is:

```yaml
diagnostic_artifacts:
  stages: [sam2, clip, blip3, visualization]
  candidate_ids: null
  page: 1
  page_size: 48
```

Stages are unique fixed tokens, candidate IDs are unique one-based source IDs
1..256, pages are 1..65535, and page sizes are 1..48. Candidate filtering
applies only to CLIP/BLIP3 candidate PNGs; aggregate SAM2 and visualization
streams are not reinterpreted. Selection and pagination occur in deterministic
pipeline/name order. At verbosity below 3 the section is valid but `applied`
is false and no optional diagnostic artifact is delivered.

## `visualization`

Controls which intermediate and final results are rendered. The loader copies
`visualization.alpha` into a top-level `alpha` field (default `0.6`) used by the
composite builder. Additional keys:

- `labels`: optional whitelist of CLIP labels to keep at the end of the
  pipeline. Accepts either a comma-separated string or a list.
- `sam2` / `clip` / `blip3`: each is a list of renderer specifications. Every
  entry must contain an `id` (used to reference the rendered frame) and a
  `renderer` name.

The service renderer policy exposes only bounded in-memory annotated streams:

- `annotated` (the legacy spelling `alpha-overlay` is accepted): draws masks on
  top of the original image with an optional alpha override. At most eight safe
  ID streams execute, and only at service verbosity 3.
- `annotated-labelled`: a deterministic, Detectron2-free L3 stream accepted
  only under `visualization.blip3`. It begins with the same mask overlay and
  draws each final object's sanitized label and exact instance number. The
  optional strict-boolean `show_confidence` adds a finite two-decimal CLIP
  suffix. Structured labels remain available independently of rendering.
- `panoptic`, composite/file/video rules, unknown renderers and unsafe IDs are
  rejected by the service. The trusted CLI retains its compatibility writers.

Supported API request shape:

```yaml
visualization:
  blip3:
    - id: labelled-result
      renderer: annotated-labelled
      alpha: 0.55
      show_confidence: true
```

## `images`

Maps visualization IDs to directory names. The directories are created under
`--output-image-dir/<run-subdir>/` when the batch script is launched with an
`--output-image-dir` value. Each frame is written as a zero-padded JPEG sequence
(`0000001.jpg`, `0000002.jpg`, …).

## `video`

Maps visualization IDs to MJPEG encoders. Each value can be either a string (the
AVI filename) or a mapping with extra options:

- `filename`: desired output name.
- `fps`: frames per second (defaults to `24`).

Videos are written to `--output-video-dir/<run-subdir>/` when that CLI flag is
provided.

## `export_yolo_det` (optional)

Enables YOLO dataset export. When present the batch runner writes annotations to
a `yolo/` folder alongside the image outputs for each processed directory.
Supported options:

- `labels` (string): comma-separated class names. The order defines the class
  indices in the generated `dataset.yaml` and `data.yaml` files.
- `trainsplit` (number): percentage of samples placed in the `train` split
  (default `80`). Remaining samples go to `val`.
- `sample_roi` (bool): when `true`, the YOLO exporter crops the saved images to
  the configured preprocessing ROI before deriving boxes.

If YOLO export is enabled while `--output-image-dir` points elsewhere, the
exporter nests the dataset under that root to keep outputs together.

## Objective 020 migration

Service YAML uses `clip.labels` as `identifier: natural-language prompt` and
adds `clip_routing.route_to_blip3`. Use `candidate_views.clip.mode:
raw_bbox_crop`; its half-up context radius changes only the source crop
boundary. Add one matching BLIP3 rule per routing target with `question`,
`trueresult`, `falseresult`, `newcategory`, and `falsecategory`. Final
visualization labels name terminal BLIP3 categories.

The old `padding` and `label "name"` forms, `any,<score>` BLIP rules,
implicit `negative` mapping, and masked CLIP views are trusted-CLI
compatibility only. Canonical optional geometry fields replace the old aliases;
aliases remain accepted with a warning. `geometry` and `blip2` are batch-only.
Objective 021 implements non-fatal optional artifact delivery with structured
selection, pagination and truncation.
