# Configuration reference

ZAP-IT pipelines are driven by YAML configuration files. Tracked examples live
in `configs/`. The trusted batch loader preserves legacy flexibility; the HTTP
service applies a strict allowlist and rejects unknown or unsafe fields.

At a glance the batch runner understands these top-level sections:

- `preprocessing`
- `mask_generator`
- `postsam2processing`
- `clip`
- `blip3`
- `geometry`
- `visualization`
- `images`
- `video`
- `export_yolo_det`

The same configuration works for image directories and video files. When a
video is processed, extracted frames and JSON manifests are written to an
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
the resident pinned model. It accepts only the following strict scalars;
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
cap or estimate cap returns non-retryable `resource_limit` (HTTP 413).
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

## `postsam2processing`

Filters applied after SAM2 but before CLIP. All thresholds are optional; if a
value is missing the code uses a very large default so that nothing is removed.

- `maxsize`: maximum mask area in pixels.
- `max_w`: maximum bounding-box width in pixels.
- `max_h`: maximum bounding-box height in pixels.
- `debug`: when `true`, the final mask patches that survive filtering are saved
  as JPEGs.

At service verbosity 3, the response reports the filter's deterministic
short-circuit decisions in `post_filter_diagnostics`. The area comparison is
terminal and occurs before segmentation access: a `maxsize` rejection carries
the exact area and zero bbox dimensions because those dimensions were not
evaluated. Empty masks also carry zero dimensions for their distinct reason.
Rejection precedence is `maxsize`, empty mask, `max_w`, then `max_h`; each
comparison rejects only when the measured value is strictly greater, so equality
is retained. Width and height use inclusive extents of the remapped
segmentation. Aggregate counts reconcile evaluated, retained and each removal
reason. Numeric-only rejection records are ordered by source candidate and capped
at 256, with
`rejections_truncated` for rejected candidates not represented. These records
describe configured filtering and do not establish SAM2 recall or semantic
accuracy.

## `clip` (optional)

Enables the CLIP-based zero-shot classifier. When the section is absent the
CLIP stage is skipped entirely.

- `padding` (integer, default `20`): how many pixels to expand each bounding box
  before cropping.
- `debug` (bool): if set, every crop is written as `*-patch*.jpg` with the
  winning prompt in the filename.
- `labels` (mapping): label names to comma-separated prompt strings. Literal
  block style (`|`) is recommended so that commas and line breaks are preserved.

The loader also supports flattened keys such as `"label goat": "prompt1, prompt2"`
for convenience. Every mask receives the label with the highest CLIP score; the
score is stored in `clip_score` for later stages.

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

- Nested mappings define rules. Keys that match existing CLIP labels trigger the
  associated question whenever that label is assigned. Keys that start with
  `"any,"` declare a CLIP score threshold. For example `any,0.15` fires when the
  mask's CLIP score is ≤ 0.15, regardless of label.

Each rule supports these fields:

- `question`: prompt passed to BLIP-3.
- `trueresult` / `falseresult`: substrings that signal a positive/negative
  answer (case-insensitive).
- `newcategory`: optional replacement label applied when the answer is true.
- `debug`: when `true`, service verbosity 3 returns the exact paired
  mask-aware verification image as a lossless PNG named
  `blip3-verification-{candidate_index:04d}-{question_index:04d}.png`.
  The image is not a semantic-accuracy guarantee, and structured answers are
  returned independently; raw answers are never logged or used as metric labels.
  At lower service verbosity the nested flag is set to `false` and one
  aggregate warning is returned.

BLIP3 does not receive the old rectangle-only crop. The verifier derives one
half-open crop from the complete mask bbox, adds symmetric padding of
`max(16, ceil(12.5% of the larger bbox dimension))`, and enforces a 128-pixel
minimum dimension. It scales uniformly by explicit nearest-neighbor mapping
toward a 256-pixel short side, capped at 768 pixels on the long side. The
paired image has untouched context on the left, a four-pixel dark divider, and
an aligned right spotlight with exact selected pixels, 40% integer-channel
dimming outside the mask, and a yellow exterior four-pixel contour.

The service allows at most 32 nested rules/questions and fixes generation to at
most 32 new tokens per question. It rejects `model_name`, `revision`, `dtype`,
tokenizer/processor controls, paths, URLs, cache/download settings, devices,
commands and remote-code controls anywhere in an upload.

Masks that match the false string are relabelled to `negative`; matches on the
true string keep their label or take `newcategory` if provided.

## `geometry` (legacy-only, not a service stage)

The Canny/Hough helpers remain available for trusted legacy integrations, but
the canonical core does not execute them and the API rejects a top-level
`geometry` field with `unsupported_field` before inference. They write TSV and
debug files through their legacy adapter. Future service activation requires a
separate governed scientific-stage order and an in-memory refactor.

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
