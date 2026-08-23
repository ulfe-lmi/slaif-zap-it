# Configuration Files

ZAP-IT pipelines are driven by YAML configuration files. Examples live in
`configs/` and can be used as a starting point. Unknown keys are ignored so you
can keep notes in the file, but the sections below document the entries that
are consumed by the current codebase.

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

Parameters forwarded to the SAM2 automatic mask generator. Any omitted value
uses the library default (`None` in the constructor call).

- `model_name` (string, default `facebook/sam2-hiera-large`)
- `points_per_side`
- `pred_iou_thresh`
- `stability_score_thresh`
- `min_mask_region_area`
- `crop_n_layers`
- `crop_n_points_downscale_factor`
- `crop_overlap_ratio`
- `box_nms_thresh`
- `multimask_output`
- `debug` (bool) – when enabled ZAP-IT saves each SAM2 patch to disk for
  inspection before any filtering.

## `postsam2processing`

Filters applied after SAM2 but before CLIP. All thresholds are optional; if a
value is missing the code uses a very large default so that nothing is removed.

- `maxsize`: maximum mask area in pixels.
- `max_w`: maximum bounding-box width in pixels.
- `max_h`: maximum bounding-box height in pixels.
- `debug`: when `true`, the final mask patches that survive filtering are saved
  as JPEGs.

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

Configures the BLIP-3 VQA verifier that refines CLIP labels. The section can
mix model-level options with per-label rules:

- `model_name` (string, default `Salesforce/xgen-mm-phi3-mini-instruct-r-v1`).
- Any other top-level scalar is forwarded to the BLIP loader unchanged.
- Nested mappings define rules. Keys that match existing CLIP labels trigger the
  associated question whenever that label is assigned. Keys that start with
  `"any,"` declare a CLIP score threshold. For example `any,0.15` fires when the
  mask's CLIP score is ≤ 0.15, regardless of label.

Each rule supports these fields:

- `question`: prompt passed to BLIP-3.
- `trueresult` / `falseresult`: substrings that signal a positive/negative
  answer (case-insensitive).
- `newcategory`: optional replacement label applied when the answer is true.
- `debug`: when `true`, crops and the raw BLIP-3 answer text are saved.

Masks that match the false string are relabelled to `negative`; matches on the
true string keep their label or take `newcategory` if provided.

## `geometry` (legacy-only, not a service stage)

The Canny/Hough helpers remain available for trusted legacy integrations, but
the canonical core does not execute them and the API rejects a top-level
`geometry` field with `unsupported_field` before inference. They write TSV and
debug files through their legacy adapter. Future service activation requires a
separate governed scientific-stage order and an in-memory refactor.

## `visualization`

Controls which intermediate results are rendered. The loader copies
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
- `panoptic`, composite/file/video rules, unknown renderers and unsafe IDs are
  rejected by the service. The trusted CLI retains its compatibility writers.

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
