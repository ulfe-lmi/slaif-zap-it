# Algorithms

ZAP-IT builds object annotations by composing pretrained models and deterministic
post-processing. It does not train or fine-tune these models. Configuration
selects prompts, thresholds, filtering, rendering, and trusted batch outputs.

## Pipeline

```text
image
  -> optional ROI and resize
  -> SAM2 candidate masks
  -> area and bounding-box filtering
  -> optional CLIP labeling
  -> optional BLIP3 verification/relabeling
  -> keep-label filtering
  -> deterministic object ordering
  -> masks, metadata, visualizations, YOLO
```

The batch CLI and HTTP service share this algorithm chain through
`src.core.run_single_image()`. They differ at their trust and I/O boundaries:
the CLI accepts trusted paths and writers, while the service accepts one bounded
image and an allowlisted YAML subset.

## Preprocessing

The `preprocessing` section can crop a region of interest and uniformly resize
the inference image. Candidate masks are remapped to the original image
coordinates before areas, bounding boxes, identity images, and YOLO coordinates
are produced.

Downscaled masks use inverse nearest-neighbor projection so every destination
pixel samples a valid source pixel. This avoids the edge loss caused by the
legacy forward projection.

## SAM2 segmentation

SAM2 automatic mask generation proposes candidate regions without
task-specific training. The trusted CLI may tune generator parameters such as
point density, crop layers, intersection-over-union thresholds, and stability
thresholds.

The local service keeps the pinned SAM2 model resident and builds one fresh
generator per request from the strict safe scalar/profile contract. Uploaded
YAML cannot change generator construction controls, model identity, revision,
or device. Field and exact estimated-work caps are checked before inference;
the response records the source and raw candidate count in `service.sam2`.

SAM2 provides masks and available quality values; ZAP-IT does not reinterpret
those values as calibrated object confidence.

## Candidate filtering

Area and bounding-box limits remove unsuitable candidates before the more
expensive classification stages. These are deterministic pixel-space filters,
not learned predictions.

Filtering early reduces CLIP/BLIP3 work and bounds response construction. Final
keep-label filtering occurs after optional relabeling.

## CLIP classification

CLIP performs zero-shot classification over each surviving mask crop. A YAML
label maps to one or more natural-language prompts. ZAP-IT encodes the prompt
set, embeds each crop, and assigns the class associated with the highest
similarity.

The service keeps the pinned CLIP model resident and refreshes only
request-specific prompt embeddings. CLIP similarity is useful ranking evidence,
but it is not a calibrated probability.

Tracked examples such as [`configs/tomato.yaml`](../configs/tomato.yaml) show
the prompt format without depending on private or operator-held fixtures.

## BLIP3 verification

BLIP3/XGen-MM is an optional visual question-answering verifier. Canonical
service routing selects one target rule per candidate, asks a bounded question,
compares normalized true/false tokens exactly, and assigns a request-authored
terminal class. Trusted CLI retains explicit low-score/substring compatibility.

Service requests provide only rule mappings. The model, revision, FP16 dtype,
tokenizer, processor, device, cache, and residency strategy remain pinned. The
service limits each request to 32 planned questions and 32 generated tokens per
question. Each applicable candidate is composed once into one RGB image. Its
inclusive raw-mask bbox determines a bounded centered crop; exact Euclidean
dilation produces support, and a second exact dilation produces an exterior
contour. Support D pixels are restored from source bytes while the exterior
contour is painted with the configured RGB color and all other crop pixels are
Pillow-Gaussian-blurred scene context. A crop that cannot
contain support plus contour after independent endpoint clamping is rejected
for that candidate before image/model work. The fully composed image alone is
bilinearly resized for QA, with short side 256 and long side capped at 768.

Below 24,576 MiB, BLIP3 lives in host RAM until its stage. SAM2 and CLIP run on
GPU first; the registry then swaps them out, executes BLIP3 on GPU, and restores
the baseline. At or above 24,576 MiB, all three pinned FP16 holders remain on
the assigned GPU and no request-time movement occurs. Objective 009's real
matrix covers all four supported profiles. The resulting single images are
bounded before the pinned processor maps
arbitrary aspect ratios to a finite 378-pixel tile grid. The verifier's fixed
instruction follows the delimited client question and asks whether the region
inside the yellow outline itself is the requested object. Both modes expose only
logical `cuda:0` after an explicit operator index and UUID pin; the evidence is
bounded local research, not an SLA, accuracy claim, license clearance, or
external deployment.

## Deterministic object results

After final filtering, objects are ordered by descending area, centroid row,
centroid column, and original candidate index. Request-local instance IDs are
assigned from this order.

The same order drives:

- five-field normalized YOLO lines;
- per-object metadata;
- uint16 identity-mask values;
- overlap-preserving RLE masks;
- annotated visualizations.

Identity-mask overlaps use the documented deterministic winner policy. The
single-valued PNG cannot preserve overlap by itself, so the full response keeps
each object's source mask/RLE.

## Visualization

The service supports bounded in-memory annotated and alpha-overlay streams at
verbosity 3. It preflights raw RGB memory and artifact budgets before inference.

The L3 `mask_generator.debug: true` path has a separate raw-SAM2 diagnostic
renderer because one combined overlay cannot show which overlapping candidate
owned a pixel. It emits independent candidate contact-sheet tiles and three
all-candidate diagnostics: `sam2-union-coverage.png` is black for uncovered
pixels and white for covered pixels, `sam2-overlap-heatmap.png` is black at zero
and uses a fixed blue-to-red ramp scaled by the observed maximum overlap, and
`sam2-uncovered-pixels.png` is the exact source-resolution binary inverse of
the union before any display downscale. Coverage and overlap counts include
every non-empty raw mask, including candidates omitted from the bounded sheets.

Candidate IDs are one-based `source_index + 1` values in ascending generator
order; gaps identify empty raw proposals omitted before remapping. Each page is
three columns by four rows of 320x240 content tiles plus a 28-pixel label bar,
with at most eight pages and 96 represented candidates. A tile uses clamped
context padding of `ceil(10% of the larger bbox dimension)`, at least four
source pixels, bilinear RGB resizing, nearest-neighbor mask resizing and 45%
mask-color alpha. Padded tiles are enlarged when needed to fill their content
area; full-image diagnostics never upscale. Labels use three decimal
IoU/stability values, or `n/a` for
absent/non-finite values. The fixed names are
`sam2-candidates-page-0001.png` through `-0008.png`, the three diagnostic names
above, and no client text enters an artifact name.

The manifest reports exact full-resolution covered/uncovered counts, maximum
overlap and a bounded histogram (keys 0 through 255 plus an exact overflow
count), source and diagnostic dimensions, represented IDs and truncation. The
diagnostics never upscale and are nearest-neighbor downscaled to at most
2,000,000 pixels. Rendering is deterministic for equal inputs in one pinned
environment; PNG byte identity across arbitrary Pillow versions is not claimed.

Panoptic/Detectron2 rendering and Canny/Hough geometry helpers remain legacy
components. They are not executed by the canonical service and no absent field
is fabricated.

## YOLO output and dataset export

Every final object can produce:

```text
<class_id> <center_x> <center_y> <width> <height>
```

Coordinates are normalized to the original image dimensions with fixed decimal
formatting. The service returns these lines in `choices[0].text` and ZIP output.

The trusted batch exporter can additionally construct YOLO image/label folder
trees and a dataset manifest. Dataset export is intentionally not available
through the stateless API.

## Scientific limitations

- Outputs depend on pretrained model behavior, prompts, thresholds, and image
  conditions; they are not guaranteed ground truth.
- CLIP scores and SAM2 quality values are not calibrated end-to-end confidence.
- BLIP3 answers are generated text mapped by exact normalized configured tokens
  in the canonical service route; trusted legacy rules retain explicit
  substring compatibility.
- Qualification demonstrates bounded execution and stability, not accuracy or
  fitness for a specific deployment.
- Model licenses and use restrictions apply independently of repository code.

## Objective 020 responsibility split

SAM2 proposes; optional geometry removes only impossible candidates; CLIP2 sees
the complete rectangular source crop and supplies all finite cosine similarities
in configuration order; a permissive OR router chooses which target question
BLIP3 verifies. BLIP3 receives one separately composed contextual image and its
request-authored answer mapping determines the terminal label. CLIP identifiers
are machine-safe keys while their natural-language values are the exact prompts.

The raw CLIP radius is `floor(context_fraction * max(inclusive_bbox_width,
inclusive_bbox_height) + 0.5)`, then min/max bounded and clamped to the source.
Semantic accuracy, recall, and precision are not proved by the deterministic
CPU/fake tests.
