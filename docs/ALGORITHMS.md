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

The local service uses an operator-fixed resident generator. Uploaded YAML
cannot change generator construction, model identity, revision, or device.

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

BLIP3/XGen-MM is an optional visual question-answering verifier. Rules can
target a CLIP label or low-score candidates, ask a bounded question, interpret
configured true/false substrings, and optionally assign a replacement class.

Service requests provide only rule mappings. The model, revision, FP16 dtype,
tokenizer, processor, device, cache, and residency strategy remain pinned. The
service limits each request to 32 planned questions and 32 generated tokens per
question. Each question receives a deterministic mask-aware paired image rather
than a rectangle-only patch. The complete mask bbox receives symmetric
`max(16, ceil(12.5% of the larger bbox dimension))` context and a 128-pixel
minimum crop extent. The crop is uniformly nearest-neighbor scaled toward a
256-pixel short side, with a 768-pixel long-side cap, then rendered as
untouched context, a four-pixel divider, and a spotlight that preserves selected
pixels, dims other exterior pixels to 40%, and paints only the exterior
four-pixel dilation ring yellow.

Below 24,576 MiB, BLIP3 lives in host RAM until its stage. SAM2 and CLIP run on
GPU first; the registry then swaps them out, executes BLIP3 on GPU, and restores
the baseline. At or above 24,576 MiB, all three pinned FP16 holders remain on
the assigned GPU and no request-time movement occurs. Objective 009's real
matrix covers all four supported profiles. The resulting paired images are
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
- BLIP3 answers are generated text interpreted by configured substring rules.
- Qualification demonstrates bounded execution and stability, not accuracy or
  fitness for a specific deployment.
- Model licenses and use restrictions apply independently of repository code.
