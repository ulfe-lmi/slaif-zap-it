# ZAP-IT algorithm overview (zero-shot segmentation → zero-shot verification → dataset export)

This document explains how the ZAP-IT pipeline works, using the `configs/goats2.yaml` configuration as a concrete example. The goal of the pipeline is to *detect* objects (goats here) in images without supervised training data, then turn those detections into labeled training samples (e.g., YOLO). The implementation is a sequence of modular stages driven by configuration, with each stage referenced below using GitHub-compatible links.

## 1) Pipeline at a glance

The `zap-it-batch.py` entrypoint wires together the pipeline for images and video. It loads configuration and runs the segmentation/classification pipeline for each frame or image. See the high-level orchestration in [zap-it-batch.py](zap-it-batch.py) and the actual per-frame pipeline in [src/batch.py](src/batch.py).

At a high level, the pipeline for each image is:

1. **Preprocessing (ROI selection + optional resize).**
2. **SAM2 segmentation** to get candidate masks.
3. **Post-SAM2 filtering** (area and bounding-box constraints).
4. **Zero-shot CLIP classification** of each mask crop into labels (e.g., `goat` vs `negative`).
5. **Optional BLIP3 verification** to double-check low-confidence or targeted masks.
6. **Final label filtering** (keep only configured labels).
7. **Export results** as masks/visualizations/YOLO annotations.

You can see the precise control flow in [src/batch.py](src/batch.py).

## 2) Preprocessing: region-of-interest and resize

In `configs/goats2.yaml`, the `preprocessing` block defines the spatial ROI and resize behavior:

```yaml
preprocessing:
  roi: "3,1644,5565,1210"
  resize: 1.0
```

The ROI is applied via [modules/input/images.py](modules/input/images.py) and the resize is handled in the same input module, then used by the per-frame pipeline in [src/batch.py](src/batch.py). This step focuses computation on the experimental field area where goats appear, saving time and reducing false positives.

## 3) SAM2 segmentation: proposing candidate objects

The segmentation stage runs SAM2 to generate dense masks on the ROI (or full image). The per-frame pipeline calls `run_sam2` in [modules/segmenter](modules/segmenter), and then scales masks back into original image coordinates in [src/batch.py](src/batch.py).

The goats configuration uses a fairly dense grid and multi-scale crops:

```yaml
mask_generator:
  points_per_side: 64
  crop_n_layers: 2
  crop_overlap_ratio: 0.4
```

This provides many candidate masks for goats and other objects so later stages can filter them down.

## 4) Post-SAM2 filtering: fast pruning

The `postsam2processing` block discards masks that are too large or have bounding boxes bigger than expected for goats in the scene:

```yaml
postsam2processing:
  maxsize: 60000
  max_w: 300
  max_h: 200
```

The filtering logic is in [src/postprocessing.py](src/postprocessing.py), and is called inside the main pipeline in [src/batch.py](src/batch.py). This step reduces the number of candidate masks before running more expensive CLIP classification.

## 5) Zero-shot CLIP classification (core of zero-shot goat detection)

### How CLIP is applied

The CLIP classifier is configured in `configs/goats2.yaml` under `clip`. This section lists **natural language prompts** for each label (e.g., `goat`) and a `negative` category to explicitly discard masks that are likely not goats.

The classifier is implemented in [modules/classifier/clip.py](modules/classifier/clip.py). It:

- Flattens the prompt lists for each label into a single prompt set.
- Computes text embeddings for all prompts.
- For each mask:
  - crops the image around the mask (with padding),
  - computes CLIP image embeddings,
  - chooses the prompt with the highest similarity,
  - assigns that prompt’s label to the mask.

You can see these operations in the `filter_masks` and `classify_single` methods of [modules/classifier/clip.py](modules/classifier/clip.py).

### Why this is zero-shot

CLIP provides a **zero-shot classification** mechanism: it can evaluate how well an image crop matches a prompt like “a Boer goat in a grassy field” *without any training on your dataset*. In the goats example, the pipeline learns to label a mask as `goat` because the CLIP similarity between the image patch and goat prompts is higher than similarity to negative prompts.

This is the core zero-shot principle in ZAP-IT:

- **No task-specific training data is required.**
- **The prompts define the concept.**
- **The model generalizes from natural language descriptions.**

The masks and crops come from SAM2 segmentation in [src/batch.py](src/batch.py), and the prompt-based classification happens in [modules/classifier/clip.py](modules/classifier/clip.py).

## 6) Optional BLIP3 verification

The `blip3` section in `configs/goats2.yaml` adds a second, optional verification pass using a VLM. It asks a yes/no question (e.g., “Is there an animal in the image? Yes or no!”) to confirm that a candidate patch actually contains an animal, and can re-label or demote masks based on the answer. This step is implemented in [modules/verifier/blip3.py](modules/verifier/blip3.py) and wired in [src/batch.py](src/batch.py).

This is useful for catching low-confidence CLIP classifications or cleaning up ambiguous masks (like signs in the test field) without training a dedicated classifier.

## 7) Final label filter and outputs

After CLIP (and optional BLIP3), the pipeline drops any mask whose label is not in the visualization labels list (e.g., only keep `goat`). This is done in [src/batch.py](src/batch.py). Visualizations are generated by [modules/visualizer.py](modules/visualizer.py), using the stage-specific labels (`sam2`, `clip`, `blip3`) defined in `configs/goats2.yaml`.

## 8) Exporting training data for object detection (YOLO)

The real value for object detection comes from converting the zero-shot detections into training samples. ZAP-IT can export bounding boxes as YOLO labels using the `export_yolo_det` configuration block in `configs/goats2.yaml`:

```yaml
export_yolo_det:
  labels: "goat"
  trainsplit: 80
  sample_roi: true
```

The exporter is implemented in [modules/output/yolo.py](modules/output/yolo.py). It:

- Crops the original image to the ROI (if `sample_roi` is true).
- Converts each mask’s pixel region to a YOLO bounding box.
- Writes images/labels into `images/train`, `images/val`, `labels/train`, `labels/val`, with a generated `dataset.yaml`.

This means you can **bootstrap a labeled dataset** from unlabeled imagery using zero-shot segmentation + classification, then fine-tune a detector like YOLO on those labels for faster and more precise inference later.

## 9) Why this is useful for training data creation

Using the goats example, the pipeline enables you to:

- **Find goats without existing labels**: CLIP prompts plus SAM2 masks provide automatic goat detections.
- **Reduce annotation burden**: Instead of drawing boxes, you get them from masks.
- **Iterate quickly**: You can tweak prompts, thresholds, or ROI without re-labeling.
- **Bootstrap detector training**: YOLO labels can be used for supervised training, improving speed and precision over zero-shot inference in production.

The implementation details are embedded across the batch pipeline [src/batch.py](src/batch.py), CLIP classifier [modules/classifier/clip.py](modules/classifier/clip.py), and YOLO export logic [modules/output/yolo.py](modules/output/yolo.py).
