# Algorithms Used

ZAP-IT combines several well known algorithms and models to form a complete pipeline.

## SAM2 – Segment Anything Model 2
SAM2 generates segmentation masks given an input image. The pipeline runs it in a single pass, optionally on a resized version of the image or ROI. Parameters such as point density and IoU thresholds are configurable in the YAML file.

## CLIP – Contrastive Language–Image Pre-training
When prompts are provided, CLIP performs zero-shot classification of each mask. The best label among the prompts is attached to the mask and used to filter the results.

## BLIP‑3 – Verification
After CLIP, cropped masks can optionally be verified with the BLIP‑3 VQA model.
Each mask is questioned using a yes/no prompt. If the model's answer contains the
configured false string, the mask is re-labelled as `negative` and discarded.

## Post‑Segmentation Filtering
After SAM2, masks can be discarded based on area or bounding-box size. This keeps only relevant regions before classification or visualization.

## Geometry (Canny + Hough, legacy-only)
The repository retains Canny/Hough helpers for trusted legacy integrations, but
the canonical in-memory core does not execute this stage. The current service
rejects geometry before inference; future activation requires a governed
scientific-stage order and an in-memory refactor of the file-writing helper.

## Visualization
The core can return configured in-memory overlays. The service supports only
bounded `annotated` streams at L3; panoptic and file/video writers remain
legacy-only capabilities.
