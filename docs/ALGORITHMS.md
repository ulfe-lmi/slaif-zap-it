# Algorithms Used

ZAP-IT combines several well known algorithms and models to form a complete pipeline.

## SAM2 – Segment Anything Model 2
SAM2 generates segmentation masks given an input image. The pipeline can run it in a single pass or over overlapping tiles for large images. Parameters such as point density and IoU thresholds are configurable in the YAML file.

## CLIP – Contrastive Language–Image Pre-training
When prompts are provided, CLIP performs zero-shot classification of each mask. The best label among the prompts is attached to the mask and used to filter the results.

## Post‑Segmentation Filtering
After SAM2, masks can be discarded based on area or bounding-box size. This keeps only relevant regions before classification or visualization.

## Geometry (Canny + Hough)
If enabled, each final mask can be analysed with Canny edge detection followed by a probabilistic Hough transform. The resulting lines and intersections are stored in TSV files and can be drawn on the output images.

## Visualization
Composites showing intermediate SAM2 results, CLIP-filtered masks and final panoptic overlays are generated using `matplotlib` and `detectron2` utilities.
