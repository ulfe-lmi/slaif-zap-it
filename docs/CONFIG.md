# Configuration Files

Each run of ZAP-IT is driven by a YAML configuration file. The examples in `configs/` cover typical use cases. A configuration is composed of several sections:

## preprocessing
Defines optional region-of-interest cropping and resizing. Example:
```yaml
preprocessing:
  roi: "0,1500,4000,1500"  # x,y,w,h or False for full image
  resize: 1.0              # 1.0 => single pass; omit for tiling
  debug: true              # save ROI debug image
```

## mask_generator
Parameters for the SAM2 automatic mask generator such as grid density and IoU thresholds.

## postsam2processing
Filters masks by area and bounding-box size after segmentation.

## tiled
Controls tile size and overlap if tiling is used instead of resizing.

## clip (optional)
Zero-shot classification prompts for CLIP. Each `label "name"` line defines prompts that map to that label.

## geometry (optional)
Canny and Hough settings for line detection if geometry analysis is enabled.

## visualization
Settings for building composite images and panoptic overlays.

Use these sections as building blocks to express your own vision task. See the existing YAML files for concrete values.
