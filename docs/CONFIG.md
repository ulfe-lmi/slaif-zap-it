# Configuration Files

Each run of ZAP-IT is driven by a YAML configuration file. The examples in `configs/` cover typical use cases. A configuration is composed of several sections:

## preprocessing
Defines optional region-of-interest cropping and resizing. Example:
```yaml
preprocessing:
  roi: "0,1500,4000,1500"  # x,y,w,h or False for full image
  resize: 1.0              # scale factor for ROI; omit to run at native size
  debug: true              # save ROI debug image
```

## mask_generator
Parameters for the SAM2 automatic mask generator such as grid density and IoU thresholds.

## postsam2processing
Filters masks by area and bounding-box size after segmentation.

## clip (optional)
Zero-shot classification prompts for CLIP. Prompts are grouped under `labels` with each key being the desired class name.

```yaml
clip:
  padding: 40
  debug: true
  labels:
    goat: |
      a Boer goat in a grassy field,
      a white goat with reddish ears
    sign: |
      a white rectangular sign with a black number
```

## blip3 (optional)
Zero-shot verification with the BLIP-3 VQA model. Each key matches a CLIP label
and provides a question along with substrings representing the true and false
answers. Masks answering with the false string are re-labelled as `negative`.
If `newcategory` is set, a positive answer overwrites the label with that value.
Keys starting with `any,` treat the number after the comma as a CLIP score
threshold. When a mask's CLIP score is below that threshold the associated
question is asked and the label can be changed to `newcategory` based on the
answer.

```yaml
blip3:
  goat:
    question: "Is there an animal in the image? Yes or no!"
    trueresult: "Yes"
    falseresult: "No"
    newcategory: goat
    debug: true

  sign:
    question: "Is there anything that could be interpreted as a white sign in the image? Yes or no!"
    trueresult: "Yes"
    falseresult: "No"
    newcategory: sign
    debug: true

  any,0.1:
    question: "Is there an animal in the image? Yes or no!"
    trueresult: "Yes"
    falseresult: "No"
    newcategory: goat
    debug: true
```

## geometry (optional)
Canny and Hough settings for line detection if geometry analysis is enabled.

## visualization
Declares which mask stages should be rendered by the visualizer. Each stage (`sam2`, `clip`, `blip3`) accepts a list of
entries with an `id`, a `renderer`, and optional renderer arguments. The resulting RGB arrays are keyed by `id` and can
be consumed by the `images` and `video` sections described below.

```yaml
visualization:
  labels: ["goat", "fencepost"]        # whitelist for the final mask filter
  alpha: 0.75                           # default alpha for alpha-overlay renderer
  sam2:
    - id: sam2-goat-alpha               # visualization identifier
      renderer: alpha-overlay           # available: alpha-overlay, panoptic
      alpha: 0.75                       # overrides the default alpha for this entry
  clip:
    - id: clip-goat-alpha
      renderer: alpha-overlay
  blip3:
    - id: clip-panoptic
      renderer: panoptic
```

### images
Associates visualization IDs with output directories. Each frame is written as a seven-digit JPEG sequence
(`0000001.jpg`, `0000002.jpg`, …).

```yaml
images:
  sam2-goat-alpha: goats-sam2
  clip-goat-alpha: goats-clip
```

### video
Associates visualization IDs with MJPEG AVI files. Provide a filename or a mapping with additional parameters such as
`fps`.

```yaml
video:
  clip-panoptic:
    filename: goats-clip.avi
    fps: 24
```

If a section is omitted the corresponding writer becomes a no-op.
