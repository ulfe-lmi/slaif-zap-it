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
Settings for building composite images and panoptic overlays.

Use these sections as building blocks to express your own vision task. See the existing YAML files for concrete values.
