# Star detector v1.0.0

ResNet-50 + FPN heatmap model that returns star-center pixel coordinates.

## Validation

- Best epoch: 7
- Precision: 0.8861
- Recall: 0.7976
- F1: 0.8395
- Matching tolerance: 2 heatmap pixels (8 input pixels)

The model was selected by validation F1. Test metrics are stored in
`test_metrics.json` after the independent test evaluation.

## Independent LAMOST test

- Samples: 500
- Precision: 0.8387
- Recall: 0.7717
- F1: 0.8038
- Confidence threshold: 0.30


## Weights

The inference-ready weights are tracked at
`artifacts/star_detector_resnet50_fpn_v1/weights/best_model_inference_fp16.pt`.
They use FP16 storage and are loaded into the runtime model automatically.

SHA-256: `01e8fd523ccc1bcdf111f5d1385d7503ff5b94396a07d3857950dbb19cba211f`

## Install

```bash
python3 -m pip install -r requirements-inference.txt
```

## Predict coordinates

```bash
python3 scripts/predict_stars.py \
  --checkpoint artifacts/star_detector_resnet50_fpn_v1/weights/best_model_inference_fp16.pt \
  --input photo.jpg \
  --output-dir predictions
```

The command creates `photo.json` with `x`, `y`, and confidence values, plus
`photo_detected.png`. Use the `coordinates` field when only coordinate pairs are
required.

## Known limitation

The model was trained only on LAMOST-DET. Performance on cloudy, green-tinted,
or structurally occluded camera imagery must be evaluated separately.
