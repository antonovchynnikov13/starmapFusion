# starmapFusion

Computer vision project for autonomous drone navigation using stars visible in night-sky images.

The system detects stars in images captured by an onboard camera, matches observed star patterns with a reference star catalogue, estimates the drone or camera orientation, and provides an additional navigation source when GNSS is unavailable, unreliable, or jammed.

## Main Goals

- Detect stars in low-quality night-sky images.
- Remove noise, clouds, motion blur, and false detections.
- Match observed star patterns with a reference star catalogue.
- Estimate camera orientation and attitude.
- Fuse stellar navigation with IMU and other onboard sensors.
- Evaluate navigation accuracy under different environmental conditions.

## Development

The project is developed locally in VS Code with Codex.

Model training and computationally intensive experiments are performed in Google Colab using CUDA.

Reusable logic should be implemented in Python modules. Jupyter notebooks are used only for experiments, visualization, and model training.

## LAMOST-DET preparation

Keep the large dataset outside the repository and expose it through the ignored
`data/lamost` symlink or the `LAMOST_ROOT` environment variable. Build lightweight
JSONL manifests without copying any image data:

```bash
python3 scripts/prepare_lamost.py
```

For a smaller first experiment, select deterministic subsets:

```bash
python3 scripts/prepare_lamost.py \
  --max-train 3000 \
  --max-validation 500 \
  --max-test 500 \
  --seed 42
```

The command preserves the official LAMOST splits, maps `dev` to the
`validation.jsonl` manifest, validates image/annotation pairs, and converts each
ellipse from `gt_norm` into a `(center_x, center_y)` point. Generated manifests
are written under `data/processed/lamost/` and are not tracked by Git.

### Export a compact Colab dataset

Copy only images referenced by the subset manifests into a portable directory:

```bash
python3 scripts/export_lamost_subset.py \
  --dataset-root data/lamost \
  --manifest-dir data/processed/lamost \
  --output-root /media/mykyta/01DC91811D38E970/LAMOST_subset
```

Upload `LAMOST_subset` to Google Drive. This avoids uploading the full dataset.

### CUDA training in Google Colab

Open `notebooks/train_star_detector_colab.ipynb` in Colab, select a T4 GPU,
mount Google Drive, set the dataset path, and run the cells in order. The
reusable training entry point is:

```bash
python3 scripts/train_star_detector.py \
  --dataset-root /content/drive/MyDrive/datasets/LAMOST_subset \
  --manifest-dir /content/drive/MyDrive/datasets/LAMOST_subset/manifests \
  --output-dir /content/drive/MyDrive/starmapFusion_outputs/resnet50_fpn_heatmap \
  --epochs 30 \
  --batch-size 2 \
  --workers 2 \
  --image-size 1024
```

Training is intentionally CUDA-only. It uses automatic mixed precision, AdamW,
a cosine learning-rate schedule, gradient clipping, and writes `best.pt`,
`last.pt`, and `history.json` after validation.

## Project Structure

- `src/` – source code
- `tests/` – automated tests
- `notebooks/` – experiments and Colab notebooks
- `configs/` – configuration files
- `scripts/` – utility scripts
- `data/` – local datasets (not tracked by Git)
- `outputs/` – checkpoints, predictions, metrics, and visualizations (not tracked by Git)

## Planned Pipeline

1. Load images and metadata.
2. Preprocess night-sky images.
3. Detect stars.
4. Filter false detections.
5. Match star patterns with a catalogue.
6. Estimate camera orientation.
7. Fuse stellar navigation with IMU and other sensors.
8. Evaluate navigation performance.

## Technology

- Python
- PyTorch
- OpenCV
- NumPy
- pandas
- scikit-learn
- Astropy
- Google Colab (CUDA)
