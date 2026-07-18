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
