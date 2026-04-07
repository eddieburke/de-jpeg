# JPEG Restorer - Diffusion-based Image Enhancement

A PyQt6 GUI application for restoring JPEG-compressed images using an EDM diffusion model. Reduces compression artifacts and enhances image quality through iterative denoising with quality conditioning.

## Features

- **GUI Interface**: Full PyQt6-based application with live preview
- **Batch Processing**: Process individual files or entire folders
- **Auto Quality Detection**: Automatically detects JPEG quality from input files
- **Tiled Inference**: Memory-efficient processing for large images
- **Test-Time Augmentation (TTA)**: Optional horizontal/vertical flipping for improved results
- **Ensemble Passes**: Multiple restoration passes with quality jitter for better output
- **EMA Weight Support**: Optional exponential moving average weights from training

## Installation

1. Clone or download this repository
2. Ensure you have Python 3.8+ installed
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or run the provided install script:
   ```bash
   install.bat
   ```

## Usage

1. Launch the application:
   ```bash
   python main.py
   ```
   Or use the provided script:
   ```bash
   run.bat
   ```

2. Select a model checkpoint (auto-scans `./runs` and parent directories)
3. Choose input file(s) or folder for batch processing
4. Configure settings (quality, tile size, passes, TTA, etc.)
5. Click "Run Inference" to process
6. View results in the preview panels

## Model Checkpoints

The application automatically scans for `.pt` checkpoint files in:
- Current directory
- `./runs` subdirectory
- Parent directory's `./runs` folder

## Settings

| Setting | Description |
|---------|-------------|
| Quality | Target quality for restoration (0-100) |
| Tile Size | Image tile size for memory-efficient processing (0 = disabled) |
| Overlap | Tile overlap for blending (default: 32) |
| Batch Size | Number of tiles processed simultaneously |
| Passes | Number of ensemble passes with quality jitter |
| TTA | Enable test-time augmentation (flipping) |
| Use EMA | Use exponential moving average weights |

## Output

- Restored image saved to the specified output folder
- Original filename preserved with `_restored` suffix when batch processing

## Technical Details

The restoration process uses:
- **EDM (Elucidating the Design Space of Diffusion Models)**: Sampler with quality conditioning
- **U-Net Architecture**: Residual blocks with time and quality embeddings
- **Tiled Processing**: Splits large images into tiles for memory efficiency
- **Ensemble Averaging**: Multiple passes with small quality variations
- **Test-Time Augmentation**: Horizontal and vertical flips averaged for better results

## Requirements

Core dependencies (see `requirements.txt` for exact versions):
- PyTorch 2.0+
- PyQt6
- Pillow
- NumPy

## License

This project is provided as-is for educational and experimental purposes.