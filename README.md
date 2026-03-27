# JPEG Restorer - Diffusion-based Image Enhancement

A GUI application for restoring JPEG-compressed images using a diffusion model approach. The tool reduces compression artifacts and enhances image quality through iterative denoising steps.

## Features

- **Diffusion-based Restoration**: Uses a 4-8 step diffusion process to remove JPEG artifacts
- **Interactive GUI**: Built with PyQt6 for intuitive parameter tuning
- **Checkpoint Management**: Easy loading and browsing of model checkpoints
- **Tiling Support**: Process large images with memory-efficient tiling
- **Test-Time Augmentation (TTA)**: Improves robustness through flipping ensembles
- **Quality Control**: Adjustable JPEG quality factor and noise strength
- **Multiple Passes**: Ensemble averaging for improved results
- **Visualization Tools**: 
  - Side-by-side original/restored comparison
  - Gate map visualization (artifact heatmap)
  - Real-time progress logging

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
   Or run:
   ```bash
   run.bat
   ```

2. Click "Run Inference" to process your image
3. Monitor progress in the log and view results in the preview panels

## Model Checkpoints

The application looks for `.pt` checkpoint files in:
- Current directory
- `./runs` subdirectory
- Parent directory's `./runs` folder

Checkpoints should contain:
- Model state dictionary
- Model configuration (base_channels, emb_dim, depth)
- Training arguments
- Optional EMA weights

## Output

- Restored image saved as specified filename
- Optional comparison image (`*_comparison.ext`) showing original vs restored
- Gate map visualization available in GUI showing artifact regions

## Technical Details

The restoration process uses:
- U-Net architecture with residual blocks
- Time and quality conditioning
- Diffusion sampling with configurable steps
- Tiled processing for large images
- Stacking averaging with quality jitter
- Optional augmentation via flipping (TTA)
  
## Recommended Settings
- Steps: 4-8
- Noise 0.4-0.5

## Requirements

See `requirements.txt` for exact versions, but core dependencies include:
- PyTorch
- Torchvision
- Pillow
- PyQt6
- NumPy

## License

This project is provided as-is for educational and experimental purposes.
```

To update the README, you'll need to copy this content into your README.md file manually, as I'm currently unable to make file modifications due to the read-only constraint.
