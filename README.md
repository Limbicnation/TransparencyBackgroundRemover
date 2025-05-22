# TransparencyBackgroundRemover

A Python script optimized for removing backgrounds from pixel art and regular images while preserving transparency. Features perceptual color processing, advanced edge detection, intelligent dithering detection, and pixel-perfect scaling.

## Features

- Advanced background detection using K-means clustering with spatial weighting
- Enhanced edge detection with dedicated dithering pattern recognition
- Content-aware parameter auto-tuning
- Two-step pixel scaling optimized for both upscaling and downscaling
- Multi-threaded processing with progress tracking
- Preview mode for quick testing

## Installation

```bash
# Create and activate environment
conda create --name bgremover python=3.11
conda activate bgremover

# Install dependencies
pip install Pillow numpy opencv-python scikit-learn tqdm
```

## Usage

Basic usage with default settings:
```bash
python image_conversion.py --input_folder ./input_folder --output_folder ./output --target_size 512 512
```

Preview a single image:
```bash
python image_conversion.py --preview ./input_folder/sprite.png --output_folder ./preview
```

All options:
```bash
python image_conversion.py \
  --input_folder ./input_folder \
  --output_folder ./output \
  --target_size 512 512 \
  --tolerance 30 \
  --edge_sensitivity 0.8 \
  --foreground_bias 0.7 \
  --edge_refinement \
  --dither_handling \
  --num_workers 4
```

## Parameters

- `--target_size`: Output dimensions as width height (default: 64 64)
- `--tolerance`: Color similarity threshold 0-255 (default: 30, lower = stricter)
- `--edge_sensitivity`: Edge detection sensitivity 0-1 (default: 0.8, higher = more edges)
- `--foreground_bias`: Bias towards foreground preservation 0-1 (default: 0.7)
- `--edge_refinement`: Apply edge refinement (default: enabled)
- `--dither_handling`: Apply dithering pattern detection (default: enabled)
- `--num_workers`: Number of parallel processing threads (default: 4)
- `--preview`: Create preview for single image instead of processing folder

## Tips

- For pixel art with complex backgrounds: Increase edge_sensitivity to 0.9
- For dithered pixel art: Ensure dither_handling is enabled
- For photos with complex subjects: Increase foreground_bias to 0.8
- For batch processing: Adjust num_workers based on your CPU cores
