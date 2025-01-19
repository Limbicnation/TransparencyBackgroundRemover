# TransparencyBackgroundRemover

A Python script optimized for removing backgrounds from pixel art and regular images while preserving transparency. Features advanced edge detection, intelligent color clustering, and pixel-perfect scaling.

## Features

- Advanced background detection using K-means clustering
- Enhanced edge detection combining Sobel and Canny methods
- Two-step pixel scaling for optimal quality
- Detailed logging and error handling
- Optimized for pixel art preservation

## Installation

```bash
# Create and activate environment
conda create --name bgremover python=3.11
conda activate bgremover

# Install dependencies
pip install Pillow numpy opencv-python scikit-learn
```

## Usage

Basic usage with default settings:
```bash
python image_conversion.py --input_folder ./input_folder --output_folder ./output --target_size 512 512
```

All options:
```bash
python image_conversion.py \
  --input_folder ./input_folder \
  --output_folder ./output \
  --target_size 512 512 \
  --tolerance 30 \
  --edge_sensitivity 0.8
```

## Parameters

- `--target_size`: Output dimensions as width height (default: 64 64)
- `--tolerance`: Color similarity threshold 0-255 (default: 30, lower = stricter)
- `--edge_sensitivity`: Edge detection sensitivity 0-1 (default: 0.8, higher = more edges)

## Advanced Features

- **Intelligent Background Detection**: Uses K-means clustering to identify dominant background colors
- **Enhanced Edge Detection**: Combines multiple methods for accurate edge preservation
- **Optimal Scaling**: Two-step process for maintaining pixel art quality
- **Progress Tracking**: Detailed logging of processing steps and any issues

## Tips

- For pixel art with complex backgrounds: Adjust edge_sensitivity up to 0.9
- For cleaner edges: Lower the tolerance value (20-30)
- For upscaling: Use power-of-2 target sizes (256, 512, 1024) for best results
- Supports common image formats (PNG, JPG, JPEG)
- Outputs optimized PNG files with transparency

## Example

Input folder structure:
```
input_folder/
  ├── sprite1.png
  ├── sprite2.jpg
  └── character.png
```

Run command:
```bash
python image_conversion.py --input_folder ./input_folder --output_folder ./output --target_size 512 512 --tolerance 30 --edge_sensitivity 0.8
```

## Error Handling

The script provides detailed logging for troubleshooting:
- Input file validation
- Processing progress
- Background removal steps
- Scaling operations
- Any errors encountered