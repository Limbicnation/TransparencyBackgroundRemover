# TransparencyBackgroundRemover

A Python script optimized for removing backgrounds from pixel art and regular images while preserving transparency. Supports multiple background removal methods and pixel-perfect scaling.

## Installation

```bash
# Create and activate environment
conda create --name bgremover python=3.11
conda activate bgremover

# Install dependencies
pip install Pillow numpy opencv-python
```

## Usage

Basic usage with default settings:
```bash
python script.py --input_folder ./input --output_folder ./output
```

All options:
```bash
python script.py \
  --input_folder ./input \
  --output_folder ./output \
  --target_size 64 64 \
  --method hybrid \
  --tolerance 30
```

## Parameters

- `--method`: Background removal method (`hybrid`, `color`, or `grabcut`)
  - `hybrid`: Best for pixel art (combines color and edge detection)
  - `color`: Good for solid backgrounds
  - `grabcut`: Better for photographs
- `--target_size`: Output dimensions as width height (default: 64 64)
- `--tolerance`: Color similarity threshold 0-255 (default: 30, lower = stricter)

## Tips

- For pixel art: Use `hybrid` method with lower tolerance (20-30)
- For photos: Use `grabcut` method
- If background isn't fully removed: Adjust tolerance or try different method
- Supports both JPG and PNG formats
- Outputs PNG files with transparency