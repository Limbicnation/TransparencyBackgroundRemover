# TransparencyBackgroundRemover

A Python script that removes the background of images while preserving transparency. It uses OpenCV and PIL (Python Imaging Library) to perform background removal and saves the images in PNG format with transparency intact.

## Installation Instructions:

1. Make sure you have Python installed on your system.
2. Create a new Python environment using virtualenv or conda.
3. Activate the created environment.
4. Install the required dependencies by running the following command:

# Installation on Windows and Linux:

Setting Up a Virtual Environment

```bash
conda create --name bgremover python=3.10
conda activate bgremover
```
```bash
pip install Pillow numpy opencv-python
```

With this script, you can specify the ```input folder```, ```output folder```, ```target size```, and ```interpolation method``` as optional command-line arguments when running the script. For example:

```bash
python image_conversion.py --input_folder ./input_folder --output_folder ./output_folder --target_size 1024 1024 --interpolation_method NEAREST
```

