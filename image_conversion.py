import os
import glob
import cv2
import numpy as np
from PIL import Image


def remove_background_with_grabcut(image_path):
    # Load image
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Initialize mask
    mask = np.zeros(image_rgb.shape[:2], np.uint8)

    # Temporary arrays for grabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define a rectangle around the assumed foreground area
    # For better results, this needs to be defined per image or through user interaction
    height, width = image_rgb.shape[:2]
    rectangle = (int(width * 0.1), int(height * 0.1), int(width * 0.8), int(height * 0.8))

    # Apply GrabCut algorithm
    cv2.grabCut(image_rgb, mask, rectangle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to ensure the background is properly marked
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the image to remove the background
    image_rgb_nobg = image_rgb * mask[:, :, np.newaxis]

    # Convert to 4-channel RGBA image
    image_rgba = cv2.cvtColor(image_rgb_nobg, cv2.COLOR_RGB2RGBA)
    image_rgba[:, :, 3] = mask * 255

    return image_rgba


def remove_background_and_resize(folder_path, output_folder, target_size, interpolation_method):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_path in glob.glob(os.path.join(folder_path, '*.[jJ][pP][gG]')) + glob.glob(
            os.path.join(folder_path, '*.png')):
        filename = os.path.basename(file_path)

        # Remove background from image using GrabCut algorithm
        image_rgba = remove_background_with_grabcut(file_path)

        # Convert to PIL Image for resizing
        image_pil = Image.fromarray(image_rgba)

        # Resize image
        resized_image = image_pil.resize(target_size, resample=interpolation_method)
        output_file_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_resized.png')
        resized_image.save(output_file_path, 'PNG')


if __name__ == '__main__':
    folder_path = './input_folder'
    output_folder = './output_folder'
    target_size = (64, 64)
    interpolation_method = Image.NEAREST  # Use nearest-neighbor interpolation method

    remove_background_and_resize(folder_path, output_folder, target_size, interpolation_method)
