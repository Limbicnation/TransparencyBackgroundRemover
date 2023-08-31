import os
import glob
from PIL import Image
import numpy as np
import cv2


def remove_background_and_resize(folder_path, output_folder, target_size, keyout_color):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_path in glob.glob(os.path.join(folder_path, '*.[jJ][pP]*[gG]')) + glob.glob(
            os.path.join(folder_path, '*.png')):
        filename = os.path.basename(file_path)

        with Image.open(file_path) as image:
            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            image_array = np.array(image)

            # Compute a binary mask for the keyout color.
            distance = np.sum(np.abs(image_array[:, :, :3] - keyout_color[:3]), axis=2)
            mask = (distance < 30).astype(np.uint8) * 255  # Adjust the threshold as needed

            # Set the alpha channel to 0 for the keyout_color
            image_array[mask == 255, 3] = 0
            result_image = Image.fromarray(image_array)

            resized_image = result_image.resize(target_size, resample=Image.LANCZOS)
            output_file_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.png')
            resized_image.save(output_file_path, 'PNG')


if __name__ == '__main__':
    folder_path = './input'
    output_folder = './output'
    target_size = (64, 64)
    keyout_color = (65, 62, 67)  # RGB for 413e43

    remove_background_and_resize(folder_path, output_folder, target_size, keyout_color)
