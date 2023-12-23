import os
import glob
from PIL import Image
import numpy as np

def remove_background_and_resize(folder_path, output_folder, target_size, keyout_color, interpolation_method):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_path in glob.glob(os.path.join(folder_path, '*.[jJ][pP][gG]')) + glob.glob(os.path.join(folder_path, '*.png')):
        filename = os.path.basename(file_path)

        with Image.open(file_path) as image:
            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            image_array = np.array(image)

            # Compute a binary mask for the keyout color.
            distance = np.sum(np.abs(image_array[:, :, :3] - np.array(keyout_color)[:3]), axis=2)
            mask = (distance < 30).astype(np.uint8) * 255  # Adjust the threshold as needed

            # Set the alpha channel to 0 for the keyout_color
            image_array[mask == 255, 3] = 0
            result_image = Image.fromarray(image_array)

            resized_image = result_image.resize(target_size, resample=interpolation_method)
            output_file_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_resized.png')
            resized_image.save(output_file_path, 'PNG')

if __name__ == '__main__':
    folder_path = '/mnt/Qsync_Ubuntu/Qsync/53_PixelQuest/6_Output/2_Images/PixelArtAvatars'
    output_folder = '/mnt/Qsync_Ubuntu/Qsync/53_PixelQuest/6_Output/2_Images/PixelArtAvatars_Output/'
    target_size = (64, 64)
    keyout_color = (117, 87, 100)  # RGB for 755764
    interpolation_method = Image.NEAREST  # Use nearest-neighbor interpolation method

    remove_background_and_resize(folder_path, output_folder, target_size, keyout_color, interpolation_method)
