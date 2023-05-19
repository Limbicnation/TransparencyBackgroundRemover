import os
from PIL import Image
import numpy as np
import cv2

# Set the folder path containing the images
folder_path = 'F:/Path/To/Your/Input/Folder'

# Set the output folder path where the PNG files will be saved
output_folder = '/Path/To/Your/Output/Folder'

# Set the desired image size after background removal
target_size = (64, 64)

# Set the background color (RGB values)
background_color = (255, 255, 255)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    # Check if the file is an image
    if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Open the image file
        image = Image.open(file_path)
        
        # Convert the image to RGBA mode to preserve alpha channel
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Remove the background using OpenCV
        image_array = np.array(image)
        image_alpha = image_array[:, :, 3]  # Extract alpha channel
        _, mask = cv2.threshold(image_alpha, 1, 255, cv2.THRESH_BINARY)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        bg_mask = cv2.inRange(image_array, background_color, background_color)
        bg_mask_rgb = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2RGB)
        masked_image = cv2.bitwise_and(image_array, mask_rgb)
        masked_image = cv2.bitwise_or(masked_image, bg_mask_rgb)
        result_image = Image.fromarray(masked_image)
        
        # Resize the image to the target size
        resized_image = result_image.resize(target_size, resample=Image.LANCZOS)
        
        # Create the output file path
        output_file_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.png')
        
        # Save the resized image as PNG with the Lanczos resampling and preserving transparency
        resized_image.save(output_file_path, 'PNG')
        
        # Close the image files
        image.close()
        result_image.close()
        resized_image.close()
