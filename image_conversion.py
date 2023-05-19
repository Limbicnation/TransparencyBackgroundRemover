import os
from PIL import Image
import numpy as np
import cv2

# Set the folder path containing the images
folder_path = 'F:/GitHub/Python/input'

# Set the output folder path where the PNG files will be saved
output_folder = 'F:/GitHub/Python/output'

# Set the desired image size after background removal
target_size = (64, 64)

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
        
        # Convert the image to RGB mode if it's not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Remove the background using OpenCV
        image_array = np.array(image)
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masked_image = cv2.bitwise_and(image_array, image_array, mask=mask)
        result_image = Image.fromarray(masked_image)
        
        # Resize the image to the target size
        resized_image = result_image.resize(target_size, resample=Image.NEAREST)
        
        # Create the output file path
        output_file_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.png')
        
        # Save the resized image as PNG with the nearest neighbor resampling
        resized_image.save(output_file_path, 'PNG', resample=Image.NEAREST)
        
        # Close the image files
        image.close()
        result_image.close()
        resized_image.close()
