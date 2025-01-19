import os
import glob
import cv2
import numpy as np
from PIL import Image
import argparse
from typing import Tuple, Optional

class PixelArtProcessor:
    def __init__(self, tolerance: int = 30):
        """
        Initialize the PixelArtProcessor with a color tolerance value.
        
        Args:
            tolerance: Color similarity threshold (0-255)
        """
        self.tolerance = tolerance

    def detect_background_color(self, image: np.ndarray) -> Tuple[int, int, int]:
        """
        Detect the most likely background color by sampling corners.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of RGB values representing background color
        """
        corners = [
            image[0, 0],
            image[0, -1],
            image[-1, 0],
            image[-1, -1]
        ]
        # Use the most common corner color as background
        corner_colors = [tuple(color[:3]) for color in corners]
        return max(set(corner_colors), key=corner_colors.count)

    def color_distance(self, color1: np.ndarray, color2: Tuple[int, int, int]) -> float:
        """
        Calculate Euclidean distance between two colors.
        
        Args:
            color1: First color (numpy array)
            color2: Second color (tuple)
            
        Returns:
            Float representing color distance
        """
        return np.sqrt(np.sum((color1[:3] - np.array(color2)) ** 2))

    def remove_background_color_based(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background using color similarity.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            RGBA image with transparent background
        """
        bg_color = self.detect_background_color(image)
        
        # Create alpha channel
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if self.color_distance(image[y, x], bg_color) < self.tolerance:
                    alpha[y, x] = 0
                else:
                    alpha[y, x] = 255
                    
        # Create RGBA image
        rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = alpha
        
        return rgba

    def detect_edges_pixel_art(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges optimized for pixel art.
        
        Args:
            image: Input image
            
        Returns:
            Binary edge mask
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges

    def remove_background_hybrid(self, image_path: str) -> np.ndarray:
        """
        Remove background using both color and edge detection.
        
        Args:
            image_path: Path to input image
            
        Returns:
            RGBA image with transparent background
        """
        # Load image
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Get color-based mask
        color_mask = self.remove_background_color_based(image_rgb)[:, :, 3]
        
        # Get edge-based mask
        edges = self.detect_edges_pixel_art(image_rgb)
        
        # Combine masks
        final_mask = np.maximum(color_mask, edges)
        
        # Apply mask to image
        rgba = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = final_mask
        
        return rgba

    @staticmethod
    def nearest_neighbor_resize(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Resize image using nearest neighbor interpolation with pixel-perfect scaling.
        
        Args:
            image: PIL Image
            target_size: Desired output size
            
        Returns:
            Resized PIL Image
        """
        return image.resize(target_size, Image.NEAREST)

    def process_image(self, 
                     input_path: str, 
                     output_path: str, 
                     target_size: Tuple[int, int],
                     method: str = 'hybrid') -> None:
        """
        Process a single image.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            target_size: Desired output size
            method: Background removal method ('hybrid', 'color', or 'grabcut')
        """
        # Remove background
        if method == 'color':
            image_bgr = cv2.imread(input_path)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgba = self.remove_background_color_based(image_rgb)
        elif method == 'grabcut':
            image_rgba = remove_background_with_grabcut(input_path)
        else:  # hybrid
            image_rgba = self.remove_background_hybrid(input_path)
        
        # Convert to PIL Image
        image_pil = Image.fromarray(image_rgba)
        
        # Resize
        resized_image = self.nearest_neighbor_resize(image_pil, target_size)
        
        # Save
        resized_image.save(output_path, 'PNG')

def remove_background_with_grabcut(image_path: str) -> np.ndarray:
    """Original GrabCut implementation (kept for comparison)"""
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    mask = np.zeros(image_rgb.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    height, width = image_rgb.shape[:2]
    rectangle = (int(width * 0.1), int(height * 0.1), 
                int(width * 0.8), int(height * 0.8))
    
    cv2.grabCut(image_rgb, mask, rectangle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    image_rgb_nobg = image_rgb * mask[:, :, np.newaxis]
    image_rgba = cv2.cvtColor(image_rgb_nobg, cv2.COLOR_RGB2RGBA)
    image_rgba[:, :, 3] = mask * 255
    
    return image_rgba

def process_folder(input_folder: str, 
                  output_folder: str, 
                  target_size: Tuple[int, int],
                  method: str = 'hybrid',
                  tolerance: int = 30) -> None:
    """
    Process all images in a folder.
    
    Args:
        input_folder: Path to input folder
        output_folder: Path to output folder
        target_size: Desired output size
        method: Background removal method
        tolerance: Color similarity threshold
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    processor = PixelArtProcessor(tolerance=tolerance)
    
    for file_path in glob.glob(os.path.join(input_folder, '*.[jJ][pP][gG]')) + \
                     glob.glob(os.path.join(input_folder, '*.[pP][nN][gG]')):
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, 
                                 f'{os.path.splitext(filename)[0]}_processed.png')
        
        processor.process_image(file_path, output_path, target_size, method)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process pixel art images.')
    parser.add_argument('--input_folder', type=str, default='./input_folder',
                      help='Path to input folder')
    parser.add_argument('--output_folder', type=str, default='./output_folder',
                      help='Path to output folder')
    parser.add_argument('--target_size', type=int, nargs=2, default=[64, 64],
                      help='Target size (width height)')
    parser.add_argument('--method', type=str, default='hybrid',
                      choices=['hybrid', 'color', 'grabcut'],
                      help='Background removal method')
    parser.add_argument('--tolerance', type=int, default=30,
                      help='Color similarity threshold (0-255)')

    args = parser.parse_args()

    process_folder(
        args.input_folder,
        args.output_folder,
        tuple(args.target_size),
        args.method,
        args.tolerance
    )