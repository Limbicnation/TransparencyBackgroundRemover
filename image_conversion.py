import os
import glob
import cv2
import numpy as np
from PIL import Image
import argparse
from typing import Tuple, Optional, List
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PixelArtProcessor:
    def __init__(self, 
                 tolerance: int = 30,
                 edge_sensitivity: float = 0.8,
                 color_clusters: int = 8):
        """
        Initialize the Enhanced Pixel Art Processor.
        
        Args:
            tolerance: Color similarity threshold (0-255)
            edge_sensitivity: Edge detection sensitivity (0-1)
            color_clusters: Number of color clusters for background detection
        """
        self.tolerance = tolerance
        self.edge_sensitivity = edge_sensitivity
        self.color_clusters = color_clusters
        
    def detect_background_color_advanced(self, image: np.ndarray) -> Tuple[int, int, int]:
        """
        Advanced background color detection using K-means clustering and edge analysis.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of RGB values representing background color
        """
        # Sample pixels from image edges
        border_pixels = []
        h, w = image.shape[:2]
        
        # Sample from all edges, not just corners
        border_pixels.extend(image[0, :])  # Top edge
        border_pixels.extend(image[-1, :])  # Bottom edge
        border_pixels.extend(image[1:-1, 0])  # Left edge
        border_pixels.extend(image[1:-1, -1])  # Right edge
        
        border_pixels = np.array(border_pixels)
        
        # Use K-means to find most common color clusters
        kmeans = KMeans(n_clusters=min(self.color_clusters, len(border_pixels)))
        kmeans.fit(border_pixels[:, :3])
        
        # Find largest cluster
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        background_cluster = unique[np.argmax(counts)]
        
        # Get the center of the largest cluster
        background_color = tuple(map(int, kmeans.cluster_centers_[background_cluster]))
        
        return background_color
    
    def detect_pixel_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced edge detection optimized for pixel art.
        
        Args:
            image: Input image
            
        Returns:
            Binary edge mask
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Apply multiple edge detection methods
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize and threshold
        sobel = np.uint8(255 * sobel / np.max(sobel))
        _, edges_sobel = cv2.threshold(sobel, 127 * self.edge_sensitivity, 255, cv2.THRESH_BINARY)
        
        # Combine with Canny edges for better detail
        edges_canny = cv2.Canny(gray, 100 * self.edge_sensitivity, 200 * self.edge_sensitivity)
        
        # Combine edge detection methods
        edges = cv2.bitwise_or(edges_sobel, edges_canny)
        
        # Clean up edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges

    def remove_background_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced background removal using color clustering and intelligent edge detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            RGBA image with transparent background
        """
        # Detect background color using clustering
        bg_color = self.detect_background_color_advanced(image)
        
        # Create initial alpha mask using color distance
        alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Vectorized color distance calculation
        color_distances = np.sqrt(np.sum((image[:, :, :3].astype(float) - 
                                        np.array(bg_color).astype(float))**2, axis=2))
        
        # Create color-based mask
        alpha = np.where(color_distances < self.tolerance, 0, 255).astype(np.uint8)
        
        # Get edge mask
        edges = self.detect_pixel_edges(image)
        
        # Combine masks with edge priority
        final_mask = np.maximum(alpha, edges)
        
        # Create RGBA image
        rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = final_mask
        
        return rgba

    def optimize_pixel_scaling(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Optimized scaling for pixel art that preserves pixel sharpness.
        
        Args:
            image: PIL Image
            target_size: Desired output size
            
        Returns:
            Scaled PIL Image
        """
        # Calculate optimal intermediate size
        current_w, current_h = image.size
        target_w, target_h = target_size
        
        # Find the scale factor that's a power of 2
        scale_x = target_w / current_w
        scale_y = target_h / current_h
        
        # Use two-step scaling for better quality
        if scale_x > 1 or scale_y > 1:
            # First upscale to power of 2
            power_2_scale = 2 ** np.floor(np.log2(min(scale_x, scale_y)))
            if power_2_scale > 1:
                intermediate_size = (
                    int(current_w * power_2_scale),
                    int(current_h * power_2_scale)
                )
                image = image.resize(intermediate_size, Image.NEAREST)
        
        # Final resize to target size
        return image.resize(target_size, Image.NEAREST)

    def process_image(self, 
                     input_path: str, 
                     output_path: str, 
                     target_size: Tuple[int, int]) -> None:
        """
        Process a single image with enhanced features.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            target_size: Desired output size
        """
        try:
            logger.info(f"Processing {input_path}")
            
            # Load image
            image_bgr = cv2.imread(input_path)
            if image_bgr is None:
                raise ValueError(f"Could not load image: {input_path}")
                
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Remove background
            logger.info("Removing background...")
            image_rgba = self.remove_background_advanced(image_rgb)
            
            # Convert to PIL Image
            image_pil = Image.fromarray(image_rgba)
            
            # Optimize scaling
            logger.info("Optimizing pixel scaling...")
            resized_image = self.optimize_pixel_scaling(image_pil, target_size)
            
            # Save with maximum quality
            logger.info(f"Saving to {output_path}")
            resized_image.save(output_path, 'PNG', optimize=True)
            
            logger.info("Processing complete")
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
            raise

def process_folder(input_folder: str, 
                  output_folder: str, 
                  target_size: Tuple[int, int],
                  tolerance: int = 30,
                  edge_sensitivity: float = 0.8) -> None:
    """
    Process all images in a folder with enhanced features.
    
    Args:
        input_folder: Path to input folder
        output_folder: Path to output folder
        target_size: Desired output size
        tolerance: Color similarity threshold
        edge_sensitivity: Edge detection sensitivity
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output directory: {output_folder}")
    
    processor = PixelArtProcessor(
        tolerance=tolerance,
        edge_sensitivity=edge_sensitivity
    )
    
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(input_folder, pattern)))
    
    if not image_files:
        logger.warning(f"No images found in {input_folder}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    for file_path in image_files:
        try:
            filename = os.path.basename(file_path)
            output_path = os.path.join(
                output_folder, 
                f'{os.path.splitext(filename)[0]}_processed.png'
            )
            
            logger.info(f"Processing {filename}...")
            processor.process_image(file_path, output_path, target_size)
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced pixel art processor.')
    parser.add_argument('--input_folder', type=str, required=True,
                      help='Path to input folder')
    parser.add_argument('--output_folder', type=str, required=True,
                      help='Path to output folder')
    parser.add_argument('--target_size', type=int, nargs=2, default=[64, 64],
                      help='Target size (width height)')
    parser.add_argument('--tolerance', type=int, default=30,
                      help='Color similarity threshold (0-255)')
    parser.add_argument('--edge_sensitivity', type=float, default=0.8,
                      help='Edge detection sensitivity (0-1)')

    args = parser.parse_args()

    try:
        process_folder(
            args.input_folder,
            args.output_folder,
            tuple(args.target_size),
            args.tolerance,
            args.edge_sensitivity
        )
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise