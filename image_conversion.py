import os
import glob
import cv2
import numpy as np
from PIL import Image
import argparse
from typing import Tuple, Optional, List, Dict, Union
from sklearn.cluster import KMeans
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import colorsys

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPixelArtProcessor:
    def __init__(self, 
                 tolerance: int = 30,
                 edge_sensitivity: float = 0.8,
                 color_clusters: int = 8,
                 foreground_bias: float = 0.7,
                 edge_refinement: bool = True,
                 dither_handling: bool = True):
        """
        Initialize the Enhanced Pixel Art Processor.
        
        Args:
            tolerance: Color similarity threshold (0-255)
            edge_sensitivity: Edge detection sensitivity (0-1)
            color_clusters: Number of color clusters for background detection
            foreground_bias: Bias towards foreground preservation (0-1)
            edge_refinement: Whether to apply edge refinement
            dither_handling: Whether to handle dithered patterns
        """
        self.tolerance = tolerance
        self.edge_sensitivity = edge_sensitivity
        self.color_clusters = color_clusters
        self.foreground_bias = foreground_bias
        self.edge_refinement = edge_refinement
        self.dither_handling = dither_handling
        
    def _calculate_color_distance(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """
        Calculate perceptual color distance between two colors.
        Uses a weighted combination of RGB and HSV spaces for better perceptual matching.
        
        Args:
            color1, color2: RGB color arrays
            
        Returns:
            Perceptual distance value
        """
        # Convert colors to floats
        c1 = color1.astype(float) / 255.0
        c2 = color2.astype(float) / 255.0
        
        # Calculate RGB Euclidean distance
        rgb_dist = np.sqrt(np.sum((c1 - c2) ** 2))
        
        # Convert to HSV for perceptual distance
        c1_hsv = np.array(colorsys.rgb_to_hsv(c1[0], c1[1], c1[2]))
        c2_hsv = np.array(colorsys.rgb_to_hsv(c2[0], c2[1], c2[2]))
        
        # Handle hue wrapping
        hue_diff = min(abs(c1_hsv[0] - c2_hsv[0]), 1 - abs(c1_hsv[0] - c2_hsv[0]))
        sat_diff = abs(c1_hsv[1] - c2_hsv[1])
        val_diff = abs(c1_hsv[2] - c2_hsv[2])
        
        # Weight the HSV components (hue more important for perception)
        hsv_dist = np.sqrt(0.5 * hue_diff**2 + 0.3 * sat_diff**2 + 0.2 * val_diff**2)
        
        # Combine distances (weight RGB slightly more for pixel art)
        return 0.6 * rgb_dist + 0.4 * hsv_dist
    
    def detect_background_color_advanced(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[int, float]]:
        """
        Advanced background color detection using K-means clustering with spatial weighting.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of background color array and cluster confidence scores
        """
        h, w = image.shape[:2]
        
        # Create spatial weighting mask (higher weight for edges and corners)
        y_coords = np.linspace(0, 1, h)
        x_coords = np.linspace(0, 1, w)
        y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Calculate distance from center
        center_y, center_x = 0.5, 0.5
        dist_from_center = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        edge_weight = np.clip(dist_from_center * 2, 0, 1)
        
        # Sample pixels with focus on edges
        flattened_image = image.reshape(-1, 3)
        flattened_weights = edge_weight.flatten()
        
        # Pixel sampling - use all pixels but weight by edge proximity
        sample_weights = flattened_weights
        
        # Use K-means to find color clusters
        n_clusters = min(self.color_clusters, len(flattened_image))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(flattened_image, sample_weight=sample_weights)
        
        # Get cluster centers and calculate cluster sizes
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Count pixels in each cluster with edge weighting
        cluster_weights = {}
        for i in range(n_clusters):
            cluster_mask = (labels == i)
            # Weight by edge proximity and count
            cluster_weights[i] = np.sum(sample_weights[cluster_mask])
            
        # Normalize weights
        total_weight = sum(cluster_weights.values())
        for k in cluster_weights:
            cluster_weights[k] /= total_weight
            
        # Find cluster with highest edge weight - likely background
        background_cluster = max(cluster_weights, key=cluster_weights.get)
        background_color = centers[background_cluster].astype(int)
        
        # If the background color is too saturated, it might be foreground
        # In that case, take the second-largest cluster
        bg_hsv = colorsys.rgb_to_hsv(*(background_color/255))
        if bg_hsv[1] > 0.5 and len(cluster_weights) > 1:  # If saturation > 50%
            # Remove the current top cluster
            del_cluster = max(cluster_weights, key=cluster_weights.get)
            del cluster_weights[del_cluster]
            # Get the next largest cluster
            background_cluster = max(cluster_weights, key=cluster_weights.get)
            background_color = centers[background_cluster].astype(int)
        
        # Create color confidence dictionary
        color_confidence = {i: cluster_weights[i] for i in range(n_clusters)}
        
        return background_color, color_confidence
    
    def detect_pixel_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced edge detection optimized for pixel art with dithering support.
        
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
        
        # For pixel art, we need both color boundaries and structure
        # Gradient-based edge detection
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize and threshold
        sobel = np.uint8(255 * sobel / np.max(sobel))
        _, edges_sobel = cv2.threshold(
            sobel, 
            int(127 * self.edge_sensitivity), 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Canny for fine details
        edges_canny = cv2.Canny(
            gray, 
            int(100 * self.edge_sensitivity), 
            int(200 * self.edge_sensitivity)
        )
        
        # Combine edge detection methods
        edges = cv2.bitwise_or(edges_sobel, edges_canny)
        
        if self.dither_handling:
            # Special handling for dithered patterns
            # Look for checkerboard-like patterns that indicate dithering
            kernel_dither = np.array([[1, -1], [-1, 1]], dtype=np.float32)
            dither_detect = cv2.filter2D(gray, -1, kernel_dither)
            dither_detect = np.abs(dither_detect)
            _, dither_mask = cv2.threshold(dither_detect, 50, 255, cv2.THRESH_BINARY)
            dither_mask = dither_mask.astype(np.uint8)
            
            # Dilate dithered regions to connect them
            kernel_dilate = np.ones((2, 2), np.uint8)
            dither_mask = cv2.dilate(dither_mask, kernel_dilate, iterations=1)
            
            # Add dithered regions to edges
            edges = cv2.bitwise_or(edges, dither_mask)
        
        # Clean up edges
        if self.edge_refinement:
            # Close small gaps
            kernel_close = np.ones((2, 2), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
            
            # Remove small noise
            kernel_open = np.ones((2, 2), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open)
        
        return edges

    def remove_background_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced background removal using color clustering, intelligent edge detection,
        and foreground bias adjustment.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            RGBA image with transparent background
        """
        # Detect background color and confidence using clustering
        bg_color, color_confidence = self.detect_background_color_advanced(image)
        
        # Create initial alpha mask using perceptual color distance
        h, w = image.shape[:2]
        alpha = np.zeros((h, w), dtype=np.uint8)
        
        # Calculate color distances for all pixels (vectorized)
        # Using our perceptual color distance
        pixel_colors = image.reshape(-1, 3)
        color_distances = np.zeros(len(pixel_colors))
        
        # More efficient vectorized calculation for basic Euclidean distance
        color_distances = np.sqrt(np.sum((pixel_colors.astype(float) - 
                                        np.array(bg_color).astype(float))**2, axis=1))
        
        # Normalize distances
        if np.max(color_distances) > 0:
            normalized_distances = color_distances / np.max(color_distances) * 255
        else:
            normalized_distances = color_distances
        
        # Create color-based mask with foreground bias
        threshold = self.tolerance * (1 - self.foreground_bias)
        alpha_flat = np.where(normalized_distances < threshold, 0, 255).astype(np.uint8)
        alpha = alpha_flat.reshape((h, w))
        
        # Get edge mask
        edges = self.detect_pixel_edges(image)
        
        # Create confidence-weighted combined mask
        # Edges have priority, then color difference
        final_mask = np.maximum(alpha, edges)
        
        # Post-processing to clean up the mask
        if self.edge_refinement:
            # Remove isolated pixels
            kernel = np.ones((3, 3), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Fill small holes
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Create RGBA image
        rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = final_mask
        
        return rgba

    def optimize_pixel_scaling(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Advanced scaling for pixel art that preserves pixel sharpness and handles
        both upscaling and downscaling scenarios optimally.
        
        Args:
            image: PIL Image
            target_size: Desired output size
            
        Returns:
            Scaled PIL Image
        """
        # Calculate optimal intermediate size
        current_w, current_h = image.size
        target_w, target_h = target_size
        
        # Different strategies for upscaling vs downscaling
        if target_w >= current_w and target_h >= current_h:
            # Upscaling - use pixel perfect scaling first
            scale_factor = min(target_w // current_w, target_h // current_h)
            if scale_factor > 1:
                # First do integer scaling
                intermediate_size = (
                    current_w * scale_factor,
                    current_h * scale_factor
                )
                image = image.resize(intermediate_size, Image.NEAREST)
                
            # Final resize to exact target
            if image.size != target_size:
                image = image.resize(target_size, Image.NEAREST)
        else:
            # Downscaling - use multiple steps
            # Step 1: If big difference, use a halfway point with BICUBIC
            if current_w / target_w > 2 or current_h / target_h > 2:
                intermediate_size = (
                    current_w // 2,
                    current_h // 2
                )
                image = image.resize(intermediate_size, Image.BICUBIC)
                return self.optimize_pixel_scaling(image, target_size)  # Recursively continue
            
            # Final resize with appropriate algorithm for small downscaling
            image = image.resize(target_size, Image.BICUBIC)
            
            # Restore pixel art look
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)  # Slightly sharpen
        
        return image

    def process_image(self, 
                     input_path: str, 
                     output_path: str, 
                     target_size: Tuple[int, int],
                     preview_mode: bool = False) -> Optional[Image.Image]:
        """
        Process a single image with enhanced features.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            target_size: Desired output size
            preview_mode: If True, return preview image without saving
            
        Returns:
            PIL Image if preview_mode is True, otherwise None
        """
        try:
            if not preview_mode:
                logger.info(f"Processing {input_path}")
            
            # Load image
            image_bgr = cv2.imread(input_path)
            if image_bgr is None:
                raise ValueError(f"Could not load image: {input_path}")
                
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Auto-adjust parameters based on image content
            self._auto_adjust_parameters(image_rgb)
            
            # Remove background
            if not preview_mode:
                logger.info("Removing background...")
            image_rgba = self.remove_background_advanced(image_rgb)
            
            # Convert to PIL Image
            image_pil = Image.fromarray(image_rgba)
            
            # Optimize scaling
            if not preview_mode:
                logger.info("Optimizing pixel scaling...")
            
            if target_size != (image_rgb.shape[1], image_rgb.shape[0]):
                resized_image = self.optimize_pixel_scaling(image_pil, target_size)
            else:
                resized_image = image_pil
            
            if preview_mode:
                return resized_image
            
            # Save with maximum quality
            logger.info(f"Saving to {output_path}")
            resized_image.save(output_path, 'PNG', optimize=True)
            
            logger.info("Processing complete")
            return None
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {str(e)}")
            raise

    def _auto_adjust_parameters(self, image: np.ndarray) -> None:
        """
        Automatically adjust processing parameters based on image content.
        
        Args:
            image: Input image as numpy array
        """
        # Detect if the image is likely pixel art
        is_pixel_art = self._detect_pixel_art(image)
        
        if is_pixel_art:
            # For pixel art, increase edge sensitivity and reduce tolerance
            self.edge_sensitivity = min(0.9, self.edge_sensitivity * 1.2)
            self.tolerance = max(20, self.tolerance * 0.8)
        else:
            # For photos, reduce edge sensitivity and increase tolerance
            self.edge_sensitivity = max(0.7, self.edge_sensitivity * 0.9)
            self.tolerance = min(40, self.tolerance * 1.2)
        
        # Analyze image complexity
        complexity = self._calculate_image_complexity(image)
        
        # Adjust parameters based on complexity
        if complexity > 0.7:  # Complex image
            self.foreground_bias = min(0.8, self.foreground_bias * 1.1)
            self.color_clusters = min(12, self.color_clusters + 2)
        elif complexity < 0.3:  # Simple image
            self.foreground_bias = max(0.6, self.foreground_bias * 0.9)
            self.color_clusters = max(6, self.color_clusters - 2)
            
    def _detect_pixel_art(self, image: np.ndarray) -> bool:
        """
        Detect if an image is likely pixel art based on color count and edge patterns.
        
        Args:
            image: Input image
            
        Returns:
            True if likely pixel art, False otherwise
        """
        # Resize to standard size for analysis
        h, w = image.shape[:2]
        analysis_size = (min(w, 256), min(h, 256))
        img_small = cv2.resize(image, analysis_size)
        
        # Count unique colors (pixel art usually has limited palette)
        unique_colors = np.unique(img_small.reshape(-1, 3), axis=0)
        color_count = len(unique_colors)
        
        # Check for regular grid patterns (common in pixel art)
        gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Look for horizontal and vertical lines
        horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=5, maxLineGap=3)
        vertical_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=5, maxLineGap=3)
        
        has_grid = False
        if horizontal_lines is not None and vertical_lines is not None:
            # Check if lines form a grid pattern
            h_angles = []
            for line in horizontal_lines:
                for x1, y1, x2, y2 in line:
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    h_angles.append(angle)
            
            v_angles = []
            for line in vertical_lines:
                for x1, y1, x2, y2 in line:
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    v_angles.append(angle)
            
            # If many lines close to 0 or 90 degrees, likely grid
            h_count = sum(1 for a in h_angles if a < 10 or a > 170)
            v_count = sum(1 for a in v_angles if a > 80 and a < 100)
            
            has_grid = (h_count > 5 and v_count > 5)
        
        # Pixel art typically has limited colors and grid-like structure
        return color_count < 256 or has_grid
    
    def _calculate_image_complexity(self, image: np.ndarray) -> float:
        """
        Calculate a complexity score for the image.
        
        Args:
            image: Input image
            
        Returns:
            Complexity score (0-1)
        """
        # Convert to grayscale for edge analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Measure edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        # Measure color variation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        color_std = np.std(hsv[:,:,0]) / 180.0  # Normalize by hue range
        
        # Combine measures (weighted average)
        complexity = 0.7 * edge_density + 0.3 * color_std
        
        return min(1.0, complexity)

    def create_preview(self, input_path: str, target_size: Tuple[int, int]) -> Image.Image:
        """
        Create a quick preview of processing result.
        
        Args:
            input_path: Path to input image
            target_size: Desired output size
            
        Returns:
            PIL Image with preview
        """
        return self.process_image(input_path, "", target_size, preview_mode=True)


def process_folder(input_folder: str, 
                  output_folder: str, 
                  target_size: Tuple[int, int],
                  tolerance: int = 30,
                  edge_sensitivity: float = 0.8,
                  foreground_bias: float = 0.7,
                  edge_refinement: bool = True,
                  dither_handling: bool = True,
                  num_workers: int = 4) -> None:
    """
    Process all images in a folder with enhanced features and multi-threading.
    
    Args:
        input_folder: Path to input folder
        output_folder: Path to output folder
        target_size: Desired output size
        tolerance: Color similarity threshold
        edge_sensitivity: Edge detection sensitivity
        foreground_bias: Bias towards foreground preservation
        edge_refinement: Whether to apply edge refinement
        dither_handling: Whether to handle dithered patterns
        num_workers: Number of worker threads
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logger.info(f"Created output directory: {output_folder}")
    
    processor = EnhancedPixelArtProcessor(
        tolerance=tolerance,
        edge_sensitivity=edge_sensitivity,
        foreground_bias=foreground_bias,
        edge_refinement=edge_refinement,
        dither_handling=dither_handling
    )
    
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(input_folder, pattern)))
    
    if not image_files:
        logger.warning(f"No images found in {input_folder}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Define worker function for thread pool
    def process_single_image(file_path):
        try:
            filename = os.path.basename(file_path)
            output_path = os.path.join(
                output_folder, 
                f'{os.path.splitext(filename)[0]}_processed.png'
            )
            
            processor.process_image(file_path, output_path, target_size)
            return True
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            return False
    
    # Process images using thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_image, image_files),
            total=len(image_files),
            desc="Processing images"
        ))
    
    success_count = sum(results)
    logger.info(f"Processed {success_count} of {len(image_files)} images successfully")


def main():
    """
    Main entry point of the application.
    """
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
    parser.add_argument('--foreground_bias', type=float, default=0.7,
                      help='Bias towards foreground preservation (0-1)')
    parser.add_argument('--edge_refinement', action='store_true', default=True,
                      help='Apply edge refinement')
    parser.add_argument('--dither_handling', action='store_true', default=True,
                      help='Apply dithering pattern detection')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of worker threads')
    parser.add_argument('--preview', type=str, default=None,
                      help='Create preview for single image instead of processing folder')

    args = parser.parse_args()

    try:
        if args.preview:
            # Preview mode for a single image
            processor = EnhancedPixelArtProcessor(
                tolerance=args.tolerance,
                edge_sensitivity=args.edge_sensitivity,
                foreground_bias=args.foreground_bias,
                edge_refinement=args.edge_refinement,
                dither_handling=args.dither_handling
            )
            
            preview = processor.create_preview(args.preview, tuple(args.target_size))
            preview_path = os.path.join(
                args.output_folder, 
                os.path.basename(args.preview).replace('.', '_preview.')
            )
            preview.save(preview_path)
            logger.info(f"Preview saved to {preview_path}")
        else:
            # Process entire folder
            process_folder(
                args.input_folder,
                args.output_folder,
                tuple(args.target_size),
                args.tolerance,
                args.edge_sensitivity,
                args.foreground_bias,
                args.edge_refinement,
                args.dither_handling,
                args.num_workers
            )
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise


if __name__ == '__main__':
    # Add additional imports for extended functionality
    try:
        from PIL import ImageEnhance
        from tqdm import tqdm
    except ImportError:
        logger.error("Additional dependencies required. Install with:")
        logger.error("pip install Pillow tqdm")
        exit(1)
        
    main()