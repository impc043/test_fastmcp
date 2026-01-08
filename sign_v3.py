import cv2
import numpy as np
from typing import Tuple, Optional


class SignatureEnhancer:
    """
    Robust signature enhancement pipeline that handles various ink colors
    and suppresses text noise while preserving signature quality.
    """
    
    @staticmethod
    def enhance_signature(image: np.ndarray, 
                         method: str = 'multi_channel',
                         morphology_strength: int = 2,
                         threshold_sensitivity: float = 1.2) -> np.ndarray:
        """
        Main enhancement pipeline with multiple strategies.
        
        Args:
            image: Input cropped image containing signature
            method: 'multi_channel', 'adaptive', or 'combined' (best)
            morphology_strength: 1-5, controls noise removal (2 is balanced)
            threshold_sensitivity: 0.8-1.5, lower = more aggressive enhancement
            
        Returns:
            Enhanced binary image (white signature on black background)
        """
        if method == 'multi_channel':
            return SignatureEnhancer._multi_channel_approach(
                image, morphology_strength, threshold_sensitivity)
        elif method == 'adaptive':
            return SignatureEnhancer._adaptive_approach(
                image, morphology_strength)
        else:  # combined (recommended)
            return SignatureEnhancer._combined_approach(
                image, morphology_strength, threshold_sensitivity)
    
    @staticmethod
    def _preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocessing: denoise and prepare image."""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Denoise while preserving edges
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        
        # Create grayscale version
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
        return denoised, gray
    
    @staticmethod
    def _multi_channel_approach(image: np.ndarray, 
                                morph_strength: int,
                                thresh_sensitivity: float) -> np.ndarray:
        """
        Multi-channel analysis to detect signatures in any color.
        Works by analyzing each color channel separately.
        """
        denoised, gray = SignatureEnhancer._preprocess_image(image)
        
        # Split into color channels
        b, g, r = cv2.split(denoised)
        
        # Process each channel to find ink strokes
        channels_processed = []
        for channel in [b, g, r, gray]:
            # Apply CLAHE for local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(channel)
            
            # Otsu's thresholding to find dark regions (ink)
            _, thresh = cv2.threshold(enhanced, 0, 255, 
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            channels_processed.append(thresh)
        
        # Combine all channels - signature appears in at least one channel
        combined = np.zeros_like(channels_processed[0])
        for ch in channels_processed:
            combined = cv2.bitwise_or(combined, ch)
        
        # Morphological operations to clean up
        result = SignatureEnhancer._apply_morphology(combined, morph_strength)
        
        return result
    
    @staticmethod
    def _adaptive_approach(image: np.ndarray, 
                          morph_strength: int) -> np.ndarray:
        """
        Adaptive thresholding approach - good for uneven lighting.
        """
        denoised, gray = SignatureEnhancer._preprocess_image(image)
        
        # Apply bilateral filter to smooth while keeping edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(bilateral, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 10)
        
        # Morphological operations
        result = SignatureEnhancer._apply_morphology(adaptive, morph_strength)
        
        return result
    
    @staticmethod
    def _combined_approach(image: np.ndarray,
                          morph_strength: int,
                          thresh_sensitivity: float) -> np.ndarray:
        """
        Combined approach using both methods for robust detection.
        This is the recommended method for most cases.
        """
        # Get results from both methods
        multi_channel = SignatureEnhancer._multi_channel_approach(
            image, morph_strength, thresh_sensitivity)
        adaptive = SignatureEnhancer._adaptive_approach(
            image, morph_strength)
        
        # Combine using intersection (keeps only strong candidates)
        combined = cv2.bitwise_and(multi_channel, adaptive)
        
        # If too aggressive, use union instead
        if np.sum(combined) < np.sum(multi_channel) * 0.3:
            combined = cv2.bitwise_or(multi_channel, adaptive)
        
        # Final cleanup
        result = SignatureEnhancer._apply_morphology(combined, morph_strength)
        
        # Remove very small noise (likely text fragments)
        result = SignatureEnhancer._remove_small_components(result)
        
        return result
    
    @staticmethod
    def _apply_morphology(image: np.ndarray, strength: int) -> np.ndarray:
        """
        Apply morphological operations to clean noise.
        
        Args:
            image: Binary image
            strength: 1-5, controls aggressiveness
        """
        # Adjust kernel size based on strength
        kernel_size = strength + 1
        
        # Remove small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (kernel_size, kernel_size))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)
        
        # Close small gaps in signature strokes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                (kernel_size + 1, kernel_size + 1))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
        
        return closed
    
    @staticmethod
    def _remove_small_components(image: np.ndarray, 
                                 min_area_ratio: float = 0.01) -> np.ndarray:
        """
        Remove small connected components (likely text noise).
        Keeps components larger than min_area_ratio of total image area.
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            image, connectivity=8)
        
        # Calculate minimum area threshold
        total_area = image.shape[0] * image.shape[1]
        min_area = total_area * min_area_ratio
        
        # Create output image
        output = np.zeros_like(image)
        
        # Keep components larger than threshold (skip label 0 = background)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                output[labels == i] = 255
        
        return output
    
    @staticmethod
    def visualize_pipeline(image: np.ndarray) -> np.ndarray:
        """
        Show comparison of different methods side by side.
        Useful for debugging and parameter tuning.
        """
        # Get results from different methods
        multi_channel = SignatureEnhancer.enhance_signature(
            image, method='multi_channel')
        adaptive = SignatureEnhancer.enhance_signature(
            image, method='adaptive')
        combined = SignatureEnhancer.enhance_signature(
            image, method='combined')
        
        # Resize for display
        h, w = image.shape[:2]
        display_h = 300
        display_w = int(w * display_h / h)
        
        original_resized = cv2.resize(image, (display_w, display_h))
        multi_resized = cv2.resize(multi_channel, (display_w, display_h))
        adaptive_resized = cv2.resize(adaptive, (display_w, display_h))
        combined_resized = cv2.resize(combined, (display_w, display_h))
        
        # Convert grayscale to BGR for stacking
        if len(original_resized.shape) == 2:
            original_resized = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)
        multi_resized = cv2.cvtColor(multi_resized, cv2.COLOR_GRAY2BGR)
        adaptive_resized = cv2.cvtColor(adaptive_resized, cv2.COLOR_GRAY2BGR)
        combined_resized = cv2.cvtColor(combined_resized, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original_resized, 'Original', (10, 30), 
                   font, 1, (0, 255, 0), 2)
        cv2.putText(multi_resized, 'Multi-Channel', (10, 30), 
                   font, 1, (0, 255, 0), 2)
        cv2.putText(adaptive_resized, 'Adaptive', (10, 30), 
                   font, 1, (0, 255, 0), 2)
        cv2.putText(combined_resized, 'Combined (Best)', (10, 30), 
                   font, 1, (0, 255, 0), 2)
        
        # Stack vertically
        result = np.vstack([original_resized, multi_resized, 
                           adaptive_resized, combined_resized])
        
        return result


# Example usage
if __name__ == "__main__":
    # Load your cropped signature image
    image = cv2.imread('signature_crop.jpg')
    
    # Method 1: Quick enhancement (recommended)
    enhanced = SignatureEnhancer.enhance_signature(image, method='combined')
    cv2.imwrite('signature_enhanced.jpg', enhanced)
    
    # Method 2: With custom parameters
    enhanced_custom = SignatureEnhancer.enhance_signature(
        image, 
        method='combined',
        morphology_strength=3,  # More aggressive noise removal
        threshold_sensitivity=1.1  # Slightly more sensitive
    )
    cv2.imwrite('signature_enhanced_custom.jpg', enhanced_custom)
    
    # Method 3: Visualize all methods for comparison
    comparison = SignatureEnhancer.visualize_pipeline(image)
    cv2.imwrite('signature_comparison.jpg', comparison)
    
    print("Enhancement complete!")
    print(f"Output shape: {enhanced.shape}")
    print(f"Signature pixels: {np.sum(enhanced == 255)}")