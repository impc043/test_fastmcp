"""
Robust Signature Extraction Module
Extracts signatures from various document formats (images, PDFs, TIFs)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SignatureRegion:
    """Data class to store signature information"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    image: np.ndarray
    confidence: float
    page_number: int = 0


class SignatureExtractor:
    """
    Robust signature extraction from documents using OpenCV
    """
    
    def __init__(
        self,
        min_area: int = 1000,
        max_area: int = 100000,
        aspect_ratio_range: Tuple[float, float] = (0.5, 10.0),
        morph_kernel_size: Tuple[int, int] = (3, 3),
        contour_approx_method: int = cv2.CHAIN_APPROX_SIMPLE,
        padding: int = 10
    ):
        """
        Initialize signature extractor with configurable parameters
        
        Args:
            min_area: Minimum contour area to consider
            max_area: Maximum contour area to consider
            aspect_ratio_range: (min, max) aspect ratio for signature bounding boxes
            morph_kernel_size: Kernel size for morphological operations
            contour_approx_method: OpenCV contour approximation method
            padding: Padding around extracted signature
        """
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_range = aspect_ratio_range
        self.morph_kernel_size = morph_kernel_size
        self.contour_approx_method = contour_approx_method
        self.padding = padding
        
    def load_document(self, file_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Load document from various formats (image, PDF, TIFF)
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of images (pages) as numpy arrays
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        # Handle multi-page TIFF
        if suffix in ['.tif', '.tiff']:
            return self._load_multipage_tiff(str(file_path))
        
        # Handle PDF (requires pdf2image)
        elif suffix == '.pdf':
            return self._load_pdf(str(file_path))
        
        # Handle standard image formats
        elif suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
            img = cv2.imread(str(file_path))
            if img is None:
                raise ValueError(f"Failed to load image: {file_path}")
            return [img]
        
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_multipage_tiff(self, file_path: str) -> List[np.ndarray]:
        """Load all pages from a multi-page TIFF"""
        images = []
        ret, img = True, None
        
        # Use OpenCV's multi-image support
        imobj = cv2.imreadmulti(file_path)
        if imobj[0]:
            images = imobj[1]
        else:
            # Fallback method
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                images = [img]
        
        logger.info(f"Loaded {len(images)} pages from TIFF")
        return images
    
    def _load_pdf(self, file_path: str) -> List[np.ndarray]:
        """Load PDF pages as images (requires pdf2image library)"""
        try:
            from pdf2image import convert_from_path
            
            pil_images = convert_from_path(file_path, dpi=300)
            images = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 
                     for img in pil_images]
            
            logger.info(f"Loaded {len(images)} pages from PDF")
            return images
            
        except ImportError:
            raise ImportError(
                "pdf2image library required for PDF support. "
                "Install: pip install pdf2image"
            )
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for signature detection
        
        Args:
            image: Input image
            
        Returns:
            Binary image optimized for signature detection
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Adaptive thresholding for varying lighting conditions
        binary = cv2.adaptiveThreshold(
            filtered,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            self.morph_kernel_size
        )
        
        # Close small gaps in strokes
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove small noise
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return morph
    
    def detect_signatures(
        self, 
        image: np.ndarray,
        page_number: int = 0
    ) -> List[SignatureRegion]:
        """
        Detect signature regions in preprocessed image
        
        Args:
            image: Input image (BGR or grayscale)
            page_number: Page number for multi-page documents
            
        Returns:
            List of SignatureRegion objects
        """
        original = image.copy()
        preprocessed = self.preprocess_image(image)
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            preprocessed,
            cv2.RETR_EXTERNAL,
            self.contour_approx_method
        )
        
        signatures = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter by aspect ratio
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue
            
            # Calculate confidence based on contour properties
            confidence = self._calculate_confidence(contour, preprocessed)
            
            # Add padding
            x_pad = max(0, x - self.padding)
            y_pad = max(0, y - self.padding)
            x_end = min(original.shape[1], x + w + self.padding)
            y_end = min(original.shape[0], y + h + self.padding)
            
            # Extract signature region from original image
            sig_img = original[y_pad:y_end, x_pad:x_end]
            
            signatures.append(SignatureRegion(
                bbox=(x_pad, y_pad, x_end - x_pad, y_end - y_pad),
                image=sig_img,
                confidence=confidence,
                page_number=page_number
            ))
        
        # Sort by confidence
        signatures.sort(key=lambda s: s.confidence, reverse=True)
        
        # Remove overlapping detections (Non-Maximum Suppression)
        signatures = self._apply_nms(signatures, iou_threshold=0.3)
        
        logger.info(f"Detected {len(signatures)} signatures on page {page_number}")
        return signatures
    
    def _calculate_confidence(
        self, 
        contour: np.ndarray, 
        binary_image: np.ndarray
    ) -> float:
        """
        Calculate confidence score for signature detection
        
        Args:
            contour: Detected contour
            binary_image: Binary image
            
        Returns:
            Confidence score (0-1)
        """
        # Get contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Compactness (signatures are typically moderately compact)
        compactness = 4 * np.pi * area / (perimeter ** 2)
        
        # Extent (ratio of contour area to bounding box area)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        # Solidity (ratio of contour area to convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Weighted confidence score
        confidence = (
            0.3 * min(compactness, 1.0) +
            0.3 * extent +
            0.4 * solidity
        )
        
        return confidence
    
    def _apply_nms(
        self, 
        signatures: List[SignatureRegion], 
        iou_threshold: float = 0.3
    ) -> List[SignatureRegion]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            signatures: List of signature regions
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of signatures
        """
        if len(signatures) == 0:
            return []
        
        # Extract bounding boxes
        boxes = np.array([s.bbox for s in signatures])
        scores = np.array([s.confidence for s in signatures])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        areas = boxes[:, 2] * boxes[:, 3]
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep only non-overlapping boxes
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [signatures[i] for i in keep]
    
    def extract_signatures(
        self, 
        file_path: Union[str, Path],
        confidence_threshold: float = 0.3
    ) -> List[SignatureRegion]:
        """
        Main method to extract all signatures from a document
        
        Args:
            file_path: Path to document file
            confidence_threshold: Minimum confidence to include signature
            
        Returns:
            List of all detected signatures across all pages
        """
        images = self.load_document(file_path)
        all_signatures = []
        
        for page_num, image in enumerate(images):
            signatures = self.detect_signatures(image, page_number=page_num)
            
            # Filter by confidence
            filtered = [s for s in signatures if s.confidence >= confidence_threshold]
            all_signatures.extend(filtered)
        
        logger.info(f"Total signatures extracted: {len(all_signatures)}")
        return all_signatures
    
    def save_signatures(
        self, 
        signatures: List[SignatureRegion],
        output_dir: Union[str, Path],
        prefix: str = "signature"
    ):
        """
        Save extracted signatures to files
        
        Args:
            signatures: List of signature regions
            output_dir: Directory to save signatures
            prefix: Filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, sig in enumerate(signatures):
            filename = f"{prefix}_page{sig.page_number}_{i+1}_conf{sig.confidence:.2f}.png"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), sig.image)
            logger.info(f"Saved: {filepath}")
    
    def visualize_detections(
        self, 
        image: np.ndarray,
        signatures: List[SignatureRegion]
    ) -> np.ndarray:
        """
        Draw bounding boxes around detected signatures
        
        Args:
            image: Original image
            signatures: List of detected signatures
            
        Returns:
            Image with drawn bounding boxes
        """
        vis_image = image.copy()
        
        for sig in signatures:
            x, y, w, h = sig.bbox
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add confidence label
            label = f"{sig.confidence:.2f}"
            cv2.putText(
                vis_image, 
                label, 
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        return vis_image


# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = SignatureExtractor(
        min_area=1000,
        max_area=100000,
        aspect_ratio_range=(0.5, 10.0),
        padding=10
    )
    
    # Example: Extract signatures from a document
    # signatures = extractor.extract_signatures("document.pdf")
    # extractor.save_signatures(signatures, "output_signatures")
    
    print("\n=== Example 1: Basic Extraction ===")
    
    extractor = SignatureExtractor( min_area=800,  max_area=150000, )
    # Extract signatures from an image file
    signatures = extractor.extract_signatures(r"C:\Users\patil\Downloads\in3.jpg")
    
    print(f"Found {len(signatures)} signatures")
    
    # Save extracted signatures
    extractor.save_signatures(signatures, "output/basic")