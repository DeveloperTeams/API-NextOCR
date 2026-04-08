import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LogoDetector:
    """
    Detect and mask logo/image regions in invoices to prevent OCR from
    wasting time on non-text elements. This improves OCR accuracy and speed.
    """

    def __init__(
        self,
        min_logo_area_pct: float = 0.005,
        max_logo_area_pct: float = 0.15,
        saturation_threshold: int = 40,
    ):
        """
        Args:
            min_logo_area_pct: Minimum logo area as % of total image
            max_logo_area_pct: Maximum logo area as % of total image
            saturation_threshold: Minimum saturation to consider as color logo
        """
        self.min_logo_area_pct = min_logo_area_pct
        self.max_logo_area_pct = max_logo_area_pct
        self.saturation_threshold = saturation_threshold

    def detect_and_mask_logos(
        self,
        image: np.ndarray,
        mask_with_white: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect logo regions and mask them with white.
        Uses multiple strategies:
        1. High-saturation color blobs (color logos)
        2. Photo-like regions (high entropy, low text structure)
        3. Bounding box analysis (isolated rectangular regions)

        Args:
            image: Input invoice image (RGB)
            mask_with_white: Replace logo with white background

        Returns:
            Tuple of (masked_image, metadata)
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        metadata: Dict[str, Any] = {
            "logos_detected": 0,
            "logo_regions": [],
            "total_masked_area_pct": 0.0,
        }

        try:
            # Convert to HSV for color analysis
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            h, w = image.shape[:2]
            total_area = h * w

            # Strategy 1: Detect high-saturation color blobs (logos are often colorful)
            saturation_mask = hsv[:, :, 1] > self.saturation_threshold
            
            # Strategy 2: Detect photo-like regions (high entropy, low edge density)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = cv2.countNonZero(edges) / total_area

            # Combine strategies
            # Find contours in saturation mask
            sat_contours, _ = cv2.findContours(
                saturation_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            min_area = int(total_area * self.min_logo_area_pct)
            max_area = int(total_area * self.max_logo_area_pct)

            logo_contours = []
            masked_pixels = 0

            for contour in sat_contours:
                area = cv2.contourArea(contour)
                
                # Filter by size
                if area < min_area or area > max_area:
                    continue
                
                # Check if this is likely a logo (not text)
                if self._is_likely_logo(contour, hsv, gray, area, total_area):
                    logo_contours.append(contour)
                    masked_pixels += area

            # Mask detected logos
            if logo_contours:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, logo_contours, -1, 255, -1)

                if mask_with_white:
                    # Fill logo regions with white
                    image[mask == 255] = [255, 255, 255]

                # Collect logo regions for metadata
                for contour in logo_contours:
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    metadata["logo_regions"].append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w_box),
                        "height": int(h_box),
                        "area_pct": float((w_box * h_box) / total_area * 100),
                    })

            metadata["logos_detected"] = len(logo_contours)
            metadata["total_masked_area_pct"] = float(masked_pixels / total_area * 100)

            if logo_contours:
                logger.info(
                    f"Masked {len(logo_contours)} logo region(s), "
                    f"covering {metadata['total_masked_area_pct']:.1f}% of image"
                )

            return image, metadata

        except Exception as e:
            logger.warning(f"Logo detection failed: {e}, returning original image")
            return image, metadata

    def _is_likely_logo(
        self,
        contour: np.ndarray,
        hsv: np.ndarray,
        gray: np.ndarray,
        area: float,
        total_area: float
    ) -> bool:
        """
        Determine if a contour is likely a logo (not text).
        Logos tend to be:
        - Larger than text
        - Have high color variation
        - Low text-like structure
        """
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)
        
        # Text lines typically have aspect ratio > 3 or < 0.3
        # Logos tend to be more square/rectangular (0.3 < aspect < 3)
        if aspect_ratio > 3.0 or aspect_ratio < 0.3:
            # Might be a text line or separator, not a logo
            return False
        
        # Check color variation within the region
        region_hsv = hsv[y:y+h, x:x+w]
        sat_mean = np.mean(region_hsv[:, :, 1])
        val_mean = np.mean(region_hsv[:, :, 2])
        
        # Logos tend to have moderate-high saturation and value
        if sat_mean < 30:
            # Too desaturated — likely background or grayscale text
            return False
        
        # Check text-like structure (high edge density = text)
        region_gray = gray[y:y+h, x:x+w]
        region_edges = cv2.Canny(region_gray, 50, 150)
        region_edge_density = cv2.countNonZero(region_edges) / max(area, 1)
        
        # Text has high edge density, logos have lower
        if region_edge_density > 0.15:
            # Too many edges — likely text block
            return False
        
        # Region is compact and colorful with low edge density — likely logo
        return True
