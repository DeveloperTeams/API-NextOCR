import cv2
import numpy as np
import os
import urllib.request
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

from .document_segmenter import DocumentSegmenter


class ImagePreprocessor:
    """Adaptive + language-aware preprocessing for OCR"""

    MAX_WIDTH = 2000
    MIN_WIDTH = 1200

    SR_MODEL_NAME = "EDSR_x2.pb"
    SR_MODEL_URL = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb"

    QUALITY_THRESHOLDS = {
        "blur_min": 80,
        "contrast_min": 40,
        "brightness_min": 80,
        "skew_max": 2.0,
    }

    def __init__(
        self,
        use_segmenter: bool = True,
        unet_model_path: Optional[str] = None,
    ):
        self._superres = None
        self._superres_loaded = False
        self._segmenter: Optional[DocumentSegmenter] = None

        if use_segmenter:
            try:
                self._segmenter = DocumentSegmenter(model_path=unet_model_path)
            except Exception as e:
                logger.warning(f"Failed to initialize DocumentSegmenter: {e}")

    def assess_quality(self, image: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = float(gray.std())
        brightness = float(gray.mean())

        return {
            "blur_score": blur,
            "contrast": contrast,
            "brightness": brightness,
            "skew_angle": 0.0,  # computed later
        }

    def compute_confidence(self, metrics: Dict[str, float]) -> float:
        score = 1.0

        if metrics["blur_score"] < 80:
            score -= 0.2
        if metrics["contrast"] < 40:
            score -= 0.2
        if metrics["brightness"] < 70:
            score -= 0.1

        return max(score, 0.0)

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def resize_for_ocr(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]

        if w < self.MIN_WIDTH:
            scale = self.MIN_WIDTH / w
            return cv2.resize(image, (self.MIN_WIDTH, int(h * scale)))

        if w > self.MAX_WIDTH:
            scale = self.MAX_WIDTH / w
            return cv2.resize(image, (self.MAX_WIDTH, int(h * scale)))

        return image

    def _load_superres(self):
        if self._superres_loaded:
            return self._superres

        self._superres_loaded = True

        try:
            if not hasattr(cv2, "dnn_superres"):
                return None

            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            os.makedirs(model_dir, exist_ok=True)

            path = os.path.join(model_dir, self.SR_MODEL_NAME)

            if not os.path.exists(path):
                urllib.request.urlretrieve(self.SR_MODEL_URL, path)

            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(path)
            sr.setModel("edsr", 2)

            self._superres = sr
            return sr

        except Exception as e:
            logger.warning(f"Superres failed: {e}")
            return None

    def apply_superres(self, image):
        sr = self._load_superres()
        if sr is None:
            return image

        try:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out = sr.upsample(bgr)
            return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        except:
            return image

    def enhance_contrast(self, image, lang="en"):
        clip = 2.0 if lang == "km" else 3.0

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l = clahe.apply(l)

        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    def denoise(self, image, lang="en"):
        if lang == "km":
            return cv2.medianBlur(image, 3)

        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    def detect_and_crop_document(
        self,
        image: np.ndarray,
        use_segmenter: bool = True
    ) -> Tuple[np.ndarray, bool]:
        """
        Detect document boundaries and crop to the document only.
        Tries U²-Net segmentation first (if available), falls back to contour detection.

        Returns: (cropped_image, success_flag)
        """
        # Try U²-Net segmentation first (more accurate)
        if use_segmenter and self._segmenter is not None:
            try:
                cropped, metadata = self._segmenter.crop_document(image, return_metadata=True)
                if cropped is not None:
                    logger.info(f"Document cropped via U²-Net: {metadata.get('crop_dimensions', 'unknown')}")
                    return cropped, True
            except Exception as e:
                logger.debug(f"U²-Net segmentation failed, falling back to contour method: {e}")

        # Fallback: contour-based detection (from image_preprocessor)
        img_copy = image.copy()
        gray = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)

        # Step 1: Denoise and enhance edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        # Step 2: Find contours
        contours, _ = cv2.findContours(
            edged.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image, False

        # Step 3: Find the largest contour by area
        doc_contour = max(contours, key=cv2.contourArea)

        # Filter: must be large enough (>50% of image area)
        image_area = image.shape[0] * image.shape[1]
        if cv2.contourArea(doc_contour) < image_area * 0.5:
            return image, False

        # Step 4: Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(doc_contour, True)
        approx = cv2.approxPolyDP(doc_contour, epsilon, True)

        # Step 5: If we found 4 points, we have a rectangle
        if len(approx) == 4:
            # Sort points: top-left, top-right, bottom-right, bottom-left
            pts = self._order_points(approx.reshape(4, 2))

            # Apply perspective warp
            warped = self._four_point_transform(image, pts)
            return warped, True

        # Fallback: use bounding rect
        x, y, w, h = cv2.boundingRect(doc_contour)
        cropped = image[y:y+h, x:x+w]

        # Only return if crop is significantly smaller than original
        if cropped.shape[0] * cropped.shape[1] < image_area * 0.95:
            return cropped, True

        return image, False

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply perspective transform to get a top-down view"""
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute width of new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = int(max(widthA, widthB))

        # Compute height of new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def deskew(self, image):
        """
        Detect and correct text skew using projection profile analysis.
        More reliable than minAreaRect for documents with white backgrounds.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Invert: text should be white on black for analysis
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        text_pixels = cv2.countNonZero(binary)
        if text_pixels < 100:
            return image, 0.0

        best_angle = 0.0
        best_score = -1

        for angle in np.arange(-5.0, 5.5, 0.5):
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            rotated = cv2.warpAffine(
                binary, M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            projection = np.sum(rotated, axis=1)
            score = np.var(projection)

            if score > best_score:
                best_score = score
                best_angle = angle

        if abs(best_angle) < 0.5:
            return image, 0.0

        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), best_angle, 1)

        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated, best_angle

    def preprocess_adaptive(
        self,
        image: np.ndarray,
        for_nextocr: bool = True,
        lang: str = "en",
        auto_crop: bool = True,
        use_segmenter: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Full adaptive preprocessing pipeline for OCR.

        Args:
            image: Input image (RGB/BGR)
            for_nextocr: Apply super-resolution for low-res images
            lang: Language code ("en", "km", etc.) - affects enhancement
            auto_crop: Auto-detect and crop document boundaries
            use_segmenter: Use U²-Net segmentation (more accurate, slower)

        Returns:
            (processed_image, metadata) with metrics and confidence
        """
        image = self._ensure_rgb(image)

        # STEP 0: Auto-crop document (detect & crop invoice)
        crop_metadata = {}
        if auto_crop:
            image, crop_success = self.detect_and_crop_document(image, use_segmenter=use_segmenter)
            crop_metadata["crop_success"] = crop_success
            logger.info(f"Document detection: {'success' if crop_success else 'skipped/failed'}")

        metrics = self.assess_quality(image)

        # STEP 1 resize
        image = self.resize_for_ocr(image)

        h, w = image.shape[:2]

        # STEP 2 super-res (resolution-based)
        if for_nextocr and w < 1000:
            logger.info("Applying super-resolution")
            image = self.apply_superres(image)

        # STEP 3 denoise
        if metrics["contrast"] < 40:
            image = self.denoise(image, lang)

        # STEP 4 contrast
        if metrics["contrast"] < 50:
            image = self.enhance_contrast(image, lang)

        # STEP 5 deskew
        image, angle = self.deskew(image)
        metrics["skew_angle"] = angle

        confidence = self.compute_confidence(metrics)

        return image, {
            "metrics": metrics,
            "confidence": confidence,
            **crop_metadata
        }
