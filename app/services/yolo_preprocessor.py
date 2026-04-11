import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import os
import urllib.request
import logging

logger = logging.getLogger(__name__)

from .logo_detector import LogoDetector


class YOLOInvoicePreprocessor:
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

    def __init__(self):
        self._superres = None
        self._superres_loaded = False
        self._logo_detector = LogoDetector()

    def preprocess_invoice(
        self,
        image: np.ndarray,
        target_width: int = 1600,
        enhance_quality: bool = True,
        lang: str = "en"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Ensure RGB
        image = self._ensure_rgb(image)

        metadata: Dict[str, Any] = {
            "original_shape": image.shape,
            "steps_applied": []
        }

        try:
            # Quality assessment
            metrics = self._assess_quality(image)
            metadata["initial_metrics"] = {k: float(v) for k, v in metrics.items()}
            metadata["steps_applied"].append("quality_assessment")

            # Resize to optimal dimensions for OCR
            image = self._resize_for_ocr(image, target_width)
            metadata["resized_shape"] = image.shape
            metadata["steps_applied"].append("resize_for_ocr")

            # Super-resolution (if image is still low-res)
            h, w = image.shape[:2]
            if enhance_quality and w < 1000:
                logger.info("Applying super-resolution to low-res invoice")
                image = self._apply_superres(image)
                metadata["steps_applied"].append("super_resolution")
                metadata["superres_applied"] = True
            else:
                metadata["superres_applied"] = False

            # Detect and mask logos (prevents OCR from scanning logos)
            image, logo_metadata = self._logo_detector.detect_and_mask_logos(image)
            metadata["logo_detection"] = logo_metadata
            if logo_metadata.get("logos_detected", 0) > 0:
                metadata["steps_applied"].append("logo_masking")
            else:
                metadata["steps_applied"].append("logo_masking_skipped")

            # quality after logo masking
            metrics = self._assess_quality(image)

            # Denoise (only if quality is poor, match original behavior)
            if metrics["contrast"] < self.QUALITY_THRESHOLDS["contrast_min"]:
                logger.info("Applying denoise due to low contrast")
                image = self._denoise(image, lang)
                metadata["steps_applied"].append("denoise")
            else:
                metadata["steps_applied"].append("denoise_skipped")

            # Contrast enhancement (ALWAYS apply, match original behavior)
            logger.info(f"Applying contrast enhancement (current contrast: {metrics['contrast']:.1f})")
            image = self._enhance_contrast(image, lang)
            metadata["steps_applied"].append("contrast_enhancement")

            post_contrast_1 = float(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).std())
            metadata["contrast_after_pass1"] = post_contrast_1
            metrics["skew_angle"] = 0.0
            metadata["deskew_applied"] = False
            metadata["steps_applied"].append("deskew_skipped")

            # Final Assessment and confidence scoring
            final_metrics = self._assess_quality(image)
            metadata["final_metrics"] = {k: float(v) for k, v in final_metrics.items()}
            confidence = self._compute_confidence(final_metrics)
            metadata["confidence"] = float(confidence)

            logger.info(
                f"Invoice preprocessing complete: "
                f"blur={final_metrics['blur_score']:.1f}, "
                f"contrast={final_metrics['contrast']:.1f}, "
                f"skew={final_metrics['skew_angle']:.2f}, "
                f"confidence={confidence:.2f}"
            )

            return image, metadata

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", exc_info=True)
            return image, {"error": str(e), "steps_applied": ["failed"]}

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is in RGB format"""
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Assume BGR (OpenCV default)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _assess_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess image quality metrics"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        brightness = float(gray.mean())

        return {
            "blur_score": blur_score,
            "contrast": contrast,
            "brightness": brightness,
            "skew_angle": 0.0,
        }

    def _compute_confidence(self, metrics: Dict[str, float]) -> float:
        """Compute preprocessing confidence score (0.0 - 1.0)"""
        score = 1.0

        if metrics["blur_score"] < self.QUALITY_THRESHOLDS["blur_min"]:
            score -= 0.2
        if metrics["contrast"] < self.QUALITY_THRESHOLDS["contrast_min"]:
            score -= 0.2
        if metrics["brightness"] < self.QUALITY_THRESHOLDS["brightness_min"]:
            score -= 0.1
        if abs(metrics["skew_angle"]) > self.QUALITY_THRESHOLDS["skew_max"]:
            score -= 0.15

        return max(score, 0.0)

    def _resize_for_ocr(self, image: np.ndarray, target_width: int) -> np.ndarray:
        """Resize to optimal dimensions for OCR"""
        h, w = image.shape[:2]

        # Use target_width but respect bounds
        if w < self.MIN_WIDTH:
            scale = self.MIN_WIDTH / w
            new_w = self.MIN_WIDTH
            new_h = int(h * scale)
        elif w > self.MAX_WIDTH:
            scale = self.MAX_WIDTH / w
            new_w = self.MAX_WIDTH
            new_h = int(h * scale)
        else:
            # Within bounds, use target_width if specified
            if target_width and target_width != w:
                scale = target_width / w
                new_w = target_width
                new_h = int(h * scale)
            else:
                return image

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    def _load_superres(self):
        """Load super-resolution model"""
        if self._superres_loaded:
            return self._superres

        self._superres_loaded = True

        try:
            if not hasattr(cv2, "dnn_superres"):
                logger.warning("OpenCV dnn_superres not available")
                return None

            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            os.makedirs(model_dir, exist_ok=True)

            path = os.path.join(model_dir, self.SR_MODEL_NAME)

            if not os.path.exists(path):
                logger.info(f"Downloading super-resolution model to {path}")
                urllib.request.urlretrieve(self.SR_MODEL_URL, path)

            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(path)
            sr.setModel("edsr", 2)

            self._superres = sr
            logger.info("Super-resolution model loaded")
            return sr

        except Exception as e:
            logger.warning(f"Super-resolution failed: {e}")
            return None

    def _apply_superres(self, image: np.ndarray) -> np.ndarray:
        """Apply super-resolution to increase image quality"""
        sr = self._load_superres()
        if sr is None:
            return image

        try:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out = sr.upsample(bgr)
            return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Super-resolution upscaling failed: {e}")
            return image

    def _denoise(self, image: np.ndarray, lang: str = "en") -> np.ndarray:
        """Denoise image - language-aware"""
        if lang == "km":
            return cv2.medianBlur(image, 3)

        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    def _enhance_contrast(self, image: np.ndarray, lang: str = "en") -> np.ndarray:
        """Apply CLAHE contrast enhancement — match original pipeline exactly"""
        # Match original ImagePreprocessor: clipLimit=3.0 for English
        clip = 2.0 if lang == "km" else 3.0

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        l = clahe.apply(l)

        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    def _deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct text skew using projection profile analysis.
        Only rotates if skew angle is SIGNIFICANT (>1.0 degree) to avoid
        unnecessary contrast degradation from interpolation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Invert: text should be white on black for analysis
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        text_pixels = cv2.countNonZero(binary)
        if text_pixels < 100:
            return image, 0.0

        best_angle = 0.0
        best_score = -1.0

        # Scan angles from -5 to +5 degrees
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
            score = float(np.var(projection))

            if score > best_score:
                best_score = score
                best_angle = angle
        if abs(best_angle) <= 1.0:
            logger.debug(f"Skew angle {best_angle:.1f}° too small, skipping deskew")
            return image, 0.0

        logger.info(f"Deskewing by {best_angle:.1f} degrees")
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), best_angle, 1)

        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated, best_angle

    def generate_multi_strategy(
        self,
        image: np.ndarray,
        lang: str = "en"
    ) -> List[Tuple[np.ndarray, str, Dict[str, Any]]]:
        """
        Generate multiple preprocessing strategies for best OCR results.
        Returns list of (image, strategy_name, metadata).
        """
        strategies = []

        # Light enhancement (English documents)
        img1, meta1 = self.preprocess_invoice(image, lang=lang, enhance_quality=False)
        strategies.append((img1, "light_en", meta1))

        # Khmer-enhanced (if Khmer text expected)
        img2, meta2 = self.preprocess_invoice(image, lang="km", enhance_quality=True)
        strategies.append((img2, "khmer_enhanced", meta2))

        # Heavy enhancement (poor quality images)
        img3, meta3 = self.preprocess_invoice(image, lang=lang, enhance_quality=True)
        strategies.append((img3, "heavy_enhanced", meta3))

        # Original
        img4 = self._resize_for_ocr(self._ensure_rgb(image), 1600)
        strategies.append((img4, "original", {"steps_applied": ["resize_only"]}))

        return strategies
