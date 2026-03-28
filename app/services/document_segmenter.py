import cv2
import numpy as np
from typing import Optional, List, Tuple
import onnxruntime as ort
import os
import logging
import urllib.request

logger = logging.getLogger(__name__)


class DocumentSegmenter:
    """
    Document segmentation using U²-Net ONNX model.
    Returns precise quadrilateral corners for perspective transform.
    
    Model source: https://github.com/xuebinqin/U-2-Net
    Pre-converted ONNX: https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
    """
    
    MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
    INPUT_SIZE = 320  # Fallback size when model input shape is dynamic/unknown
    MIN_MODEL_SIZE_BYTES = 10 * 1024 * 1024
    
    def __init__(self, model_path: Optional[str] = None):
        
        self.model_path = model_path or self._get_default_model_path()
        self.session: Optional[ort.InferenceSession] = None
        self.input_size: Tuple[int, int] = (self.INPUT_SIZE, self.INPUT_SIZE)
        self._load_model()
    
    def _get_default_model_path(self) -> str:
        """Get default model path, download if missing"""
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "u2net.onnx")

        if not os.path.exists(path):
            logger.info(f"Downloading U²-Net model to {path}")
            self._download_model(path)

        return path

    def _download_model(self, target_path: str) -> None:
        """Download model atomically to avoid partially-written files."""
        tmp_path = f"{target_path}.tmp"
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            urllib.request.urlretrieve(self.MODEL_URL, tmp_path)

            if not os.path.exists(tmp_path):
                raise RuntimeError("model download failed: temporary file missing")

            file_size = os.path.getsize(tmp_path)
            if file_size < self.MIN_MODEL_SIZE_BYTES:
                raise RuntimeError(f"model download appears incomplete (size={file_size} bytes)")

            os.replace(tmp_path, target_path)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def _load_model(self) -> None:
        """Load ONNX model with CPU provider (add CUDA for GPU)"""
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        use_cuda = os.getenv("ONNXRUNTIME_USE_CUDA", "0").strip().lower() in {"1", "true", "yes"}
        if use_cuda and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif use_cuda and "CUDAExecutionProvider" not in available:
            logger.warning("ONNXRUNTIME_USE_CUDA is enabled but CUDAExecutionProvider is unavailable; using CPU")

        for attempt in range(2):
            try:
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                model_input = self.session.get_inputs()[0]
                self.input_name = model_input.name
                self.input_size = self._resolve_input_size(model_input.shape)
                active_providers = self.session.get_providers()
                logger.info(
                    f"U²-Net model loaded: {self.model_path}; "
                    f"input_size={self.input_size}; active_providers={active_providers}"
                )
                return
            except Exception as e:
                message = str(e)
                is_corrupt = "INVALID_PROTOBUF" in message.upper() or "PROTOBUF" in message.upper()

                if attempt == 0 and is_corrupt:
                    logger.warning("U²-Net model file appears corrupted, re-downloading model")
                    try:
                        self._download_model(self.model_path)
                        continue
                    except Exception as download_error:
                        logger.error(f"Failed to re-download U²-Net model: {download_error}")

                logger.error(f"Failed to load U²-Net model: {e}")
                self.session = None
                return

    def _resolve_input_size(self, input_shape: list) -> Tuple[int, int]:
        """
        Resolve model input (width, height) from ONNX input shape.
        Falls back to INPUT_SIZE when shape is dynamic or unavailable.
        """
        try:
            # Typical NCHW: [batch, channels, height, width]
            if len(input_shape) >= 4:
                h = input_shape[2]
                w = input_shape[3]

                if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                    return (w, h)
        except Exception:
            pass

        return (self.INPUT_SIZE, self.INPUT_SIZE)
    
    def get_document_mask(
        self,
        image: np.ndarray,
        adaptive_threshold: bool = True,
        refine_edges: bool = True
    ) -> Optional[np.ndarray]:
        """
        Generate binary mask of document region.
        
        Args:
            image: Input RGB/BGR image
            adaptive_threshold: Use adaptive thresholding for better edge detection
            refine_edges: Apply GrabCut for precise boundary refinement
        
        Returns: uint8 mask (0=background, 255=document) or None if failed
        """
        if self.session is None:
            return None

        try:
            h, w = image.shape[:2]

            # Ensure RGB format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # Check if BGR (OpenCV default) by comparing blue/red channels
                if image[0, 0, 0] > image[0, 0, 2]:  # B > R suggests BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Preprocess: resize + normalize + transpose for ONNX
            input_img = cv2.resize(image, self.input_size)
            input_img = input_img.astype(np.float32) / 255.0
            input_img = np.transpose(input_img, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)

            # Run inference
            outputs = self.session.run(None, {self.input_name: input_img})
            mask = outputs[0][0][0]  # (1, 1, H, W) -> (H, W)

            # Postprocess: resize back + threshold
            mask = cv2.resize(mask, (w, h))

            # Adaptive thresholding for better boundary detection
            if adaptive_threshold:
                # Use Otsu's thresholding on the probability mask
                mask_norm = (mask * 255).astype(np.uint8)
                _, mask = cv2.threshold(
                    mask_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            else:
                mask = (mask > 0.5).astype(np.uint8) * 255

            # Morphological cleanup (remove small noise)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Edge refinement using GrabCut for precise boundaries
            if refine_edges:
                mask = self._refine_mask_edges(image, mask)

            return mask

        except Exception as e:
            logger.error(f"Segmentation inference failed: {e}")
            return None

    def _refine_mask_edges(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        iterations: int = 3
    ) -> np.ndarray:
        """
        Refine mask edges using GrabCut algorithm for precise document boundaries.
        """
        if mask.sum() == 0:
            return mask

        try:
            # Create grabCut mask (GD_PR/FILL/PR_BGD)
            gc_mask = np.zeros(mask.shape[:2], np.uint8)
            gc_mask[mask > 0] = cv2.GC_PR_FGD
            gc_mask[mask == 0] = cv2.GC_PR_BGD

            # Estimate bounding rect for initialization
            coords = cv2.findNonZero(mask)
            if coords is None:
                return mask

            x, y, w, h = cv2.boundingRect(coords)

            # Add small margin
            margin = 5
            x, y = max(0, x - margin), max(0, y - margin)
            w, h = min(image.shape[1] - x, w + 2 * margin), min(image.shape[0] - y, h + 2 * margin)

            bgd_model = np.zeros((1, 65), np.float32)
            fgd_model = np.zeros((1, 65), np.float32)

            cv2.grabCut(
                image, gc_mask, (x, y, w, h),
                bgd_model, fgd_model,
                iterCount=iterations,
                mode=cv2.GC_INIT_WITH_MASK
            )

            # Convert grabCut result to binary mask
            refined_mask = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)

            # Final cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)

            return refined_mask

        except Exception as e:
            logger.debug(f"GrabCut refinement failed: {e}, using original mask")
            return mask
    
    def extract_corners_from_mask(
        self,
        mask: np.ndarray,
        min_area_ratio: float = 0.05,
        use_multiscale: bool = True
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Extract ordered quadrilateral corners from binary mask.
        
        Args:
            mask: Binary document mask
            min_area_ratio: Minimum contour area ratio to consider
            use_multiscale: Try multiple epsilon values for better quad detection
        
        Returns: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] in TL, TR, BR, BL order
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Get largest contour (assume it's the document)
        doc_contour = max(contours, key=cv2.contourArea)

        # Filter: ignore tiny contours (noise)
        image_area = mask.shape[0] * mask.shape[1]
        if cv2.contourArea(doc_contour) < image_area * min_area_ratio:
            logger.debug(f"Document contour too small: {cv2.contourArea(doc_contour)}/{image_area}")
            return None

        # Try multiple epsilon values for better quadrilateral approximation
        if use_multiscale:
            epsilon_ratios = [0.01, 0.015, 0.02, 0.025, 0.03]
            for epsilon_ratio in epsilon_ratios:
                epsilon = epsilon_ratio * cv2.arcLength(doc_contour, True)
                approx = cv2.approxPolyDP(doc_contour, epsilon, True)

                if len(approx) == 4:
                    points = approx.reshape(4, 2).astype(np.float32)
                    ordered = self._order_points(points)
                    # Validate: check aspect ratio is reasonable (not too extreme)
                    if self._validate_quadrilateral(ordered, mask.shape):
                        return ordered

        # Fallback: get minimum area rectangle → 4 corners
        rect = cv2.minAreaRect(doc_contour)
        box = cv2.boxPoints(rect)
        ordered = self._order_points(box.astype(np.float32))

        if self._validate_quadrilateral(ordered, mask.shape):
            return ordered

        logger.debug("Failed to extract valid quadrilateral from mask")
        return None

    def _validate_quadrilateral(
        self,
        points: List[Tuple[float, float]],
        image_shape: Tuple[int, int]
    ) -> bool:
        """
        Validate quadrilateral is reasonable (not degenerate or too skewed).
        """
        h, w = image_shape

        # Check all points are within image bounds (with small tolerance)
        margin = 10
        for pt in points:
            if not (-margin <= pt[0] <= w + margin and -margin <= pt[1] <= h + margin):
                return False

        # Check area is not too small
        points_array = np.array(points, dtype=np.float32).reshape(4, 1, 2)
        area = cv2.contourArea(points_array)
        if area < image_shape[0] * image_shape[1] * 0.01:
            return False

        # Check aspect ratio is reasonable (document-like)
        ordered = self._order_points(np.array(points, dtype=np.float32))
        width_top = np.sqrt((ordered[1][0] - ordered[0][0]) ** 2 + (ordered[1][1] - ordered[0][1]) ** 2)
        width_bottom = np.sqrt((ordered[2][0] - ordered[3][0]) ** 2 + (ordered[2][1] - ordered[3][1]) ** 2)
        height_left = np.sqrt((ordered[3][0] - ordered[0][0]) ** 2 + (ordered[3][1] - ordered[0][1]) ** 2)
        height_right = np.sqrt((ordered[2][0] - ordered[1][0]) ** 2 + (ordered[2][1] - ordered[1][1]) ** 2)

        avg_width = (width_top + width_bottom) / 2
        avg_height = (height_left + height_right) / 2

        if avg_height == 0 or avg_width == 0:
            return False

        aspect_ratio = max(avg_width, avg_height) / min(avg_width, avg_height)

        # Documents typically have aspect ratio between 0.3 and 3.0
        if aspect_ratio > 3.0:
            return False

        return True
    
    def _order_points(self, pts: np.ndarray) -> List[Tuple[float, float]]:
        """Order 4 points: TL, TR, BR, BL"""
        # Sort by x, then split left/right
        x_sorted = pts[np.argsort(pts[:, 0]), :]
        left = x_sorted[:2, :]
        right = x_sorted[2:, :]
        
        # Sort left by y (top first)
        left_sorted = left[np.argsort(left[:, 1]), :]
        tl, bl = left_sorted[0], left_sorted[1]
        
        # Sort right by y (top first)
        right_sorted = right[np.argsort(right[:, 1]), :]
        tr, br = right_sorted[0], right_sorted[1]
        
        return [
            (float(tl[0]), float(tl[1])),
            (float(tr[0]), float(tr[1])),
            (float(br[0]), float(br[1])),
            (float(bl[0]), float(bl[1])),
        ]
    
    def detect(
        self,
        image: np.ndarray,
        return_mask: bool = False,
        adaptive_threshold: bool = True,
        refine_edges: bool = True
    ) -> Tuple[Optional[List[Tuple[float, float]]], str, Optional[dict]]:
        """
        Main detection method.
        
        Args:
            image: Input image
            return_mask: Also return the segmentation mask
            adaptive_threshold: Use adaptive thresholding
            refine_edges: Apply GrabCut edge refinement
        
        Returns:
            (corners, status, metadata) where:
            - corners: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] or None
            - status: "unet", "unet_unavailable", "unet_failed", "unet_no_corners"
            - metadata: dict with mask, confidence, etc. or None
        """
        metadata = {}

        if self.session is None:
            return None, "unet_unavailable", None

        mask = self.get_document_mask(
            image,
            adaptive_threshold=adaptive_threshold,
            refine_edges=refine_edges
        )

        if mask is None:
            return None, "unet_failed", None

        metadata["mask"] = mask if return_mask else None
        metadata["mask_coverage"] = float(np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1]))

        corners = self.extract_corners_from_mask(mask)
        if corners is None:
            return None, "unet_no_corners", metadata

        # Compute confidence based on mask coverage and corner validity
        metadata["confidence"] = self._compute_confidence(corners, mask, image.shape)

        return corners, "unet", metadata

    def _compute_confidence(
        self,
        corners: List[Tuple[float, float]],
        mask: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> float:
        """
        Compute detection confidence score (0.0 - 1.0).
        """
        confidence = 1.0

        # Factor 1: Mask coverage (ideal: 20-80% of image)
        coverage = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
        if coverage < 0.1:
            confidence -= 0.3
        elif coverage > 0.9:
            confidence -= 0.2
        elif 0.2 <= coverage <= 0.7:
            confidence += 0.1

        # Factor 2: Corner validity (already validated, but check geometry)
        h, w = image_shape[:2]
        points = np.array(corners, dtype=np.float32)

        # Check if corners are within image bounds
        for pt in corners:
            if not (0 <= pt[0] <= w and 0 <= pt[1] <= h):
                confidence -= 0.2
                break

        # Factor 3: Aspect ratio preference (documents are usually rectangular)
        width_top = np.sqrt((corners[1][0] - corners[0][0]) ** 2 + (corners[1][1] - corners[0][1]) ** 2)
        width_bottom = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
        height_left = np.sqrt((corners[3][0] - corners[0][0]) ** 2 + (corners[3][1] - corners[0][1]) ** 2)
        height_right = np.sqrt((corners[2][0] - corners[1][0]) ** 2 + (corners[2][1] - corners[1][1]) ** 2)

        # Check trapezoid distortion (perspective effect)
        width_diff = abs(width_top - width_bottom) / max(width_top, width_bottom, 1)
        height_diff = abs(height_left - height_right) / max(height_left, height_right, 1)

        if width_diff > 0.3 or height_diff > 0.3:
            confidence -= 0.15  # Significant perspective distortion

        return max(0.0, min(1.0, confidence))

    def crop_document(
        self,
        image: np.ndarray,
        corners: Optional[List[Tuple[float, float]]] = None,
        return_metadata: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Detect document and crop to perspective-corrected rectangle.
        
        Args:
            image: Input image
            corners: Pre-detected corners (if None, will detect automatically)
            return_metadata: Also return detection metadata
        
        Returns:
            (cropped_image, metadata) or (None, None) if detection failed
        """
        metadata = {}

        if corners is None:
            corners, status, detect_meta = self.detect(image, return_mask=True)
            if corners is None:
                logger.debug(f"Document detection failed: {status}")
                return None, None
            metadata.update(detect_meta or {})

        # Compute output dimensions
        width_top = np.sqrt((corners[1][0] - corners[0][0]) ** 2 + (corners[1][1] - corners[0][1]) ** 2)
        width_bottom = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
        height_top = np.sqrt((corners[1][0] - corners[2][0]) ** 2 + (corners[1][1] - corners[2][1]) ** 2)
        height_bottom = np.sqrt((corners[0][0] - corners[3][0]) ** 2 + (corners[0][1] - corners[3][1]) ** 2)

        max_width = int(max(width_top, width_bottom))
        max_height = int(max(height_top, height_bottom))

        # Ensure minimum dimensions
        max_width = max(max_width, 100)
        max_height = max(max_height, 100)

        # Create destination points for perspective transform
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        src = np.array(corners, dtype=np.float32)

        # Apply perspective transform
        M = cv2.getPerspectiveTransform(src, dst)
        cropped = cv2.warpPerspective(
            image, M, (max_width, max_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        metadata["crop_dimensions"] = (max_width, max_height)

        if return_metadata:
            return cropped, metadata
        return cropped, None