import cv2
import numpy as np
from typing import Tuple, List, Optional
import os
import logging
from app.services.document_segmenter import DocumentSegmenter

logger = logging.getLogger(__name__)


class DocumentDetector:
    """
    Hybrid document corner detection (CamScanner-style):
    1. YOLOv10 (fast, good for clean docs)
    2. U²-Net segmentation (precise, handles complex backgrounds)
    3. Enhanced OpenCV edge detection + contour finding (CamScanner-style)
    """

    def __init__(self, yolo_model_path: Optional[str] = None, unet_model_path: Optional[str] = None):
        self.yolo_model = None
        self.segmenter = DocumentSegmenter(model_path=unet_model_path)
        self._load_yolo_model(yolo_model_path)

    def _load_yolo_model(self, model_path: Optional[str]) -> None:
        """Load YOLO model if available"""
        try:
            from doclayout_yolo import YOLOv10
            path = model_path or "juliozhao/DocLayout-YOLO-DocStructBench"
            if model_path and os.path.exists(model_path):
                self.yolo_model = YOLOv10(model_path)
            else:
                self.yolo_model = YOLOv10.from_pretrained(path)
            logger.info("YOLOv10 model loaded")
        except Exception as e:
            logger.warning(f"YOLOv10 not available: {e}. Install missing deps such as 'huggingface_hub'.")
            self.yolo_model = None

    def detect(self, image):
        # 1. YOLO FIRST (fast)
        if self.yolo_model:
            try:
                corners = self._detect_with_yolo(image)
                if corners:
                    return corners, "yolo"
            except Exception:
                pass

        # 2. U-Net ONLY if needed (returns 3 values: corners, status, metadata)
        result = self.segmenter.detect(image)
        corners = result[0]
        status = result[1]
        
        if corners:
            return corners, "unet"

        # 3. Enhanced OpenCV (CamScanner-style)
        corners = self._detect_with_opencv(image)
        if corners:
            return corners, "opencv"

        # 4. Last fallback: image bounds
        h, w = image.shape[:2]
        logger.warning("No document detected; returning image bounds")
        return [(0, 0), (w, 0), (w, h), (0, h)], "fallback"

    def _detect_with_yolo(self, image: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """Detect corners using YOLO bounding box → approximate quad"""
        if self.yolo_model is None:
            return None

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        results = self.yolo_model(image, verbose=False)
        if not results or len(results) == 0:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        # Get largest box by area
        all_boxes = boxes.xyxy.cpu().numpy()
        areas = (all_boxes[:, 2] - all_boxes[:, 0]) * (all_boxes[:, 3] - all_boxes[:, 1])
        max_idx = int(np.argmax(areas))
        xyxy = all_boxes[max_idx]
        x1, y1, x2, y2 = xyxy

        return [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2)
        ]

    def _detect_with_opencv(self, image: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """
        CamScanner-style document detection using edge detection + contour finding.
        This method detects the actual document boundaries from a photographed image.
        """
        # Ensure proper format
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = np.ascontiguousarray(image)

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Step 1: Denoise the image
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Step 2: Adaptive thresholding (handles varying lighting conditions)
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Step 3: Canny edge detection
        edges = cv2.Canny(denoised, 75, 200)

        # Step 4: Morphological operations to close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges_dilated = cv2.dilate(edges_closed, kernel, iterations=2)

        # Step 5: Combine adaptive threshold and edges for better contour detection
        combined = cv2.bitwise_or(adaptive_thresh, edges_dilated)

        # Step 6: Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Step 7: Find the largest quadrilateral contour (the document)
        doc_contour = None
        min_area = gray.size * 0.05  # Document should be at least 5% of image

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Check if it has 4 points (quadrilateral)
            if len(approx) == 4:
                doc_contour = approx
                break

            # If not exactly 4 points, check if it's close to a rectangle
            if len(approx) >= 4:
                # Get minimum bounding rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box.astype(np.int32))

                # Check if contour fills most of its bounding box (rectangular shape)
                if area / box_area > 0.8:
                    doc_contour = box.astype(np.int32).reshape(-1, 1, 2)
                    break

        if doc_contour is None:
            # Fallback: try to find any large rectangular shape
            for contour in contours[:5]:  # Check top 5 contours
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box.astype(np.int32))

                if area / box_area > 0.7:
                    doc_contour = box.astype(np.int32).reshape(-1, 1, 2)
                    break

        if doc_contour is None:
            logger.warning("No document contour found")
            return None

        # Step 8: Order the points (TL, TR, BR, BL)
        points = doc_contour.reshape(-1, 2).astype(np.float32)
        return self._order_points(points)

    def _order_points(self, pts: np.ndarray) -> List[Tuple[float, float]]:
        """Order points: TL, TR, BR, BL"""
        x_sorted = pts[np.argsort(pts[:, 0]), :]
        left = x_sorted[:2, :]
        right = x_sorted[2:, :]
        left_sorted = left[np.argsort(left[:, 1]), :]
        right_sorted = right[np.argsort(right[:, 1]), :]
        return [
            (float(left_sorted[0][0]), float(left_sorted[0][1])),  # TL
            (float(right_sorted[0][0]), float(right_sorted[0][1])), # TR
            (float(right_sorted[1][0]), float(right_sorted[1][1])), # BR
            (float(left_sorted[1][0]), float(left_sorted[1][1])),   # BL
        ]

    def crop_and_transform(self, image: np.ndarray, corners: List[Tuple[float, float]]) -> np.ndarray:
        """Apply perspective transform with aspect-ratio preservation"""
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        pts = np.array(corners, dtype=np.float32)

        # Compute output dimensions (preserve aspect ratio)
        width_a = np.linalg.norm(pts[0] - pts[1])
        width_b = np.linalg.norm(pts[2] - pts[3])
        height_a = np.linalg.norm(pts[0] - pts[3])
        height_b = np.linalg.norm(pts[1] - pts[2])

        max_width = max(int(width_a), int(width_b))
        max_height = max(int(height_a), int(height_b))

        # Enforce reasonable output size
        max_width = min(max_width, 2500)
        max_height = min(max_height, 3500)

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(
            image, matrix, (max_width, max_height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
            borderValue=(255, 255, 255)
        )

        return warped
