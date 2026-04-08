import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)


class YOLOInvoiceDetector:
    """
    Enhanced YOLO-based invoice detection focused on detecting entire invoice documents.
    This service ensures complete invoice cropping without cutting off parts.
    """

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize YOLO invoice detector.
        
        Args:
            model_path: Path to YOLO model (uses DocLayout-YOLO by default)
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.yolo_model = None
        
        # Use provided path or default to local model
        if model_path is None:
            default_path = r"d:/Project/backend/api/app/models/models--juliozhao--DocLayout-YOLO-DocStructBench/snapshots/8c3299a30b8ff29a1503c4431b035b93220f7b11/doclayout_yolo_docstructbench_imgsz1024.pt"
            if os.path.exists(default_path):
                model_path = default_path
        
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]) -> None:
        """Load YOLO model for invoice detection"""
        try:
            from doclayout_yolo import YOLOv10
            
            if model_path and os.path.exists(model_path):
                self.yolo_model = YOLOv10(model_path)
                logger.info(f"YOLO model loaded from local path: {model_path}")
                return

            path = "juliozhao/DocLayout-YOLO-DocStructBench"
            self.yolo_model = YOLOv10.from_pretrained(path)
            logger.info(f"YOLO model loaded from HuggingFace: {path}")
                
        except ImportError as e:
            logger.error(f"doclayout_yolo package not installed: {e}")
            self.yolo_model = None
        except Exception as e:
            logger.warning(f"Failed to load YOLO model from HuggingFace: {e}. Will use fallback detection.")
            self.yolo_model = None

    def detect_invoice(
        self, 
        image: np.ndarray,
        ensure_full_invoice: bool = True
    ) -> Tuple[Optional[List[Tuple[float, float]]], str, Dict[str, Any]]:
        
        if self.yolo_model is None:
            logger.warning("YOLO model not available")
            return self._fallback_to_full_image(image)

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        try:
            results = self.yolo_model(image, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)
            
            if not results or len(results) == 0:
                logger.warning("YOLO returned no results")
                return self._fallback_to_full_image(image)

            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                logger.warning("YOLO detected no boxes")
                return self._fallback_to_full_image(image)

            all_boxes = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
  
            valid_mask = confidences >= self.conf_threshold
            if not np.any(valid_mask):
                logger.warning("No boxes meet confidence threshold")
                return self._fallback_to_full_image(image)
            
            valid_boxes = all_boxes[valid_mask]
            
            # Get the largest invoice box by area
            areas = (valid_boxes[:, 2] - valid_boxes[:, 0]) * (valid_boxes[:, 3] - valid_boxes[:, 1])
            max_idx = int(np.argmax(areas))
            best_box = valid_boxes[max_idx]
            
            # If ensure_fullInvoice is True, try to merge overlapping boxes
            # to capture the entire invoice (header + body + footer)
            if ensure_full_invoice and len(valid_boxes) > 1:
                merged_box = self._merge_invoice_boxes(valid_boxes, areas)
                if merged_box is not None:
                    best_box = merged_box
                    status = "yolo_expanded"
                else:
                    status = "yolo"
            else:
                status = "yolo"
            
            # Convert box to corners (TL, TR, BR, BL)
            x1, y1, x2, y2 = best_box
            
            # Add margin to ensure we don't cut off edges
            if ensure_full_invoice:
                margin = self._calculate_margin(x1, y1, x2, y2, image.shape)
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(image.shape[1], x2 + margin)
                y2 = min(image.shape[0], y2 + margin)
            
            corners = [
                (float(x1), float(y1)),  # TL
                (float(x2), float(y1)),  # TR
                (float(x2), float(y2)),  # BR
                (float(x1), float(y2)),  # BL
            ]
            
            metadata = {
                "confidence": float(confidences[max_idx]),
                "box_area": float(areas[max_idx]),
                "detection_class": "invoice",
                "num_boxes_detected": int(len(valid_boxes)),
                "merged_boxes": bool(status == "yolo_expanded"),
            }
            
            logger.info(f"YOLO invoice detection successful: {status}, confidence={metadata['confidence']:.2f}")
            return corners, status, metadata
            
        except Exception as e:
            logger.error(f"YOLO invoice detection failed: {e}", exc_info=True)
            return self._fallback_to_full_image(image)

    def _merge_invoice_boxes(
        self, 
        boxes: np.ndarray, 
        areas: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Merge overlapping invoice boxes to capture entire document.
        Only merges boxes that are truly part of the same document (high overlap).
        """
        if len(boxes) == 0:
            return None
        
        # Sort boxes by area (largest first)
        sorted_indices = np.argsort(-areas)
        
        # Start with the largest box
        merged_box = boxes[sorted_indices[0]].copy()
        for idx in sorted_indices[1:]:
            box = boxes[idx]
            
            # Calculate IoU with current merged box
            iou = self._calculate_iou(merged_box, box)
            
            # Only merge if boxes are clearly part of the same document
            # Higher threshold = more selective merging
            if iou > 0.3:
                merged_box[0] = min(merged_box[0], box[0])  # x1
                merged_box[1] = min(merged_box[1], box[1])  # y1
                merged_box[2] = max(merged_box[2], box[2])  # x2
                merged_box[3] = max(merged_box[3], box[3])  # y2
        
        # Validate merged box is reasonable size
        merged_area = (merged_box[2] - merged_box[0]) * (merged_box[3] - merged_box[1])
        largest_area = areas[sorted_indices[0]]
        
        # Only use merged box if it's not dramatically larger
        if merged_area < largest_area * 2.5:
            return merged_box
        
        # If merged box is too large, return the largest single box instead
        return None

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        return intersection / union

    def _calculate_margin(
        self, 
        x1: float, 
        y1: float, 
        x2: float, 
        y2: float, 
        image_shape: Tuple[int, ...]
    ) -> float:
        """Calculate appropriate margin to add around detection"""
        width = x2 - x1
        height = y2 - y1
        
        # Add 2-5% margin based on invoice size
        margin_percent = 0.03
        margin = int(max(width, height) * margin_percent)
        
        # Cap margin at reasonable value
        margin = min(margin, 50)
        
        return margin

    def _fallback_to_full_image(
        self, 
        image: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], str, Dict[str, Any]]:
        """Fallback to full image when detection fails"""
        h, w = image.shape[:2]
        corners = [(0, 0), (float(w), 0), (float(w), float(h)), (0, float(h))]
        
        metadata = {
            "confidence": 0.0,
            "box_area": int(w * h),
            "detection_class": "fallback",
            "num_boxes_detected": 0,
            "merged_boxes": False,
        }
        
        logger.warning("Using full image as fallback for invoice detection")
        return corners, "fallback", metadata

    def crop_and_transform(
        self, 
        image: np.ndarray, 
        corners: List[Tuple[float, float]],
        preserve_aspect_ratio: bool = True
    ) -> np.ndarray:
        """
        Apply perspective transform to crop invoice with high quality.
        Uses similar approach to U-Net preprocessing.
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        pts = np.array(corners, dtype=np.float32)

        # Compute output dimensions
        width_a = np.linalg.norm(pts[0] - pts[1])
        width_b = np.linalg.norm(pts[2] - pts[3])
        height_a = np.linalg.norm(pts[0] - pts[3])
        height_b = np.linalg.norm(pts[1] - pts[2])

        max_width = max(int(width_a), int(width_b))
        max_height = max(int(height_a), int(height_b))

        # Enforce reasonable output size (cap at 3000x4000 for high quality)
        max_width = min(max_width, 3000)
        max_height = min(max_height, 4000)

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

    def visualize_detection(
        self, 
        image: np.ndarray, 
        corners: Optional[List[Tuple[float, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        vis_image = image.copy()
        
        if corners and len(corners) == 4:
            # Draw detection box
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(vis_image, [pts], True, (0, 255, 0), 3)
            
            # Draw corner points
            for i, (x, y) in enumerate(corners):
                cv2.circle(vis_image, (int(x), int(y)), 8, (0, 0, 255), -1)
                labels = ['TL', 'TR', 'BR', 'BL']
                cv2.putText(
                    vis_image, labels[i], (int(x) + 10, int(y) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
            
            # Add metadata text
            if metadata:
                y_pos = 30
                for key, value in metadata.items():
                    if isinstance(value, float):
                        text = f"{key}: {value:.2f}"
                    else:
                        text = f"{key}: {value}"
                    cv2.putText(
                        vis_image, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                    )
                    y_pos += 25
        
        return vis_image
