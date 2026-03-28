import os
import io
import cv2
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import List

from app.config import Config
from app.models.schemas import (
    CornerDetectionResponse, CornerPoint, DetectionMethod,
    ApplyCropResponse, OCRResponse, HealthResponse,
    UnifiedOCRRequest, UnifiedOCRResponse, OCRAttempt,
    DetectionStage, OCRStage, ExtractionStage, PipelineMetadata
)
from app.services.document_detector import DocumentDetector
from app.services.image_preprocessor import ImagePreprocessor
from app.services.ocr_client import OCRClient
from app.services.data_extractor import DataExtractor
from app.services.ocr_service import UnifiedOCRService
from app.utils.helpers import generate_unique_filename, ensure_dir_exists

import logging

# -------------------------------
# Initialize logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _corners_to_bbox(corners: List[tuple]) -> dict:
    xs = [float(c[0]) for c in corners]
    ys = [float(c[1]) for c in corners]
    return {
        "x": min(xs),
        "y": min(ys),
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
    }


def _to_detection_method(method: str) -> DetectionMethod:
    if method == "manual":
        return DetectionMethod.FALLBACK
    if method == "yolo":
        return DetectionMethod.YOLO
    if method == "unet":
        return DetectionMethod.UNET
    if method == "opencv":
        return DetectionMethod.OPENCV
    return DetectionMethod.FALLBACK

# -------------------------------
# Initialize FastAPI app
# -------------------------------
app = FastAPI(title="Invoice OCR Extractor", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Upload directory
# -------------------------------
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), Config.UPLOAD_FOLDER)
ensure_dir_exists(UPLOAD_DIR)

# -------------------------------
# Initialize services
# -------------------------------
detector = DocumentDetector(yolo_model_path=Config.YOLO_MODEL)
preprocessor = ImagePreprocessor()
ocr_client = OCRClient(
    ocr_space_key=Config.OCR_SPACE_API_KEY,
    nextocr_endpoint=Config.NEXTOCR_ENDPOINT,
    nextocr_username=Config.NEXTOCR_USERNAME,
    nextocr_secret_key=Config.NEXTOCR_SECRET_KEY
)
extractor = DataExtractor()

# Unified OCR Service (new enhanced pipeline)
unified_ocr = UnifiedOCRService(
    ocr_space_key=Config.OCR_SPACE_API_KEY,
    nextocr_endpoint=Config.NEXTOCR_ENDPOINT,
    nextocr_username=Config.NEXTOCR_USERNAME,
    nextocr_secret_key=Config.NEXTOCR_SECRET_KEY,
    use_segmenter=True,
    cache_enabled=True
)

# -------------------------------
# Mount static files
# -------------------------------
app.mount("/api/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

web_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.exists(web_folder):
    app.mount("/", StaticFiles(directory=web_folder, html=True), name="web")

# -------------------------------
# Health check endpoint
# -------------------------------
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    yolo_available = detector.yolo_model is not None
    ocr_configured = bool(Config.OCR_SPACE_API_KEY or (Config.NEXTOCR_USERNAME and Config.NEXTOCR_SECRET_KEY))

    return HealthResponse(
        status="healthy",
        yolo_available=yolo_available,
        ocr_configured=ocr_configured
    )

# -------------------------------
# Detect document corners
# -------------------------------
@app.post("/api/detect-corners", response_model=CornerDetectionResponse)
async def detect_corners(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image.convert("RGB"))

        corners, method = detector.detect(image_array)
        if corners is None:
            h, w = image_array.shape[:2]
            corners = [(0, 0), (w, 0), (w, h), (0, h)]
            method = "fallback"

        filename = generate_unique_filename(file.filename or "image.jpg")
        filepath = os.path.join(UPLOAD_DIR, filename)
        image.save(filepath)

        corner_points = [CornerPoint(x=c[0], y=c[1]) for c in corners]
        preview_url = f"/api/uploads/{filename}"

        return CornerDetectionResponse(
            success=True,
            corners=corner_points,
            method=_to_detection_method(method),
            bounding_box=_corners_to_bbox(corners),
            image_width=image_array.shape[1],
            image_height=image_array.shape[0],
            preview_url=preview_url,
            message=f"Detected using {method}"
        )

    except Exception as e:
        logger.error(f"Corner detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# -------------------------------
# Apply crop
# -------------------------------
@app.post("/api/apply-crop", response_model=ApplyCropResponse)
async def apply_crop(file: UploadFile = File(...), corners: str = ""):
    try:
        corner_data = json.loads(corners)
        corner_points = [(c["x"], c["y"]) for c in corner_data]

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image.convert("RGB"))

        warped = detector.crop_and_transform(image_array, corner_points)

        filename = generate_unique_filename("cropped.jpg")
        filepath = os.path.join(UPLOAD_DIR, filename)

        # Convert RGB -> BGR for cv2.imwrite
        if len(warped.shape) == 3:
            warped_bgr = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, warped_bgr)
        else:
            cv2.imwrite(filepath, warped)

        return ApplyCropResponse(
            success=True,
            cropped_image_url=f"/api/uploads/{filename}",
            width=warped.shape[1],
            height=warped.shape[0]
        )

    except Exception as e:
        logger.error(f"Crop failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Crop failed: {str(e)}")

# -------------------------------
# Full OCR pipeline
# -------------------------------
@app.post("/api/process", response_model=OCRResponse)
async def process_invoice(file: UploadFile = File(...), corners: str = ""):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image.convert("RGB"))

        work_image = image_array
        used_corners = None
        detection_method = "fallback"
        if corners:
            try:
                corner_data = json.loads(corners)
                corner_points = [(c["x"], c["y"]) for c in corner_data]
                work_image = detector.crop_and_transform(image_array, corner_points)
                used_corners = corner_points
                detection_method = "manual"
            except Exception as crop_error:
                logger.warning(f"Invalid corners, using original image: {crop_error}")
        else:
            auto_corners, auto_method = detector.detect(image_array)
            if auto_corners:
                work_image = detector.crop_and_transform(image_array, auto_corners)
                used_corners = auto_corners
                detection_method = auto_method
                logger.info(f"Auto-crop applied via {auto_method}")
            else:
                logger.warning("Auto-crop unavailable, using full image")

        # Multi-pass preprocessing + OCR (preprocess_adaptive returns (image, meta))
        nextocr_enhanced, _ = preprocessor.preprocess_adaptive(work_image.copy(), for_nextocr=True, lang="km")
        nextocr_light, _ = preprocessor.preprocess_adaptive(work_image.copy(), for_nextocr=True, lang="en")
        auto_enhanced, _ = preprocessor.preprocess_adaptive(work_image.copy(), for_nextocr=False, lang="km")

        candidates = [
            ("nextocr_enhanced", nextocr_enhanced, "nextocr"),
            ("nextocr_light", nextocr_light, "nextocr"),
            ("nextocr_original", work_image.copy(), "nextocr"),
            ("auto_enhanced", auto_enhanced, "auto"),
            ("auto_original", work_image.copy(), "auto"),
        ]

        raw_text = ""
        best_processed = candidates[0][1]
        best_score = -1
        best_label = ""
        ocr_error_logs = []

        def score_text(text: str) -> int:
            return len([ch for ch in text if ch.isalnum()])

        for label, candidate_image, method in candidates:
            logger.info(f"OCR attempt: {label}")
            result = ocr_client.extract(candidate_image, method=method)

            if result.get("error"):
                ocr_error_logs.append(f"{label}: {result['error']}")
                logger.warning(f"OCR failed: {label} -> {result['error']}")

            text = (result.get("text") or "").strip()
            if text:
                current_score = score_text(text)
                if current_score > best_score:
                    best_score = current_score
                    raw_text = text
                    best_processed = candidate_image
                    best_label = label

        if not raw_text:
            logger.error("All OCR attempts returned empty text")
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "OCR failed for all attempts",
                    "errors": ocr_error_logs
                }
            )

        invoice_data = extractor.extract(raw_text)

        filename = generate_unique_filename("processed.jpg")
        filepath = os.path.join(UPLOAD_DIR, filename)
        if len(best_processed.shape) == 3:
            cv2.imwrite(filepath, cv2.cvtColor(best_processed, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(filepath, best_processed)

        return OCRResponse(
            success=True,
            data=invoice_data,
            cropped_image_url=f"/api/uploads/{filename}",
            detection_method=_to_detection_method(detection_method),
            detected_corners=[CornerPoint(x=c[0], y=c[1]) for c in used_corners] if used_corners else None,
            bounding_box=_corners_to_bbox(used_corners) if used_corners else None,
            best_ocr_attempt=best_label or None,
            ocr_errors=ocr_error_logs or None,
            message="Processing completed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# -------------------------------
# Unified OCR Pipeline (Enhanced)
# -------------------------------
@app.post("/api/ocr-unified", response_model=UnifiedOCRResponse)
async def process_invoice_unified(
    file: UploadFile = File(...),
    lang: str = "en",
    auto_crop: bool = True,
    multi_pass: bool = True,
    return_structured: bool = False
):
    """
    Enhanced unified OCR pipeline with:
    - AI-powered document detection (U²-Net)
    - Multi-pass preprocessing strategies
    - Smart OCR provider selection
    - Structured data extraction
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image.convert("RGB"))

        # Run unified OCR pipeline
        result = unified_ocr.process(
            image_array,
            lang=lang,
            auto_crop=auto_crop,
            multi_pass=multi_pass,
            return_structured=return_structured
        )

        if not result["success"]:
            failed_attempts = [OCRAttempt(**attempt) for attempt in result.get("all_attempts", [])]

            failed_detection = None
            if result.get("detection"):
                failed_detection = DetectionStage(**result["detection"])

            failed_metadata = None
            if result.get("metadata"):
                stage_data = result.get("metadata", {}).get("pipeline_stages", {})
                failed_metadata = PipelineMetadata(
                    detection=failed_detection,
                    ocr=OCRStage(**stage_data["ocr"]) if stage_data.get("ocr") else None,
                    extraction=ExtractionStage(**stage_data["extraction"]) if stage_data.get("extraction") else None,
                    total_latency=stage_data.get("total_latency", 0.0),
                    stages=result.get("metadata", {})
                )

            return UnifiedOCRResponse(
                success=False,
                all_attempts=failed_attempts,
                detection=failed_detection,
                error=result.get("error", "Unknown error"),
                message="OCR processing failed",
                metadata=failed_metadata
            )

        # Save processed image
        filename = generate_unique_filename("unified_ocr.jpg")
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        # Get the best processed image from OCR result
        # For now, we'll save the original - in production, save the best attempt
        if len(image_array.shape) == 3:
            cv2.imwrite(filepath, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(filepath, image_array)

        # Convert result to response schema
        ocr_result = None
        if result.get("ocr_result"):
            ocr_result = OCRAttempt(**result["ocr_result"])

        all_attempts = [OCRAttempt(**attempt) for attempt in result.get("all_attempts", [])]

        detection_stage = None
        if result.get("detection"):
            detection_stage = DetectionStage(**result["detection"])

        metadata = PipelineMetadata(
            detection=detection_stage,
            ocr=OCRStage(**result.get("metadata", {}).get("pipeline_stages", {}).get("ocr", {})) if result.get("metadata", {}).get("pipeline_stages", {}).get("ocr") else None,
            extraction=ExtractionStage(**result.get("metadata", {}).get("pipeline_stages", {}).get("extraction", {})) if result.get("metadata", {}).get("pipeline_stages", {}).get("extraction") else None,
            total_latency=result.get("metadata", {}).get("pipeline_stages", {}).get("total_latency", 0.0),
            stages=result.get("metadata", {})
        )

        return UnifiedOCRResponse(
            success=True,
            data=result.get("data"),
            ocr_result=ocr_result,
            all_attempts=all_attempts,
            detection=detection_stage,
            processed_image_url=f"/api/uploads/{filename}",
            metadata=metadata,
            message="Unified OCR processing completed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unified OCR processing failed: {e}", exc_info=True)
        return UnifiedOCRResponse(
            success=False,
            error=str(e),
            message="Unified OCR processing failed"
        )


# -------------------------------
# Cache Management
# -------------------------------
@app.post("/api/ocr/clear-cache")
async def clear_ocr_cache():
    """Clear the OCR result cache"""
    unified_ocr.clear_cache()
    return {"success": True, "message": "OCR cache cleared"}


# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)