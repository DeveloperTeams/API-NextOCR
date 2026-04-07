import os
import io
import cv2
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Tuple, Optional
from .config import Config
from app.models.schemas import DetectionMethod, HealthResponse, PreprocessResponse
from app.services.document_detector import DocumentDetector
from app.services.image_preprocessor import ImagePreprocessor
from app.services.ocr_client import OCRClient
from app.services.data_extractor import DataExtractor
from app.utils.helpers import get_safe_original_filename, ensure_dir_exists

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="E-Invoice Engine", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), Config.UPLOAD_FOLDER)
ensure_dir_exists(UPLOAD_DIR)

# Initialize services
detector = DocumentDetector(yolo_model_path=Config.YOLO_MODEL)
preprocessor = ImagePreprocessor()
ocr_client = OCRClient(
    ocr_space_key=Config.OCR_SPACE_API_KEY,
    nextocr_endpoint=Config.NEXTOCR_ENDPOINT,
    nextocr_username=Config.NEXTOCR_USERNAME,
    nextocr_secret_key=Config.NEXTOCR_SECRET_KEY,
)
data_extractor = DataExtractor()


def _corners_to_bbox(corners: List[Tuple[float, float]]) -> dict:
    xs = [float(c[0]) for c in corners]
    ys = [float(c[1]) for c in corners]
    return {
        "x": min(xs),
        "y": min(ys),
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
    }


def _to_detection_method(method: str) -> DetectionMethod:
    if method == "yolo":
        return DetectionMethod.YOLO
    if method == "unet":
        return DetectionMethod.UNET
    if method == "opencv":
        return DetectionMethod.OPENCV
    return DetectionMethod.FALLBACK


# -------------------------------
# Health check endpoint
# -------------------------------
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    yolo_available = detector.yolo_model is not None
    ocr_configured = bool(
        Config.OCR_SPACE_API_KEY
        or (Config.NEXTOCR_USERNAME and Config.NEXTOCR_SECRET_KEY)
    )

    return HealthResponse(
        status="healthy", yolo_available=yolo_available, ocr_configured=ocr_configured
    )


# -------------------------------
# Single preprocessing endpoint - Core MVP functionality with NextOCR
# -------------------------------
@app.post("/api/preprocess")
async def preprocess_invoice(file: UploadFile = File(...)):
    """
    Single endpoint for invoice preprocessing pipeline with NextOCR:
    1. Detect invoice using YOLOv10 (primary) with fallback to U2-Net/OpenCV
    2. Crop invoice tightly around document boundaries
    3. Apply adaptive preprocessing optimized for NextOCR
    4. Run NextOCR on the processed image
    5. Extract structured invoice data from OCR text
    6. Return structured invoice data and processing metadata

    Input: Raw invoice image (JPEG/PNG)
    Output: Structured invoice data from NextOCR with bounding box for ML consumption
    """
    try:
        # Read and convert image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image.convert("RGB"))

        logger.info(f"Processing image: {image_array.shape}")

        # Step 1: Detect document corners using hybrid approach (YOLO first)
        corners, method = detector.detect(image_array)

        # If no corners detected, use full image as fallback
        if corners is None:
            h, w = image_array.shape[:2]
            corners = [(0, 0), (w, 0), (w, h), (0, h)]
            method = "fallback"
            logger.warning("No document detected, using full image")

        # Step 2: Apply perspective transform to get cropped image
        cropped_image = detector.crop_and_transform(image_array, corners)

        # Step 3: Apply adaptive preprocessing optimized for NextOCR
        processed_image, _ = preprocessor.preprocess_adaptive(
            cropped_image.copy(),
            for_nextocr=True,
            lang="en",
            auto_crop=False,  # Already cropped
            use_segmenter=False,  # Already detected/cropped
        )

        # Save processed image using the original upload name.
        filename = get_safe_original_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)

        # Convert RGB to BGR for OpenCV saving
        if len(processed_image.shape) == 3:
            cv2.imwrite(filepath, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(filepath, processed_image)

        # Step 4: Run NextOCR on the processed image
        ocr_result = ocr_client.extract(processed_image, method="nextocr", lang="en")

        if ocr_result.get("error"):
            logger.error(f"NextOCR failed: {ocr_result['error']}")
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "NextOCR processing failed",
                    "error": ocr_result["error"],
                },
            )

        extracted_text = ocr_result.get("text", "").strip()
        if not extracted_text:
            logger.warning("NextOCR returned empty text")
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "NextOCR returned empty text",
                    "error": "No text extracted from image",
                },
            )

        logger.info(f"NextOCR successful. Extracted {len(extracted_text)} characters")

        # Step 5: Extract structured data from OCR text
        invoice_data = data_extractor.extract(extracted_text)

        # Calculate bounding box from detected corners
        bbox = _corners_to_bbox(corners)

        # Prepare processing info
        processing_info = {
            "detection_method": method,
            "bounding_box": bbox,
            "confidence": ocr_result.get("confidence", 0.0),
            "provider": ocr_result.get("provider", "nextocr"),
            "latency": ocr_result.get("latency", 0.0),
            "cropped_image_url": f"/api/uploads/{filename}",
            "image_width": processed_image.shape[1],
            "image_height": processed_image.shape[0],
        }

        return PreprocessResponse(
            success=True,
            invoice_data=invoice_data,
            processing_info=processing_info,
            message=f"Invoice preprocessed and processed with NextOCR using {method} detection",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


# -------------------------------
# Mount static files
# -------------------------------
app.mount("/api/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

web_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.exists(web_folder):
    app.mount("/", StaticFiles(directory=web_folder, html=True), name="web")

# -------------------------------
# Run app
# -------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
