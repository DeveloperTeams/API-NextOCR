import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))

    OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")
    
    # NextOCR Configuration
    NEXTOCR_ENDPOINT = os.getenv("NEXTOCR_ENDPOINT", "https://developer.nextocr.org/ocr_api")
    NEXTOCR_USERNAME = os.getenv("NEXTOCR_USERNAME", "")
    NEXTOCR_SECRET_KEY = os.getenv("NEXTOCR_SECRET_KEY", "")

    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))  # 10MB default
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

    HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN", "")

    # YOLO model path (empty string means use pretrained model)
    YOLO_MODEL = os.getenv("YOLO_MODEL", "") or None
