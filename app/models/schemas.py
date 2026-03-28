from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class CornerPoint(BaseModel):
    """Represents a corner point with x, y coordinates"""
    x: float
    y: float


class DetectionMethod(str, Enum):
    """Document detection method"""
    YOLO = "yolo"
    UNET = "unet"
    OPENCV = "opencv"
    FALLBACK = "fallback"


class CornerDetectionRequest(BaseModel):
    """Request for corner detection"""
    pass


class CornerDetectionResponse(BaseModel):
    """Response for corner detection"""
    success: bool
    corners: List[CornerPoint]
    method: DetectionMethod
    bounding_box: Optional[Dict[str, float]] = None
    image_width: int
    image_height: int
    preview_url: Optional[str] = None
    message: str = ""


class ApplyCropRequest(BaseModel):
    """Request to apply crop with user-adjusted corners"""
    corners: List[CornerPoint]
    image_name: str


class ApplyCropResponse(BaseModel):
    """Response after applying crop"""
    success: bool
    cropped_image_url: str
    width: int
    height: int


class LineItem(BaseModel):
    """Represents a line item in the invoice"""
    name: str
    quantity: Optional[float] = None
    price: Optional[float] = None
    total: Optional[float] = None


class PaymentInfo(BaseModel):
    """Payment information"""
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    method: Optional[str] = None


class InvoiceData(BaseModel):
    """Structured invoice data"""
    merchant_name: Optional[str] = None
    merchant_address: Optional[str] = None
    merchant_phone: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    invoice_time: Optional[str] = None
    items: List[LineItem] = []
    payment: Optional[PaymentInfo] = None
    dynamic_fields: Optional[Dict[str, Any]] = None
    raw_text: str = ""


class OCRRequest(BaseModel):
    """OCR processing request"""
    image_name: str
    corners: Optional[List[CornerPoint]] = None


class OCRResponse(BaseModel):
    """OCR processing response"""
    success: bool
    data: InvoiceData
    cropped_image_url: Optional[str] = None
    detection_method: Optional[DetectionMethod] = None
    detected_corners: Optional[List[CornerPoint]] = None
    bounding_box: Optional[Dict[str, float]] = None
    best_ocr_attempt: Optional[str] = None
    ocr_errors: Optional[List[str]] = None
    message: str = ""


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0.0"
    yolo_available: bool = False
    ocr_configured: bool = False


# -------------------------------
# Enhanced OCR Schemas
# -------------------------------

class OCRAttempt(BaseModel):
    """Single OCR attempt result"""
    text: str = ""
    confidence: float = 0.0
    provider: str = "unknown"
    preprocessing: str = "unknown"
    latency: float = 0.0
    structured: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DetectionStage(BaseModel):
    """Document detection stage info"""
    success: bool = False
    method: str = "none"
    original_size: Optional[tuple] = None
    cropped_size: Optional[tuple] = None


class OCRStage(BaseModel):
    """OCR stage info"""
    total_attempts: int = 0
    successful_attempts: int = 0
    best_provider: str = "unknown"
    best_preprocessing: str = "unknown"
    best_confidence: float = 0.0


class ExtractionStage(BaseModel):
    """Data extraction stage info"""
    merchant_detected: bool = False
    items_count: int = 0
    total_amount: Optional[float] = None


class PipelineMetadata(BaseModel):
    """Full pipeline execution metadata"""
    detection: Optional[DetectionStage] = None
    ocr: Optional[OCRStage] = None
    extraction: Optional[ExtractionStage] = None
    total_latency: float = 0.0
    stages: Dict[str, Any] = {}


class UnifiedOCRRequest(BaseModel):
    """Unified OCR processing request"""
    lang: str = "en"
    auto_crop: bool = True
    multi_pass: bool = True
    return_structured: bool = False


class UnifiedOCRResponse(BaseModel):
    """Unified OCR processing response"""
    success: bool
    data: Optional[InvoiceData] = None
    ocr_result: Optional[OCRAttempt] = None
    all_attempts: List[OCRAttempt] = []
    detection: Optional[DetectionStage] = None
    processed_image_url: Optional[str] = None
    metadata: Optional[PipelineMetadata] = None
    error: Optional[str] = None
    message: str = ""
