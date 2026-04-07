from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum


class CornerPoint(BaseModel):
    x: float
    y: float


class DetectionMethod(str, Enum):
    YOLO = "yolo"
    UNET = "unet"
    OPENCV = "opencv"
    FALLBACK = "fallback"


class LineItem(BaseModel):
    name: str
    quantity: int = 1
    price: float
    total: float

    @classmethod
    def create(cls, name: str, price: float, quantity: int = 1):
        return cls(name=name, price=price, quantity=quantity, total=price * quantity)


class PaymentInfo(BaseModel):
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    method: Optional[str] = None


class InvoiceData(BaseModel):
    merchant_name: Optional[str] = None
    merchant_address: Optional[str] = None
    merchant_phone: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    invoice_time: Optional[str] = None
    items: List[LineItem] = []
    payment: Optional[PaymentInfo] = None
    dynamic_fields: Dict[str, Any] = {}
    raw_text: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    yolo_available: bool = False
    ocr_configured: bool = False


class PreprocessResponse(BaseModel):
    success: bool
    invoice_data: Optional[InvoiceData] = None
    processing_info: dict
    message: str = ""
