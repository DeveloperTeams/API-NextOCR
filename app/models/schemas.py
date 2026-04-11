import re
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class CornerPoint(BaseModel):
    x: float
    y: float


class DetectionMethod(str, Enum):
    YOLO = "yolo"
    UNET = "unet"
    OPENCV = "opencv"
    FALLBACK = "fallback"


class LineItem(BaseModel):
    """Represents a single line item on an invoice."""
    name: str = Field(..., description="Item name with modifiers if applicable")
    quantity: int = Field(default=1, ge=1, description="Quantity must be >= 1")
    price: float = Field(..., ge=0, description="Unit price in USD")
    total: float = Field(..., ge=0, description="Line total = quantity × price")

    @field_validator('price', 'total', mode='before')
    @classmethod
    def parse_monetary(cls, v):
        """Convert string amounts like '$2.43' or '2,43' to float."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return round(float(v), 2)
        if isinstance(v, str):
            # Remove currency symbols, commas, whitespace
            cleaned = re.sub(r'[^\d.\-]', '', v.strip())
            if not cleaned or cleaned == '-':
                return 0.0
            try:
                return round(float(cleaned), 2)
            except ValueError:
                return 0.0
        return 0.0

    @model_validator(mode='after')
    def validate_total(self) -> 'LineItem':
        """Ensure total matches quantity × price (with small tolerance for rounding)."""
        if self.quantity and self.price:
            expected = round(self.quantity * self.price, 2)
            if abs(self.total - expected) > 0.01:
                # Auto-correct if mismatch is small
                self.total = expected
        return self

    @classmethod
    def create(cls, name: str, price: float, quantity: int = 1, total: Optional[float] = None):
        """Factory method with auto-calculated total."""
        calc_total = round(quantity * price, 2) if total is None else round(total, 2)
        return cls(name=name, price=price, quantity=quantity, total=calc_total)


class PaymentInfo(BaseModel):
    """Payment summary section of an invoice."""
    subtotal: Optional[float] = Field(None, ge=0, description="Subtotal before tax/discount")
    tax: Optional[float] = Field(None, ge=0, description="Tax amount")
    total: Optional[float] = Field(None, ge=0, description="Final total amount")
    discount_usd: Optional[float] = Field(None, ge=0, description="Discount amount in USD")
    method: Optional[str] = Field(None, description="Payment method: cash, ABA, card, etc.")

    @field_validator('subtotal', 'tax', 'total', 'discount_usd', mode='before')
    @classmethod
    def parse_monetary(cls, v):
        """Convert string amounts to float, handling OCR artifacts."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return round(float(v), 2)
        if isinstance(v, str):
            cleaned = re.sub(r'[^\d.\-]', '', v.strip())
            if not cleaned or cleaned == '-':
                return None
            try:
                return round(float(cleaned), 2)
            except ValueError:
                return None
        return None

    @model_validator(mode='after')
    def validate_payment_logic(self) -> 'PaymentInfo':
        """Basic payment logic validation."""
        if self.subtotal is not None and self.total is not None:
            # Total should be >= subtotal - discount
            min_expected = self.subtotal - (self.discount_usd or 0)
            if self.total < min_expected - 0.01:  # tolerance for rounding
                # Auto-correct if close
                self.total = round(min_expected, 2)
        return self


class InvoiceData(BaseModel):
    """
    Structured invoice data with promoted common fields.
    dynamic_fields is for truly custom/unknown fields only.
    """

    merchant_name: Optional[str] = Field(None, description="Merchant/business name")
    merchant_address: Optional[str] = Field(None, description="Full merchant address")
    merchant_phone: Optional[str] = Field(None, description="Phone number, normalized")
    
    invoice_number: Optional[str] = Field(None, description="Invoice/reference number")
    invoice_date: Optional[str] = Field(None, description="Date in YYYY-MM-DD format preferred")
    invoice_time: Optional[str] = Field(None, description="Time in HH:MM format preferred")
    
    items: List[LineItem] = Field(default_factory=list, description="List of purchased items")
    payment: Optional[PaymentInfo] = Field(default=None, description="Payment breakdown")
    cashier_name: Optional[str] = Field(None, description="Cashier name from receipt")
    exchange_rate: Optional[str] = Field(None, description="Exchange rate string, e.g., '4100KHR=$1'")
    total_khr: Optional[float] = Field(None, ge=0, description="Total amount in Cambodian Riel")
    discount_usd: Optional[float] = Field(None, ge=0, description="Discount in USD (also in payment)")
    payment_method: Optional[str] = Field(None, description="Payment method: cash, ABA, card, etc.")
    
    dynamic_fields: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional custom fields not covered by schema"
    )
    
    raw_text: str = Field(default="", description="Original OCR text, unchanged")

    @field_validator('merchant_phone', mode='before')
    @classmethod
    def normalize_phone(cls, v):
        """Normalize phone: '012589 469' → '012589469'."""
        if v is None:
            return None
        if isinstance(v, str):
            return re.sub(r'[^\d+]', '', v.strip()) or None
        return v

    @field_validator('total_khr', 'discount_usd', mode='before')
    @classmethod
    def parse_monetary_optional(cls, v):
        """Parse optional monetary fields."""
        if v is None or (isinstance(v, str) and v.strip() == ''):
            return None
        if isinstance(v, (int, float)):
            return round(float(v), 2)
        if isinstance(v, str):
            # Remove all non-numeric chars except digits, dot, minus
            cleaned = re.sub(r'[^\d.\-]', '', v.strip())
            if not cleaned or cleaned == '-' or cleaned == '.':
                return None
            try:
                return round(float(cleaned), 2)
            except ValueError:
                return None
        return None

    @model_validator(mode='after')
    def clean_dynamic_fields(self) -> 'InvoiceData':
        """
        Auto-promote common keys from dynamic_fields to first-class fields.
        Removes promoted keys from dynamic_fields to avoid duplication.
        """
        # Mapping: dynamic key patterns → target field name
        promotion_map = {
            'cashier_name': ['cashier', 'cashlef', 'cashiername', ' cashier '],
            'exchange_rate': ['exchange', 'exchangerate', 'rate', 'khrtousd'],
            'total_khr': ['total_khr', 'totalkhr', 'total_kh', 'totalkh', 'total_khr:', 'total (khr)'],
            'discount_usd': ['discount', 'discount_usd', 'disc', 'disc_usd'],
            'payment_method': ['method', 'payment_method', 'paymentmethod', 'paymethod', 'bank'],
        }

        # Monetary fields that need float conversion
        monetary_fields = {'total_khr', 'discount_usd'}

        if not self.dynamic_fields:
            return self

        # Work on a copy to avoid modifying during iteration
        to_promote = {}
        to_remove = []

        for key, value in self.dynamic_fields.items():
            key_lower = key.lower().strip()
            for target_field, patterns in promotion_map.items():
                # Check if key matches any pattern
                if any(pattern in key_lower for pattern in patterns):
                    # Only promote if target field is not already set
                    if getattr(self, target_field, None) is None:
                        # Convert monetary fields to float
                        if target_field in monetary_fields:
                            to_promote[target_field] = self._parse_monetary_value(value)
                        else:
                            to_promote[target_field] = value
                    to_remove.append(key)
                    break

        # Apply promotions
        for field_name, field_value in to_promote.items():
            setattr(self, field_name, field_value)

        # Remove promoted keys from dynamic_fields
        for key in to_remove:
            self.dynamic_fields.pop(key, None)

        return self

    @staticmethod
    def _parse_monetary_value(value: Any) -> Optional[float]:
        """Parse a monetary value (string or number) to float, handling OCR artifacts."""
        if value is None or (isinstance(value, str) and value.strip() == ''):
            return None
        if isinstance(value, (int, float)):
            return round(float(value), 2)
        if isinstance(value, str):
            cleaned = re.sub(r'[^\d.\-]', '', value.strip())
            if not cleaned or cleaned == '-' or cleaned == '.':
                return None
            try:
                return round(float(cleaned), 2)
            except ValueError:
                return None
        return None

    class Config:
        populate_by_name = True
        extra = 'allow'  # Allow extra fields in input, but they won't be in output


# === Response Schemas ===

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    yolo_available: bool = False
    ocr_configured: bool = False


class ProcessingInfo(BaseModel):
    """Metadata about the preprocessing pipeline."""
    detection_method: str
    bounding_box: Dict[str, float]
    confidence: float
    provider: str
    latency: float
    cropped_image_url: Optional[str] = None
    image_width: int
    image_height: int
    llm_enhancement: Optional[Dict[str, Any]] = Field(
        default=None,
        description="LLM enhancement metadata: enabled, applied, error, model, validation_passed"
    )


class PreprocessResponse(BaseModel):
    """Main response schema for invoice preprocessing endpoint."""
    success: bool
    invoice_data: Optional[InvoiceData] = None
    processing_info: ProcessingInfo
    message: str = ""

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "invoice_data": {
                    "merchant_name": "Aorsee (45 COFFEE)",
                    "invoice_number": "C26-11756",
                    "invoice_date": "2026-03-22",
                    "items": [{"name": "Iced Late", "quantity": 1, "price": 2.43, "total": 2.43}],
                    "payment": {"subtotal": 2.43, "total": 2.43},
                    "cashier_name": "sreyne"
                },
                "processing_info": {
                    "detection_method": "unet",
                    "bounding_box": {"x": 30, "y": 22, "width": 569, "height": 1257},
                    "confidence": 0.98,
                    "provider": "nextocr",
                    "latency": 10.073,
                    "image_width": 1200,
                    "image_height": 2627
                },
                "message": "Invoice preprocessed successfully"
            }
        }


class YOLODetectionInfo(BaseModel):
    """YOLO-specific detection metadata."""
    detection_method: str = "yolo"
    confidence: float = Field(ge=0, le=1)
    bounding_box: Dict[str, float]
    num_boxes_detected: int = Field(ge=0)
    merged_boxes: bool = False
    image_width: int
    image_height: int
    quality_metrics: Optional[Dict[str, Any]] = None


class YOLOPreprocessResponse(BaseModel):
    """Response schema for YOLO + NextOCR endpoint."""
    success: bool
    invoice_data: Optional[InvoiceData] = None
    detection_info: YOLODetectionInfo
    ocr_confidence: float = Field(ge=0, le=1, default=0.0)
    processing_time_ms: float = Field(ge=0, default=0.0)
    processed_image_url: Optional[str] = None
    visualization_url: Optional[str] = None
    message: str = ""
    metadata: Optional[Dict[str, Any]] = None