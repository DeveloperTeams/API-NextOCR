import os
import io
import uuid
from typing import Optional, Tuple
from PIL import Image
import base64


def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename preserving the extension"""
    ext = os.path.splitext(original_filename)[1] or ".jpg"
    return f"{uuid.uuid4().hex}{ext}"


def save_uploaded_file(file_data: bytes, upload_folder: str, filename: Optional[str] = None) -> str:
    """Save uploaded file and return the path"""
    os.makedirs(upload_folder, exist_ok=True)
    
    if filename is None:
        filename = generate_unique_filename("image.jpg")
    
    filepath = os.path.join(upload_folder, filename)
    with open(filepath, "wb") as f:
        f.write(file_data)
    
    return filepath


def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string"""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_data}"


def pil_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_data}"


def validate_image(file_data: bytes, max_size: int) -> tuple[bool, str]:
    """Validate image file"""
    if len(file_data) > max_size:
        return False, f"File size exceeds maximum limit ({max_size / 1024 / 1024:.1f}MB)"
    
    try:
        image = Image.open(io.BytesIO(file_data))
        image.verify()
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def ensure_dir_exists(path: str) -> None:
    """Ensure directory exists, create if not"""
    os.makedirs(path, exist_ok=True)
