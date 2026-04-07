import os
import uuid


def generate_unique_filename(original_filename: str) -> str:
    ext = os.path.splitext(original_filename)[1] or ".jpg"
    return f"{uuid.uuid4().hex}{ext}"


def ensure_dir_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)
