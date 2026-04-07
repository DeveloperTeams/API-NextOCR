import os
import uuid

def generate_unique_filename(original_filename: str) -> str:
    ext = os.path.splitext(original_filename)[1] or ".jpg"
    return f"{uuid.uuid4().hex}{ext}"

def get_safe_original_filename(original_filename: str) -> str:
    # Keep client-provided name but strip any path components for safety.
    filename = os.path.basename((original_filename or "").strip())
    if not filename:
        return generate_unique_filename("preprocessed.jpg")

    name, ext = os.path.splitext(filename)
    ext = ext or ".jpg"
    return f"{name}{ext}"

def ensure_dir_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)
