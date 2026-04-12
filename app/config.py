import os
from glob import glob
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _resolve_model_path(raw_path: str) -> str:
    """Resolve a model path from env var, supporting both absolute and repo-relative paths."""
    if not raw_path:
        return ""

    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return str(candidate)

    if not candidate.is_absolute():
        app_dir = Path(__file__).resolve().parent
        rel_candidate = app_dir / candidate
        if rel_candidate.exists():
            return str(rel_candidate)

    return ""


def _find_default_yolo_model() -> str:
    """Find DocLayout YOLO weights inside app/models snapshots."""
    app_dir = Path(__file__).resolve().parent
    pattern = (
        app_dir
        / "models"
        / "models--juliozhao--DocLayout-YOLO-DocStructBench"
        / "snapshots"
        / "*"
        / "doclayout_yolo_docstructbench_imgsz1024.pt"
    )
    matches = sorted(glob(str(pattern)))
    return matches[0] if matches else ""


class Config:
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "")
    NEXTOCR_ENDPOINT = os.getenv(
        "NEXTOCR_ENDPOINT", "https://developer.nextocr.org/ocr_api"
    )
    NEXTOCR_USERNAME = os.getenv("NEXTOCR_USERNAME", "")
    NEXTOCR_SECRET_KEY = os.getenv("NEXTOCR_SECRET_KEY", "")
    QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
    QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
    QWEN_BASE_URL = os.getenv(
        "QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    QWEN_TIMEOUT_SECONDS = int(os.getenv("QWEN_TIMEOUT_SECONDS", 20))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    QWEN_ENABLE_DEBUG = os.getenv("QWEN_ENABLE_DEBUG", "false").lower() == "true"
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))  # 10MB default
    UPLOAD_FOLDER = os.getenv(
        "UPLOAD_FOLDER", "/tmp/uploads" if os.getenv("VERCEL") else "uploads"
    )
    ALLOWED_ORIGINS = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN", "")
    HF_TOKEN = os.getenv("HF_TOKEN", "") or HF_HUB_TOKEN
    UNET_MODEL = _resolve_model_path(os.getenv("UNET_MODEL", "")) or os.getenv(
        "U2NET_MODEL_PATH", "/tmp/u2net.onnx"
    )
    YOLO_MODEL = (
        _resolve_model_path(os.getenv("YOLO_MODEL", "")) or _find_default_yolo_model()
    )
