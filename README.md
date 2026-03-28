# Invoice OCR API

A production-focused FastAPI backend for extracting invoice and receipt data from images.

It supports:

- Automatic document detection and perspective correction
- Multi-pass OCR with provider fallback
- Khmer and English invoice processing
- Structured field extraction (merchant, date, totals, line items)
- Detailed metadata to help debug OCR quality

---

## Table of Contents

- Overview
- Features
- Tech Stack
- Project Structure
- Quick Start
  - Local Setup
  - Docker Setup
- Configuration
- Run the API
- API Endpoints
- Example Workflow
- Supported Image Formats
- API Performance & Limits
- Supported OCR Providers
- Troubleshooting
- Known Limitations
- Getting Help
- Notes for Production

---

## Overview

This service receives an image of an invoice/receipt, optionally detects document corners, crops and enhances the image, then runs OCR and maps the output into structured JSON.

Main use cases:

- Mobile or web apps that upload receipt photos
- Automation pipelines for expense processing
- Khmer/English invoice extraction with fallback OCR strategies

---

## Features

- Hybrid document detection pipeline:
  - YOLOv10 (fast path)
  - U2-Net segmentation fallback
  - OpenCV contour fallback
- Perspective transform for clean, scanner-like crops
- Adaptive preprocessing (denoise, contrast, resize, optional super-resolution)
- OCR strategy routing:
  - NextOCR (header-auth, multipart upload)
  - OCR.space fallback
- Multi-attempt scoring and best-result selection
- Structured extraction:
  - Merchant info
  - Invoice date/time/number
  - Payment totals
  - Line items
  - Dynamic key-value fields
- Upload preview/static serving for processed images

---

## Tech Stack

- Python 3.11
- FastAPI + Uvicorn
- OpenCV + NumPy + Pillow
- PyTorch ecosystem for model-backed processing
- doclayout-yolo (YOLO document detection)
- U2-Net ONNX segmentation

---

## Project Structure

```text
api/
	app/
		config.py                  # Environment-driven settings
		main.py                    # FastAPI app and endpoints
		models/schemas.py          # Request/response schemas
		services/
			document_detector.py     # YOLO/U2-Net/OpenCV corner detection
			document_segmenter.py    # U2-Net segmentation support
			image_preprocessor.py    # Adaptive preprocessing pipeline
			ocr_client.py            # OCR provider routing + retries + cache
			ocr_service.py           # Unified OCR pipeline orchestration
			data_extractor.py        # OCR text -> structured invoice fields
		uploads/                   # Saved preview/processed images
	main.py                      # Minimal local entry script
	requirements.txt
	pyproject.toml
```

---

## Quick Start

### Local Setup

#### 1. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows (Git Bash):

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support (optional):

```bash
pip install onnxruntime-gpu
```

#### 3. Configure environment variables

Create a `.env` file in the project root:

```env
HOST=0.0.0.0
PORT=8000

# OCR.space (optional, fallback provider)
OCR_SPACE_API_KEY=

# NextOCR (recommended for Khmer and multilingual receipts)
NEXTOCR_ENDPOINT=https://developer.nextocr.org/ocr_api
NEXTOCR_USERNAME=
NEXTOCR_SECRET_KEY=

# Optional
MAX_FILE_SIZE=10485760
UPLOAD_FOLDER=uploads
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
HF_HUB_TOKEN=
YOLO_MODEL=
```

### Docker Setup

Build and run using Docker:

```bash
docker build -t invoice-ocr .
docker run -p 8000:8000 --env-file .env invoice-ocr
```

Or with Docker Compose:

```bash
docker-compose up
```

Example `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Configuration

The app loads environment values from .env through dotenv.

Key settings:

- HOST: API bind address
- PORT: API port
- NEXTOCR_ENDPOINT, NEXTOCR_USERNAME, NEXTOCR_SECRET_KEY: NextOCR credentials
- OCR_SPACE_API_KEY: OCR.space key
- UPLOAD_FOLDER: directory mounted at /api/uploads
- YOLO_MODEL: optional local model path, otherwise pretrained path is used

---

## Run the API

Option A (recommended):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Option B:

```bash
python app/main.py
```

Then open:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/api/health

---

## API Endpoints

### GET /api/health

Returns service and integration status.

Response fields:

- status
- version
- yolo_available
- ocr_configured

---

### POST /api/detect-corners

Detects document corners from an uploaded image.

**Request:**

- multipart/form-data
- `file`: image file

**Response highlights:**

- `corners`: 4 corner points `[{"x", "y"}, ...]`
- `method`: `yolo` | `unet` | `opencv` | `fallback`
- `bounding_box`: `{x, y, width, height}`
- `preview_url`: URL to preview image

**Examples:**

cURL:

```bash
curl -X POST "http://localhost:8000/api/detect-corners" \
  -F "file=@sample-receipt.jpg"
```

Python:

```python
import requests
with open("sample-receipt.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/api/detect-corners", files=files)
    print(response.json())
```

---

### POST /api/apply-crop

Applies perspective crop using user-provided corners.

Request:

- multipart/form-data
- file: image file
- corners: JSON string array, example:
  [{"x":10,"y":10},{"x":500,"y":20},{"x":490,"y":900},{"x":5,"y":890}]

Response:

- cropped_image_url
- width
- height

---

### POST /api/process

Classic full OCR pipeline endpoint.

**Behavior:**

- If `corners` are provided, manual crop is used.
- If `corners` are not provided, auto-detection attempts: YOLO → U2-Net → OpenCV.
- Runs multiple OCR attempts and selects best text by quality score.
- Returns structured invoice data and OCR diagnostics.

**Request:**

- multipart/form-data
- `file`: image file (required)
- `corners`: optional JSON string array

**Examples:**

cURL:

```bash
curl -X POST "http://localhost:8000/api/process" \
  -F "file=@sample-receipt.jpg"
```

Python:

```python
import requests
import json

with open("sample-receipt.jpg", "rb") as f:
    files = {"file": f}
    data = {"corners": json.dumps([{"x":0,"y":0},{"x":500,"y":0},{"x":500,"y":800},{"x":0,"y":800}])}
    response = requests.post("http://localhost:8000/api/process", files=files, data=data)
    result = response.json()
    print(f"Merchant: {result['data']['merchant_name']}")
    print(f"Total: {result['data']['payment']['total']}")
```

Success response (shape):

```json
{
  "success": true,
  "data": {
    "merchant_name": "45 COFFEE",
    "merchant_address": "#32 St. 432 ...",
    "merchant_phone": "012 589 469",
    "invoice_number": "C26-11756",
    "invoice_date": "22-03-2026",
    "invoice_time": "10:54",
    "items": [
      {
        "name": "Iced Latte",
        "quantity": 1,
        "price": 2.43,
        "total": 2.43
      }
    ],
    "payment": {
      "subtotal": 2.43,
      "tax": 0,
      "total": 2.43,
      "method": "Cash"
    },
    "dynamic_fields": {},
    "raw_text": "..."
  },
  "cropped_image_url": "/api/uploads/processed_xxx.jpg",
  "detection_method": "unet",
  "detected_corners": [{ "x": 0, "y": 0 }],
  "bounding_box": { "x": 0, "y": 0, "width": 100, "height": 200 },
  "best_ocr_attempt": "nextocr_enhanced",
  "ocr_errors": [],
  "message": "Processing completed successfully"
}
```

**Failure response (HTTP 502):**

```json
{
  "detail": {
    "message": "OCR failed for all attempts",
    "errors": [
      "nextocr_enhanced: Empty OCR response",
      "auto_enhanced: Network timeout"
    ]
  }
}
```

---

### POST /api/ocr-unified

Enhanced unified OCR endpoint with richer metadata and multi-pass support.

**Parameters:**

- `file`: UploadFile (required)
- `lang`: `en` | `km` (default `en`)
- `auto_crop`: `true`/`false` (default `true`)
- `multi_pass`: `true`/`false` (default `true`)
- `return_structured`: `true`/`false` (default `false`)

**Example:**

```bash
curl -X POST "http://localhost:8000/api/ocr-unified?lang=km&auto_crop=true&multi_pass=true" \
  -F "file=@sample-receipt.jpg"
```

**Response includes:**

- Best OCR attempt with confidence and latency
- All OCR attempts (pipeline stages and fallbacks)
- Detection stage metadata (method, cropped dimensions)
- Extraction stage metadata (merchant detected, item count, total)
- Processed image URL

---

### POST /api/ocr/clear-cache

Clears in-memory OCR cache.

Example:

```bash
curl -X POST "http://localhost:8000/api/ocr/clear-cache"
```

---

## Example Workflow

For interactive clients with manual corner adjustment:

1. Upload image to /api/detect-corners
2. Let user tweak corners in UI
3. Send final corners to /api/apply-crop (optional preview)
4. Send image + corners to /api/process for extraction

For server-side automation:

1. Send image directly to /api/ocr-unified
2. Enable auto_crop and multi_pass
3. Read structured data from response.data

---

## Supported Image Formats

- **JPEG/JPG**: Best for photos and scans
- **PNG**: With or without transparency
- **BMP**: Uncompressed format
- **TIFF**: Multi-page support (first page processed)
- **WebP**: Modern compressed format

**Max file size (configurable):** 10 MB by default

**Recommended specs:**

- Resolution: 1200x1600 pixels or higher
- Quality: Clear, well-lit document
- Angle: Ideally straight-on (auto-correction available)

---

## API Performance & Limits

| Stage                | Typical Duration | Notes                       |
| -------------------- | ---------------- | --------------------------- |
| Detection            | 100-200ms        | YOLO first, U2-Net fallback |
| Preprocessing        | 50-100ms         | Varies by strategy          |
| OCR (NextOCR)        | 800-1500ms       | Network dependent           |
| OCR (OCR.space)      | 500-1200ms       | Fallback provider           |
| Extraction           | 20-50ms          | Regex and pattern matching  |
| **Total (cached)**   | **150-250ms**    | After first request         |
| **Total (uncached)** | **1500-3000ms**  | Full pipeline               |

**Concurrency limits:**

- Default: No hard limit (depends on system resources)
- Recommended: 10-50 concurrent requests per instance
- For higher load: Use load balancer + multiple instances

**Memory usage:**

- Per instance: ~2-3 GB (with model weights loaded)
- Per request: ~50-100 MB depending on image size

---

## Supported Image Formats

- **JPEG/JPG**: Best for photos and scans
- **PNG**: With or without transparency
- **BMP**: Uncompressed format
- **TIFF**: Multi-page support (first page processed)
- **WebP**: Modern compressed format

**Max file size (configurable):** 10 MB by default

**Recommended specs:**

- Resolution: 1200x1600 pixels or higher
- Quality: Clear, well-lit document
- Angle: Ideally straight-on (auto-correction available)

---

## API Performance & Limits

| Stage                | Typical Duration | Notes                       |
| -------------------- | ---------------- | --------------------------- |
| Detection            | 100-200ms        | YOLO first, U2-Net fallback |
| Preprocessing        | 50-100ms         | Varies by strategy          |
| OCR (NextOCR)        | 800-1500ms       | Network dependent           |
| OCR (OCR.space)      | 500-1200ms       | Fallback provider           |
| Extraction           | 20-50ms          | Regex and pattern matching  |
| **Total (cached)**   | **150-250ms**    | After first request         |
| **Total (uncached)** | **1500-3000ms**  | Full pipeline               |

**Concurrency limits:**

- Default: No hard limit (depends on system resources)
- Recommended: 10-50 concurrent requests per instance
- For higher load: Use load balancer + multiple instances

**Memory usage:**

- Per instance: ~2-3 GB (with model weights loaded)
- Per request: ~50-100 MB depending on image size

---

## Supported OCR Providers

### NextOCR

Auth method:

- X-Username header
- X-Secret-Key header

Upload format:

- multipart/form-data with field name file

### OCR.space

Configured through OCR_SPACE_API_KEY and used as fallback when available.

---

## Troubleshooting

### 1) Health says ocr_configured is false

Check at least one provider is configured:

- NextOCR: NEXTOCR_USERNAME + NEXTOCR_SECRET_KEY
- OCR.space: OCR_SPACE_API_KEY

### 2) YOLO unavailable in /api/health

This can happen when model dependencies or download access are missing.
The service will still run using U2-Net/OpenCV fallback paths.

### 3) Empty OCR result

Try:

- Better lighting and less blur
- Higher-resolution image
- /api/ocr-unified with multi_pass=true
- lang=km for Khmer-heavy receipts

### 4) Slow first request

First run may download/load models and warm up OCR paths.
Subsequent requests are typically faster.

### 5) CUDA/ONNX issues

If GPU runtime is problematic on your machine, use CPU-only dependencies or align your CUDA/onnxruntime versions.

### 6) Corner detection failing

If corners are not detected:

- Check image quality (not too dark/blurry)
- Try with `lang=km` for Khmer documents
- Check /api/health to see if YOLO is available
- Falls back gracefully to full-image OCR

### 7) Private NextOCR credentials rejected

Verify:

- NEXTOCR_USERNAME and NEXTOCR_SECRET_KEY are set correctly
- Endpoint is reachable from your network
- Check firewall/proxy settings if on corporate network

---

## Known Limitations

- **Multi-page documents**: Only first page is processed
- **Handwritten text**: OCR accuracy is lower for handwriting
- **Non-Latin/Khmer scripts**: Limited support; test first
- **Rotated images**: Auto-correction works for 0-90 degree angles
- **Very small text**: Requires super-resolution (slower)
- **Low-quality scans**: May need multiple OCR attempts
- **No offline mode**: Requires internet access to NextOCR/OCR.space

---

## Getting Help

**API Documentation:**

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

**Common Issues:**

1. Check `/api/health` endpoint first
2. Review logs for detailed error messages
3. Try `/api/ocr-unified` with `multi_pass=true`
4. Test with sample images from the project

**File an issue:**

- Include steps to reproduce
- Attach a sample image (redacted if needed)
- Include logs and environment info

---

## Notes for Production

- Restrict CORS origins (do not use wildcard in production).
- Add request size/type validation and rate limiting.
- Add authentication on OCR endpoints.
- Use persistent object storage for uploaded/processed images.
- Replace in-memory cache with Redis for multi-instance deployments.
- Add structured logging and metrics (latency per provider/stage).
- Monitor cache hit rates and OCR provider performance.
- Set up alerts for API latency spikes or provider failures.

**Recommended deployment platforms:**

- Railway, Render, Heroku (simple setup)
- AWS ECS, Google Cloud Run (serverless, auto-scaling)
- Kubernetes (high availability, complex setup)
- Self-hosted VPS with Docker + Nginx + PM2

---

## License

MIT License (add your specific license here)

---

**Built with:**

- FastAPI for REST API
- NextOCR for Khmer/multilingual support
- YOLOv10 & U2-Net for document detection
- OpenCV for image processing
