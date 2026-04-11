import base64
import io
import requests
from PIL import Image
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2
import logging
import time
import hashlib
import json
from collections import OrderedDict


# Final Configurations for OCR Client
class OCRClient:

    def __init__(
        self,
        ocr_space_key: str = "",
        nextocr_endpoint: str = "",
        nextocr_username: str = "",
        nextocr_secret_key: str = "",
        cache_enabled: bool = True,
        max_retries: int = 2,
        retry_delay: float = 0.5
    ):
        self.ocr_space_key = ocr_space_key
        self.nextocr_endpoint = nextocr_endpoint
        self.nextocr_username = nextocr_username
        self.nextocr_secret_key = nextocr_secret_key

        self.logger = logging.getLogger(__name__)
        
        # Caching
        self.cache_enabled = cache_enabled
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._cache_max_size = 100
        
        # Retry config
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def extract(
        self,
        image: np.ndarray,
        method: str = "auto",
        lang: str = "en",
        use_cache: bool = True
    ) -> Dict[str, Any]:
        start_time = time.time()

        # Check cache
        cache_key = self._generate_cache_key(image, method, lang) if use_cache else None
        if cache_key and cache_key in self._cache:
            cached_result = self._cache[cache_key]
            self.logger.debug(f"Cache hit for key: {cache_key[:16]}...")
            # Move to end (LRU)
            self._cache.move_to_end(cache_key)
            cached_result["cached"] = True
            return cached_result

        if method == "auto":
            result = self._smart_route(image, lang, start_time)
        elif method == "nextocr":
            result = self._extract_with_nextocr(image, lang, start_time)
        elif method == "ocrspace":
            result = self._extract_with_ocrspace_safe(image, lang, start_time)
        else:
            result = self._error("Invalid OCR method")

        # Cache result
        if cache_key and result.get("text") and not result.get("error"):
            self._cache_result(cache_key, result)

        result["cached"] = False
        return result

    def extract_structured(
        self,
        image: np.ndarray,
        method: str = "auto",
        lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Extract text with structured output (words, lines, blocks).
        Currently works best with NextOCR.
        """
        start_time = time.time()
        
        if method == "auto":
            if self._is_nextocr_configured():
                return self._extract_structured_nextocr(image, lang, start_time)
            else:
                basic = self._smart_route(image, lang, start_time)
                return self._convert_to_structured(basic)
        elif method == "nextocr":
            return self._extract_structured_nextocr(image, lang, start_time)
        else:
            basic = self._extract_with_ocrspace_safe(image, lang, start_time)
            return self._convert_to_structured(basic)

    def _generate_cache_key(self, image: np.ndarray, method: str, lang: str) -> str:
        """Generate cache key from image hash + params"""
        # Quick hash of image data
        img_hash = hashlib.md5(image.tobytes()).hexdigest()[:16]
        return f"{method}:{lang}:{img_hash}"

    def _cache_result(self, key: str, result: Dict[str, Any]) -> None:
        """Store result in cache with LRU eviction"""
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest
            self._cache.popitem(last=False)
        
        # Store without 'cached' flag
        cache_entry = {k: v for k, v in result.items() if k != 'cached'}
        self._cache[key] = cache_entry

    def _clear_cache(self) -> None:
        """Clear the cache"""
        self._cache.clear()

    def _smart_route(self, image, lang, start_time):
        """
        Smart routing with confidence-based selection.
        """
        results = []

        if not self._is_nextocr_configured() and not self.ocr_space_key:
            return self._error(
                "No OCR provider configured. Set OCR_SPACE_API_KEY or NEXTOCR_ENDPOINT/NEXTOCR_USERNAME/NEXTOCR_SECRET_KEY."
            )

        # Rule 1: Khmer → prefer NextOCR
        if lang == "km" and self._is_nextocr_configured():
            result = self._extract_with_nextocr(image, lang, start_time)
            if result.get("text") and result.get("confidence", 0) > 0.3:
                return result
            results.append(result)

        # Rule 2: Try NextOCR first if available
        if self._is_nextocr_configured():
            result = self._extract_with_nextocr(image, lang, start_time)
            
            if result.get("text") and result.get("confidence", 0) > 0.5:
                return result
            results.append(result)

        # Rule 3: Fallback OCR.space
        if self.ocr_space_key:
            result = self._extract_with_ocrspace_safe(image, lang, start_time)
            if result.get("text"):
                return result
            results.append(result)

        # Return best failed attempt if all failed
        for r in results:
            if r.get("text"):
                return r

        error_messages = []
        for r in results:
            provider = r.get("provider") or "unknown"
            err = r.get("error")
            if err:
                error_messages.append(f"{provider}: {err}")
            elif not r.get("text"):
                error_messages.append(f"{provider}: empty OCR response")

        if error_messages:
            return self._error("All OCR providers failed. " + " | ".join(error_messages))

        return self._error("All OCR providers failed")

    def _compute_confidence(self, text: str, structured: Optional[Dict] = None) -> float:
        """Compute confidence score from text quality and structure"""
        if not text:
            return 0.0

        # Base score from text length
        length_score = min(len(text) / 500, 1.0)

        # Character quality metrics
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        alnum_ratio = sum(c.isalnum() for c in text) / max(len(text), 1)
        
        # Penalize excessive special characters
        special_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / max(len(text), 1)
        
        structure_score = min((digit_ratio + alpha_ratio), 1.0)
        penalty = special_ratio * 0.3  # Penalize if > 30% special chars

        # Boost if structured data available
        if structured:
            if structured.get("lines") and len(structured["lines"]) > 0:
                length_score += 0.1
            if structured.get("confidence", 0) > 0.7:
                length_score += 0.1

        confidence = (0.6 * length_score + 0.4 * structure_score) - penalty
        return round(max(0.0, min(1.0, confidence)), 3)

    def _convert_to_structured(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert basic OCR result to structured format"""
        text = result.get("text", "")
        
        # Split into lines and words
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        words = text.split()
        
        # Create simple block structure
        blocks = []
        current_block = []
        
        for line in lines:
            if len(line) > 50:  # Long line = new block
                if current_block:
                    blocks.append({"text": "\n".join(current_block), "type": "paragraph"})
                    current_block = []
            current_block.append(line)
        
        if current_block:
            blocks.append({"text": "\n".join(current_block), "type": "paragraph"})

        result["structured"] = {
            "words": words,
            "lines": lines,
            "blocks": blocks,
            "word_count": len(words),
            "line_count": len(lines)
        }
        
        return result

    def _extract_with_nextocr(
        self,
        image: np.ndarray,
        lang: str,
        start_time: float,
        return_structured: bool = False
    ) -> Dict[str, Any]:
        """Extract text using NextOCR with retry logic"""
        
        if not self._is_nextocr_configured():
            return self._error("NextOCR not configured")

        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                image_bytes = self._to_jpeg_bytes(image)

                headers = {
                    "X-Username": self.nextocr_username,
                    "X-Secret-Key": self.nextocr_secret_key,
                    "User-Agent": "ocr-pipeline/2.0",
                    "Accept-Language": "km" if lang == "km" else "en"
                }

                files = {
                    "file": ("image.jpg", image_bytes, "image/jpeg")
                }

                response = requests.post(
                    self.nextocr_endpoint,
                    files=files,
                    headers=headers,
                    timeout=30
                )

                response.raise_for_status()

                raw = response.json()
                
                if return_structured:
                    return self._process_nextocr_structured(raw, start_time)
                
                text = self._extract_text(raw)
                text = self._clean_text(text)

                confidence = self._compute_confidence(text)

                return self._format_response(
                    text=text,
                    provider="nextocr",
                    confidence=confidence,
                    raw=raw,
                    latency=start_time
                )

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                self.logger.warning(f"NextOCR attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                    
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"NextOCR unexpected error: {e}")
                break

        return self._error(f"NextOCR failed after {self.max_retries + 1} attempts: {last_error}", provider="nextocr")

    def _extract_structured_nextocr(
        self,
        image: np.ndarray,
        lang: str,
        start_time: float
    ) -> Dict[str, Any]:
        """Extract structured text (words, lines, blocks) from NextOCR"""
        return self._extract_with_nextocr(image, lang, start_time, return_structured=True)

    def _process_nextocr_structured(self, raw: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Process NextOCR response into structured format"""
        try:
            # Try to extract structured data from NextOCR response
            structured = {}
            
            # Common NextOCR response patterns
            if isinstance(raw, dict):
                # Look for words/lines/blocks in response
                for key in ["words", "lines", "blocks", "regions", "results"]:
                    if key in raw:
                        structured[key] = raw[key]
                
                # Extract text from structured data
                text_parts = []
                if "lines" in structured and isinstance(structured["lines"], list):
                    for line in structured["lines"]:
                        if isinstance(line, dict) and "text" in line:
                            text_parts.append(line["text"])
                        elif isinstance(line, str):
                            text_parts.append(line)
                
                if not text_parts and "words" in structured and isinstance(structured["words"], list):
                    for word in structured["words"]:
                        if isinstance(word, dict) and "text" in word:
                            text_parts.append(word["text"])
                        elif isinstance(word, str):
                            text_parts.append(word)
                
                text = self._clean_text(" ".join(text_parts) if text_parts else self._extract_text(raw))
                
                # Get confidence from response if available
                confidence = raw.get("confidence", 0.0)
                if isinstance(confidence, str):
                    try:
                        confidence = float(confidence)
                    except ValueError:
                        confidence = 0.0
                
                # Normalize confidence to 0-1 range
                if confidence > 1.0:
                    confidence = confidence / 100.0

            else:
                text = self._clean_text(self._extract_text(raw))
                confidence = self._compute_confidence(text)
                structured = None

            result = self._format_response(
                text=text,
                provider="nextocr",
                confidence=confidence,
                raw=raw,
                latency=start_time
            )
            
            if structured:
                result["structured"] = structured

            return result

        except Exception as e:
            self.logger.error(f"Failed to process NextOCR structured response: {e}")
            # Fallback to basic extraction
            text = self._clean_text(self._extract_text(raw))
            return self._format_response(
                text=text,
                provider="nextocr",
                confidence=self._compute_confidence(text),
                raw=raw,
                latency=start_time
            )


    def _extract_with_ocrspace_safe(self, image, lang, start_time):

        try:
            base64_img = self.image_to_base64(image)

            result = self._extract_with_ocrspace(base64_img, lang)

            if not result:
                return self._error("OCR.space failed", provider="ocrspace")

            text = self._clean_text(result.get("text", ""))
            confidence = self._compute_confidence(text)

            return self._format_response(
                text=text,
                provider="ocrspace",
                confidence=confidence,
                raw=result.get("raw_response"),
                latency=start_time
            )

        except Exception as e:
            return self._error(str(e), provider="ocrspace")

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        # Remove excessive spaces
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

        # Normalize spacing
        text = text.replace("  ", " ")

        return text.strip()

    def _format_response(self, text, provider, confidence, raw, latency):

        return {
            "text": text,
            "provider": provider,
            "confidence": confidence,
            "latency": round(time.time() - latency, 3),
            "raw_response": raw,
            "error": None
        }

    def _error(self, message, provider=None):
        return {
            "text": "",
            "provider": provider,
            "confidence": 0.0,
            "latency": 0,
            "raw_response": None,
            "error": message
        }

    def _is_nextocr_configured(self):
        return bool(
            self.nextocr_endpoint and
            self.nextocr_username and
            self.nextocr_secret_key
        )

    def image_to_base64(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(image)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=95)

        return base64.b64encode(buffer.getvalue()).decode()

    def _to_jpeg_bytes(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(image)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=95)

        return buffer.getvalue()

    def _extract_text(self, raw):
        if isinstance(raw, str):
            return raw

        if isinstance(raw, dict):
            for key in ["text", "result", "output", "content"]:
                if key in raw and isinstance(raw[key], str):
                    return raw[key]

            for v in raw.values():
                t = self._extract_text(v)
                if t:
                    return t

        if isinstance(raw, list):
            return "\n".join([self._extract_text(x) for x in raw if x])

        return ""

    def _extract_with_ocrspace(self, image_base64, lang):

        url = "https://api.ocr.space/parse/image"

        payload = {
            "apikey": self.ocr_space_key,
            "base64Image": f"data:image/jpeg;base64,{image_base64}",
            "language": "eng" if lang == "en" else "eng",
            "OCREngine": "2"
        }

        response = requests.post(url, data=payload, timeout=30)
        result = response.json()

        if result.get("IsErroredOnProcessing"):
            return None

        text = ""
        for parsed in result.get("ParsedResults", []):
            text += parsed.get("ParsedText", "") + "\n"

        return {
            "text": text.strip(),
            "raw_response": result
        }