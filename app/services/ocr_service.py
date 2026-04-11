import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import logging
import time

from .document_segmenter import DocumentSegmenter
from .image_preprocessor import ImagePreprocessor
from .ocr_client import OCRClient
from .data_extractor import DataExtractor

logger = logging.getLogger(__name__)


class OCRResult:

    def __init__(
        self,
        text: str,
        confidence: float,
        provider: str,
        preprocessing_label: str,
        image: np.ndarray,
        latency: float,
        structured: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        self.text = text
        self.confidence = confidence
        self.provider = provider
        self.preprocessing_label = preprocessing_label
        self.image = image
        self.latency = latency
        self.structured = structured
        self.error = error
    
    @property
    def score(self) -> float:
        if self.error or not self.text:
            return 0.0
        
        text_quality = min(len([c for c in self.text if c.isalnum()]) / 100, 1.0)
        return (0.6 * self.confidence) + (0.4 * text_quality)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "provider": self.provider,
            "preprocessing": self.preprocessing_label,
            "latency": self.latency,
            "structured": self.structured,
            "error": self.error
        }


class UnifiedOCRService:

    def __init__(
        self,
        ocr_space_key: str = "",
        nextocr_endpoint: str = "",
        nextocr_username: str = "",
        nextocr_secret_key: str = "",
        use_segmenter: bool = True,
        cache_enabled: bool = True
    ):
        # Initialize components
        self.segmenter: Optional[DocumentSegmenter] = None
        if use_segmenter:
            try:
                self.segmenter = DocumentSegmenter()
                logger.info("DocumentSegmenter initialized")
            except Exception as e:
                logger.warning(f"DocumentSegmenter unavailable: {e}")
        
        self.preprocessor = ImagePreprocessor(use_segmenter=use_segmenter)
        
        self.ocr_client = OCRClient(
            ocr_space_key=ocr_space_key,
            nextocr_endpoint=nextocr_endpoint,
            nextocr_username=nextocr_username,
            nextocr_secret_key=nextocr_secret_key,
            cache_enabled=cache_enabled
        )
        
        self.extractor = DataExtractor()
        
        logger.info("UnifiedOCRService initialized")

    def process(
        self,
        image: np.ndarray,
        lang: str = "en",
        auto_crop: bool = True,
        multi_pass: bool = True,
        return_structured: bool = False
    ) -> Dict[str, Any]:

        start_time = time.time()
        metadata = {
            "pipeline_stages": {},
            "attempts": [],
            "selection_reason": ""
        }

        # Stage 1: Document Detection & Cropping
        doc_result = self._detect_and_crop(image, auto_crop)
        work_image = doc_result["image"]
        metadata["pipeline_stages"]["detection"] = {
            "success": doc_result["success"],
            "method": doc_result["method"],
            "original_size": (image.shape[1], image.shape[0]),
            "cropped_size": (work_image.shape[1], work_image.shape[0])
        }

        # Stage 2: Multi-pass Preprocessing + OCR
        if multi_pass:
            ocr_results = self._multi_pass_ocr(work_image, lang, return_structured)
        else:
            ocr_results = self._single_pass_ocr(work_image, lang, return_structured)
        
        metadata["attempts"] = [r.to_dict() for r in ocr_results]

        # Stage 3: Select best result
        best_result = self._select_best_result(ocr_results)

        metadata["pipeline_stages"]["ocr"] = {
            "total_attempts": len(ocr_results),
            "successful_attempts": sum(1 for r in ocr_results if r.text),
            "best_provider": best_result.provider if best_result else "none",
            "best_preprocessing": best_result.preprocessing_label if best_result else "none",
            "best_confidence": best_result.confidence if best_result else 0.0,
        }
        metadata["pipeline_stages"]["total_latency"] = round(time.time() - start_time, 3)
        
        if not best_result or not best_result.text:
            return self._error_response(
                "All OCR attempts failed",
                metadata,
                attempts=ocr_results,
                detection=metadata["pipeline_stages"].get("detection")
            )

        # Extract structured data
        extracted_data = self.extractor.extract(best_result.text)
        
        metadata["pipeline_stages"]["extraction"] = {
            "merchant_detected": bool(extracted_data.get("merchant_name")),
            "items_count": len(extracted_data.get("items", [])),
            "total_amount": extracted_data.get("payment", {}).get("total") if extracted_data.get("payment") else None
        }

        # Save best processed image
        processed_image_path = self._save_processed_image(best_result.image)

        metadata["pipeline_stages"]["total_latency"] = round(time.time() - start_time, 3)

        return {
            "success": True,
            "data": extracted_data,
            "ocr_result": best_result.to_dict(),
            "all_attempts": metadata["attempts"],
            "detection": metadata["pipeline_stages"]["detection"],
            "processed_image_path": processed_image_path,
            "metadata": metadata
        }

    def _detect_and_crop(
        self,
        image: np.ndarray,
        auto_crop: bool
    ) -> Dict[str, Any]:
        """Detect document and crop to region of interest"""
        
        if not auto_crop:
            return {"image": image, "success": False, "method": "none"}

        # Try segmenter first (U²-Net)
        if self.segmenter is not None:
            try:
                cropped, meta = self.segmenter.crop_document(image, return_metadata=True)
                if cropped is not None:
                    logger.info(f"Document cropped via U²-Net: {meta.get('crop_dimensions', 'unknown')}")
                    return {
                        "image": cropped,
                        "success": True,
                        "method": "unet"
                    }
            except Exception as e:
                logger.debug(f"U²-Net crop failed: {e}")

        # Fallback: preprocessor's contour detection
        try:
            cropped, success = self.preprocessor.detect_and_crop_document(
                image, use_segmenter=False
            )
            return {
                "image": cropped,
                "success": success,
                "method": "contour" if success else "none"
            }
        except Exception as e:
            logger.warning(f"Contour detection failed: {e}")

        return {"image": image, "success": False, "method": "fallback"}

    def _generate_preprocessing_strategies(
        self,
        image: np.ndarray,
        lang: str
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate preprocessing strategies based on image characteristics"""
        strategies = []
        h, w = image.shape[:2]

        # Light enhancement (fast, good for clear images)
        strategies.append((
            "light_en",
            {"for_nextocr": True, "lang": "en", "auto_crop": False}
        ))

        # Khmer-optimized (for Khmer text)
        if lang == "km":
            strategies.append((
                "khmer_enhanced",
                {"for_nextocr": True, "lang": "km", "auto_crop": False}
            ))

        # Heavy enhancement (for low quality)
        strategies.append((
            "heavy_enhanced",
            {"for_nextocr": True, "lang": lang, "auto_crop": False}
        ))

        # No enhancement (original)
        strategies.append((
            "original",
            {"for_nextocr": False, "lang": lang, "auto_crop": False}
        ))

        # Super-resolution for small images
        if w < 1000:
            strategies.append((
                "superres",
                {"for_nextocr": True, "lang": lang, "auto_crop": False}
            ))

        return strategies

    def _multi_pass_ocr(
        self,
        image: np.ndarray,
        lang: str,
        return_structured: bool
    ) -> List[OCRResult]:
        """Try multiple preprocessing strategies + OCR providers"""
        results = []
        strategies = self._generate_preprocessing_strategies(image, lang)
        
        for label, kwargs in strategies:
            try:
                # Preprocess
                preprocessed, meta = self.preprocessor.preprocess_adaptive(
                    image, **kwargs
                )
                
                # OCR
                ocr_start = time.time()
                if return_structured:
                    ocr_result = self.ocr_client.extract_structured(
                        preprocessed, method="auto", lang=lang
                    )
                else:
                    ocr_result = self.ocr_client.extract(
                        preprocessed, method="auto", lang=lang
                    )
                
                latency = round(time.time() - ocr_start, 3)
                
                result = OCRResult(
                    text=ocr_result.get("text", ""),
                    confidence=ocr_result.get("confidence", 0.0),
                    provider=ocr_result.get("provider", "unknown"),
                    preprocessing_label=label,
                    image=preprocessed,
                    latency=latency,
                    structured=ocr_result.get("structured"),
                    error=ocr_result.get("error")
                )
                
                results.append(result)
                logger.debug(f"OCR pass '{label}': {len(result.text)} chars, confidence={result.confidence}")
                
            except Exception as e:
                logger.warning(f"OCR pass '{label}' failed: {e}")
                results.append(OCRResult(
                    text="",
                    confidence=0.0,
                    provider="error",
                    preprocessing_label=label,
                    image=image,
                    latency=0,
                    error=str(e)
                ))
        
        return results

    def _single_pass_ocr(
        self,
        image: np.ndarray,
        lang: str,
        return_structured: bool
    ) -> List[OCRResult]:
        """Single OCR pass with default preprocessing"""
        try:
            preprocessed, meta = self.preprocessor.preprocess_adaptive(
                image, for_nextocr=True, lang=lang, auto_crop=False
            )
            
            ocr_start = time.time()
            if return_structured:
                ocr_result = self.ocr_client.extract_structured(
                    preprocessed, method="auto", lang=lang
                )
            else:
                ocr_result = self.ocr_client.extract(
                    preprocessed, method="auto", lang=lang
                )
            
            return [OCRResult(
                text=ocr_result.get("text", ""),
                confidence=ocr_result.get("confidence", 0.0),
                provider=ocr_result.get("provider", "unknown"),
                preprocessing_label="default",
                image=preprocessed,
                latency=round(time.time() - ocr_start, 3),
                structured=ocr_result.get("structured"),
                error=ocr_result.get("error")
            )]
            
        except Exception as e:
            logger.error(f"Single pass OCR failed: {e}")
            return [OCRResult(
                text="",
                confidence=0.0,
                provider="error",
                preprocessing_label="default",
                image=image,
                latency=0,
                error=str(e)
            )]

    def _select_best_result(self, results: List[OCRResult]) -> Optional[OCRResult]:
        """Select best OCR result based on scoring"""
        valid_results = [r for r in results if r.text and not r.error]
        
        if not valid_results:
            # Return any result with text even if has minor errors
            for r in results:
                if r.text:
                    return r
            return None
        
        # Sort by score (descending)
        valid_results.sort(key=lambda r: r.score, reverse=True)
        
        return valid_results[0]

    def _save_processed_image(self, image: np.ndarray) -> str:
        """Save processed image and return path"""
        # This should integrate with your existing file storage
        # For now, return placeholder
        return "processed_image.jpg"

    def _error_response(
        self,
        error: str,
        metadata: Dict,
        attempts: Optional[List[OCRResult]] = None,
        detection: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format error response"""
        return {
            "success": False,
            "data": None,
            "ocr_result": None,
            "all_attempts": [a.to_dict() for a in (attempts or [])],
            "detection": detection or metadata.get("pipeline_stages", {}).get("detection"),
            "processed_image_url": None,
            "error": error,
            "metadata": metadata
        }

    def clear_cache(self) -> None:
        """Clear OCR result cache"""
        self.ocr_client._clear_cache()
