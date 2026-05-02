# llm_helper.py
import json
import logging
import re
import hashlib
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)


class LLMHelper:
    """
    Post-process OCR invoice extraction with a Qwen-compatible chat endpoint.
    
    Features:
    - JSON mode enforcement for reliable parsing
    - Few-shot examples for typo correction
    - Decimal reconstruction heuristics for OCR artifacts
    - LRU caching to reduce redundant API calls
    - Debug mode for auditing corrections
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "qwen-plus",
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        timeout_seconds: int = 20,
        enable_debug: bool = False,
        cache_size: int = 128,
    ):
        self.api_key = api_key.strip()
        self.model = model.strip() or "qwen-plus"
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.enable_debug = enable_debug
        self.cache_size = cache_size
        
        # Initialize LRU cache for enhancement results
        self._enhance_cached = lru_cache(maxsize=cache_size)(self._enhance_impl)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _generate_cache_key(self, raw_text: str, invoice_data: Dict[str, Any]) -> str:
        """Generate deterministic cache key from input content."""
        # Use hash of normalized content to avoid cache bloat
        content = f"{raw_text}|{json.dumps(invoice_data, sort_keys=True, default=str)}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def enhance_invoice_data(
        self,
        raw_text: str,
        invoice_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Correct OCR typos and normalize invoice fields via Qwen.
        
        Returns:
            {
                "applied": bool,
                "invoice_data": dict (enhanced or fallback),
                "error": str|None,
                "debug": dict|None (if enable_debug=True)
            }
        """
        fallback = deepcopy(invoice_data)

        if not self.enabled:
            return {
                "applied": False,
                "invoice_data": fallback,
                "error": "Qwen API key is not configured",
                "debug": None,
            }

        try:
            cache_key = self._generate_cache_key(raw_text, invoice_data)

            # Use cached version if available
            result = self._enhance_cached(cache_key, raw_text, json.dumps(invoice_data, sort_keys=True, default=str))
            
            # Add cache hit info to debug
            if self.enable_debug:
                cache_info = self._enhance_cached.cache_info()
                result["debug"] = result.get("debug", {}) or {}
                result["debug"]["cache_hits"] = cache_info.hits
                result["debug"]["cache_size"] = cache_info.currsize
                
            return result
            
        except Exception as exc:
            logger.warning("Qwen invoice enhancement failed: %s", exc, exc_info=self.enable_debug)
            return {
                "applied": False,
                "invoice_data": fallback,
                "error": str(exc),
                "debug": None,
            }

    def _enhance_impl(self, cache_key: str, raw_text: str, invoice_data_json: str) -> Dict[str, Any]:
        """Internal implementation of enhancement (cached)."""
        invoice_data = json.loads(invoice_data_json)
        fallback = deepcopy(invoice_data)  # Re-copy for cache safety
        
        try:
            payload = {
                "model": self.model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},  
                "messages": [
                    {
                        "role": "system",
                        "content": self._get_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": self._build_prompt(raw_text=raw_text, invoice_data=invoice_data),
                    },
                ],
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # Change this proxy via nginx in production to avoid CORS issues and hide API key
                "HTTP-Referer": "http://localhost:8000",  # Update in production
                "X-Title": "E-Invoice Engine",
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()

            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            if self.enable_debug:
                logger.debug("LLM raw response (first 800 chars): %s", content[:800])

            parsed = self._parse_json_content(content)
            if self.enable_debug:
                logger.info("LLM enhancement applied. Parsed keys: %s", list(parsed.keys()))

            normalized = self._normalize_invoice_payload(
                parsed, fallback=fallback, raw_text=raw_text, original_data=invoice_data
            )

            result = {
                "applied": True,
                "invoice_data": normalized,
                "error": None,
            }

            if self.enable_debug:
                result["debug"] = {
                    "corrections_summary": self._summarize_corrections(fallback, normalized),
                    "llm_snippet": content[:300],
                    "cache_key": cache_key,
                }

            return result

        except requests.RequestException as http_err:
            logger.error(f"HTTP error calling Qwen API: {http_err}")
            raise
        except Exception as exc:
            logger.error(f"Unexpected error in _enhance_impl: {exc}", exc_info=self.enable_debug)
            raise

    def _get_system_prompt(self) -> str:
        return (
            "You are an invoice data correction engine specialized in Khmer/English bilingual receipts.\n"
            "CRITICAL RULES:\n"
            "1. Fix OCR typos in BOTH field labels AND values:\n"
            "   - Labels: 'Cashlef'→'cashier_name', 'Involce'→'invoice_number', 'Raturn'→'return'\n"
            "   - Values: 'Iced Late 1 2 43 0% 2'→'Iced Late (+ Normal Ice, + No Sugar)'\n"
            "2. Amounts: Strip '$', '៛', commas; reconstruct decimals if context suggests (e.g., '43'→2.43)\n"
            "3. Phone: '012589 469'→'012589469'\n"
            "4. Preserve Khmer text exactly as-is.\n"
            "5. If a field cannot be confidently determined, set it to null — do NOT guess.\n"
            "6. Return ONLY valid JSON. NO markdown, NO code blocks, NO explanations.\n"
            "7. All monetary values must be numbers (float), NOT strings.\n"
            "8. Item 'total' must equal quantity × price. Validate this.\n"
            "9. Payment/discount/tax lines do NOT belong in 'items' array — put them in 'payment' object.\n"
        )

    def _build_prompt(self, raw_text: str, invoice_data: Dict[str, Any]) -> str:
        schema = {
            "merchant_name": "string|null",
            "merchant_address": "string|null",
            "merchant_phone": "string|null",
            "invoice_number": "string|null",
            "invoice_date": "string|null (YYYY-MM-DD if possible)",
            "invoice_time": "string|null (HH:MM if possible)",
            "items": [
                {
                    "name": "string - include modifiers like 'Normal Ice' if they're add-ons",
                    "quantity": "int (>=1)",
                    "price": "number - unit price ONLY, must match the invoice",
                    "total": "number - must equal quantity × price",
                }
            ],
            "payment": {
                "subtotal": "number|null - subtotal BEFORE tax/discounts",
                "tax": "number|null",
                "total": "number|null - FINAL total amount",
                "discount_usd": "number|null",
                "method": "string|null - payment method: cash, ABA, card, etc",
            },
            "cashier_name": "string|null - from 'Cashlef' or similar field",
            "exchange_rate": "string|null - e.g., '4100KHR=$1'",
            "total_khr": "number|null - total in Cambodian Riel",
            "dynamic_fields": "object - ONLY for truly custom fields not listed above",
            "raw_text": "string - the EXACT original OCR text, unchanged",
        }

        examples = '''
## Examples of Corrections (learn from these):

### Example 1: Item name cleanup + price reconstruction
Input item: {"name": "Iced Late 1 2 43 0% 2", "price": 43, "quantity": 1}
Context: raw_text shows "SUB TOTAL : $ 2.43"
→ Output: {"name": "Iced Late (+ Normal Ice, + No Sugar)", "quantity": 1, "price": 2.43, "total": 2.43}

### Example 2: Field label typo + promotion to first-class field
Input: raw_text contains "Cashlef : sreyne"
→ Output: Add "cashier_name": "sreyne" at top-level (NOT in dynamic_fields)

### Example 3: Amount with currency symbol + decimal reconstruction
Input: "$ 2,43" or raw shows "43" but context total is ~2.43
→ Output: 2.43 (as number, not string)

### Example 4: Wrong invoice number from OCR
Input: invoice_number: "OICE" but raw_text shows "C26-11756"
→ Output: "C26-11756"

### Example 5: Phone normalization
Input: "012589 469"
→ Output: "012589469"

### Example 6: Remove payment lines from items
Input items array contains: {"name": "Disc. (0%) ($) :", "price": 0}
→ Output: REMOVE from items; put discount in payment.discount_usd

### Example 7: Exchange rate extraction
Input: raw_text contains "Exchange rate, 4100KHR=$1"
→ Output: Add "exchange_rate": "4100KHR=$1" at top-level
'''

        field_placement = '''
## FIELD PLACEMENT RULES (CRITICAL):
- merchant_name, merchant_address, merchant_phone → top-level fields
- invoice_number, invoice_date, invoice_time → top-level fields  
- items array → ONLY actual products/services. REMOVE: "Disc.", "Return", "Receive", "Subtotal", "Total", "Tax", "Cashlef"
- payment object → subtotal, tax, total, discount_usd, method
- cashier_name → top-level field (extracted from "Cashlef" or similar)
- exchange_rate → top-level field (e.g., "4100KHR=$1")
- total_khr → top-level field (number, not string)
- dynamic_fields → ONLY for truly custom fields NOT listed above (e.g., customer_name, table_number, notes)
- raw_text → EXACT original OCR text, unchanged
'''

        return (
            "You are an expert invoice data extraction and correction system.\n"
            "Your task is to extract structured data from the OCR raw text of a receipt/invoice.\n\n"
            "## CRITICAL RULES:\n"
            "1) ALL monetary values (prices, totals) must be extracted as NUMBERS ONLY (e.g., 2.43, NOT 43).\n"
            "   Strip ALL currency symbols ($, ៛, etc.) and extraneous text from amounts.\n"
            "2) Item prices are UNIT prices. The 'total' for each item = quantity × unit price.\n"
            "3) The payment.total should match the grand total shown on the invoice.\n"
            "4) Extract merchant_phone from any phone number patterns (e.g., '012589 469' → '012589469').\n"
            "5) Fix OCR typos in field labels AND values (e.g., 'Cashlef' → 'cashier_name', 'Involce' → 'invoice_number',\n"
            "   'Raturn' → 'return').\n"
            "6) Preserve Khmer text exactly as-is.\n"
            "7) If a field cannot be confidently determined, set it to null rather than guessing.\n"
            "8) dynamic_fields should contain ONLY additional useful key-value pairs that don't fit the main schema.\n"
            "9) raw_text must be the EXACT original OCR text with no modifications.\n"
            "10) REMOVE payment/discount/tax lines from 'items' array — they belong in 'payment' object.\n"
            "11) Return ONLY a valid JSON object. NO markdown, NO code blocks, NO explanations.\n\n"
            f"{examples}\n\n"
            f"{field_placement}\n\n"
            f"## Target Schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            f"## OCR Raw Text:\n---\n{raw_text}\n---\n\n"
            f"## Currently Extracted (may contain errors):\n{json.dumps(invoice_data, ensure_ascii=False, indent=2)}\n\n"
            "Return the corrected and enhanced JSON now."
        )

    def _parse_json_content(self, content: str) -> Dict[str, Any]:
        if not content:
            raise ValueError("Empty Qwen response")

        content = content.strip()

        # Remove markdown code fences
        fenced_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", content)
        if fenced_match:
            content = fenced_match.group(1).strip()

        # Try direct parse first
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: extract first balanced JSON object
        candidate = self._extract_first_json_object(content)
        if not candidate:
            raise ValueError("Could not parse JSON object from Qwen response")

        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("Qwen response JSON is not an object")
        return parsed

    def _extract_first_json_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False

        for idx in range(start, len(text)):
            char = text[idx]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]

        return None

    def _normalize_invoice_payload(
        self,
        model_output: Dict[str, Any],
        fallback: Dict[str, Any],
        raw_text: str,
        original_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Normalize LLM output with type coercion and validation."""
        data = deepcopy(fallback)

        # Top-level string fields
        for key in [
            "merchant_name",
            "merchant_address",
            "merchant_phone",
            "invoice_number",
            "invoice_date",
            "invoice_time",
            "cashier_name",
            "exchange_rate",
            "payment_method",
        ]:
            if key in model_output:
                value = model_output.get(key)
                data[key] = None if value is None else str(value).strip()

        # Items normalization
        model_items = model_output.get("items")
        if isinstance(model_items, list):
            normalized_items = []
            # Get context total for decimal reconstruction
            payment_ctx = original_data.get("payment", {}) or {}
            ctx_total = self._to_float(payment_ctx.get("total"), default=None)

            for item in model_items[:100]:  # Limit to prevent abuse
                if not isinstance(item, dict):
                    continue

                name = str(item.get("name", "")).strip()
                if not name:
                    continue

                # Skip payment/discount lines that leaked into items
                skip_keywords = [
                    "disc", "return", "receive", "received", "subtotal", 
                    "total", "tax", "cashlef", "involce", "exchange"
                ]
                if any(kw in name.lower() for kw in skip_keywords):
                    continue

                quantity = self._to_int(item.get("quantity"), default=1)
                price = self._to_float(item.get("price"), default=0.0, context_total=ctx_total)
                calculated_total = round(quantity * price, 2)
                reported_total = self._to_float(item.get("total"), default=calculated_total, context_total=ctx_total)

                # Use calculated total if reported is inconsistent
                final_total = reported_total if abs(reported_total - calculated_total) < 0.01 else calculated_total

                normalized_items.append(
                    {
                        "name": name,
                        "quantity": max(quantity, 1),
                        "price": price,
                        "total": final_total,
                    }
                )

            if normalized_items:  # Only override if we have valid items
                data["items"] = normalized_items

        # Payment normalization
        payment_in = model_output.get("payment")
        if isinstance(payment_in, dict):
            data["payment"] = {
                "subtotal": self._to_nullable_float(payment_in.get("subtotal")),
                "tax": self._to_nullable_float(payment_in.get("tax")),
                "total": self._to_nullable_float(payment_in.get("total")),
                "discount_usd": self._to_nullable_float(
                    payment_in.get("discount_usd") or payment_in.get("discount")
                ),
                "method": self._to_nullable_str(payment_in.get("method")),
            }

        # Promoted dynamic fields
        for key in ["cashier_name", "exchange_rate", "total_khr", "discount_usd", "payment_method"]:
            if key in model_output and data.get(key) is None:
                value = model_output[key]
                if key in ("total_khr", "discount_usd"):
                    # Convert to float, handling strings like '+ 10,000' or '$0'
                    data[key] = self._to_nullable_float(value)
                else:
                    data[key] = self._to_nullable_str(value)

        # Dynamic fields (remaining custom fields)
        dynamic_fields = model_output.get("dynamic_fields")
        if isinstance(dynamic_fields, dict):
            # Merge with existing, preferring LLM output
            data["dynamic_fields"] = {**data.get("dynamic_fields", {}), **dynamic_fields}

        # Always preserve raw_text exactly
        data["raw_text"] = raw_text

        return data

    def _to_int(self, value: Any, default: int) -> int:
        try:
            if isinstance(value, str):
                cleaned = value.strip().replace(",", "")
                return int(float(cleaned))
            return int(value)
        except (TypeError, ValueError):
            return default

    def _to_float(
        self,
        value: Any,
        default: float,
        context_total: Optional[float] = None,
    ) -> float:
        """Convert value to float with OCR-aware decimal reconstruction."""
        try:
            if isinstance(value, str):
                # Remove currency symbols, commas, whitespace
                cleaned = re.sub(r'[^\d.\-]', '', value.strip())
                if not cleaned or cleaned == "-":
                    return default

                result = float(cleaned)

                if (
                    context_total
                    and 0.1 < context_total < 100  # reasonable invoice total range
                    and result > 10
                    and result < 1000
                    and result != context_total
                ):
                    # Try common OCR decimal errors
                    candidates = [
                        result / 100,  # "243" → 2.43
                        result / 10,   # "243" → 24.3 (less likely)
                        result,        # no change
                    ]
                    # Prefer candidate closest to context
                    best = min(candidates, key=lambda x: abs(x - context_total))
                    if abs(best - context_total) < 1.0 and best != result:
                        logger.debug(f"Reconstructed decimal: {value} → {best} (context: {context_total})")
                        return round(best, 2)

                # Sanity check for unreasonable values
                if abs(result) > 100000:
                    logger.warning("Suspicious price value: %s, using default", value)
                    return default

                return round(result, 2)

            return round(float(value), 2)
        except (TypeError, ValueError):
            return default

    def _to_nullable_float(self, value: Any) -> Optional[float]:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return None
        return self._to_float(value, default=0.0)

    def _to_nullable_str(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _summarize_corrections(self, original: Dict, corrected: Dict) -> Dict[str, Any]:
        """Simple diff for debugging — shows what changed."""
        changes = {}
        for key in set(original.keys()) | set(corrected.keys()):
            # Skip raw_text and dynamic_fields for summary
            if key in ('raw_text', 'dynamic_fields'):
                continue
            orig_val = original.get(key)
            corr_val = corrected.get(key)
            if json.dumps(orig_val, sort_keys=True, default=str) != json.dumps(corr_val, sort_keys=True, default=str):
                changes[key] = {"before": orig_val, "after": corr_val}
        return {
            "fields_changed": len(changes),
            "changes": changes
        }

    def clear_cache(self):
        """Clear the LRU cache (useful for testing or config changes)."""
        self._enhance_cached.cache_clear()

    def get_cache_info(self):
        """Return cache statistics."""
        return self._enhance_cached.cache_info()