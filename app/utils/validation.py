import logging
from copy import deepcopy
from typing import Dict, Any
from pydantic import ValidationError
from app.models.schemas import InvoiceData, PaymentInfo, LineItem

logger = logging.getLogger(__name__)


def validate_and_sanitize_invoice_data(
    raw_llm_output: Dict[str, Any],
    fallback_data: Dict[str, Any],
) -> InvoiceData:
    """
    Validate LLM output against InvoiceData schema with graceful fallback.
    
    Strategy:
    1. Try strict validation of full LLM output
    2. If fails, merge field-by-field: accept valid LLM fields, keep fallback for invalid
    3. Final validation; if still fails, return minimal valid object
    
    Args:
        raw_llm_output: Raw dict from LLM enhancement
        fallback_data: Original OCR-extracted data as fallback
    
    Returns:
        Validated InvoiceData instance
    """
    # Strategy 1: Try full validation first (fast path for good LLM output)
    try:
        validated = InvoiceData.model_validate(raw_llm_output, strict=False)
        logger.debug("LLM output validated successfully on first try")
        return validated
    except ValidationError as e:
        logger.warning(f"LLM output full validation failed: {len(e.errors())} errors")
        # Log first few errors for debugging
        for err in e.errors()[:3]:
            logger.debug(f"  - {err['loc']}: {err['msg']}")

    # Strategy 2: Field-by-field merge with validation
    logger.info("Falling back to field-by-field merge validation")
    merged = deepcopy(fallback_data)
    
    # Get all valid field names from schema
    valid_fields = set(InvoiceData.model_fields.keys())
    
    for field_name in valid_fields:
        if field_name not in raw_llm_output:
            continue  # Keep fallback value
            
        llm_value = raw_llm_output[field_name]
        
        try:
            # Try to validate just this field in isolation
            test_payload = {field_name: llm_value}
            # For nested models, we need to handle specially
            if field_name == 'payment' and isinstance(llm_value, dict):
                test_obj = InvoiceData(payment=PaymentInfo.model_validate(llm_value, strict=False))
            elif field_name == 'items' and isinstance(llm_value, list):
                validated_items = []
                for item in llm_value:
                    if isinstance(item, dict):
                        validated_items.append(LineItem.model_validate(item, strict=False))
                test_obj = InvoiceData(items=validated_items)
            else:
                test_obj = InvoiceData(**test_payload)
            
            # If validation passed, use LLM value
            merged[field_name] = getattr(test_obj, field_name)
            logger.debug(f"✓ Field '{field_name}' accepted from LLM")
            
        except Exception as field_err:
            logger.debug(f"✗ Field '{field_name}' rejected from LLM: {field_err}")
            # Keep fallback value (already in merged)
            pass
    
    # Strategy 3: Final validation with merged data
    try:
        final = InvoiceData.model_validate(merged, strict=False)
        logger.info("Field-merge validation succeeded")
        return final
    except ValidationError as final_err:
        logger.error(f"Final validation failed: {len(final_err.errors())} errors")
        
        # Last resort: return minimal valid object with raw_text preserved
        logger.warning("Returning minimal valid InvoiceData as last resort")
        return InvoiceData(
            merchant_name=fallback_data.get('merchant_name'),
            merchant_address=fallback_data.get('merchant_address'),
            merchant_phone=fallback_data.get('merchant_phone'),
            invoice_number=fallback_data.get('invoice_number'),
            invoice_date=fallback_data.get('invoice_date'),
            invoice_time=fallback_data.get('invoice_time'),
            items=[],  # Clear invalid items
            payment=PaymentInfo(
                subtotal=fallback_data.get('payment', {}).get('subtotal') if isinstance(fallback_data.get('payment'), dict) else None,
                total=fallback_data.get('payment', {}).get('total') if isinstance(fallback_data.get('payment'), dict) else None,
            ) if fallback_data.get('payment') else None,
            raw_text=fallback_data.get('raw_text', ''),
            dynamic_fields={}  # Clear to avoid pollution
        )


def summarize_corrections(original: Dict[str, Any], corrected: InvoiceData) -> Dict[str, Any]:
    """
    Generate a summary of what changed between original OCR data and corrected output.
    Useful for debugging and auditing.
    """
    changes = {}
    orig_flat = {k: v for k, v in original.items() if k in InvoiceData.model_fields}
    
    for field_name in InvoiceData.model_fields:
        orig_val = orig_flat.get(field_name)
        corr_val = getattr(corrected, field_name, None)
        
        # Skip raw_text and dynamic_fields for diff (they're special)
        if field_name in ('raw_text', 'dynamic_fields'):
            continue
            
        if orig_val != corr_val:
            changes[field_name] = {
                'before': orig_val,
                'after': corr_val
            }
    
    return {
        'fields_changed': len(changes),
        'changes': changes,
        'items_count': len(corrected.items),
        'has_payment': corrected.payment is not None
    }