import re
from typing import List, Optional, Dict, Any
from app.models.schemas import InvoiceData, LineItem, PaymentInfo


class DataExtractor:
    """Extract structured data from OCR text"""

    # Regex patterns
    PATTERNS = {
        "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "date": r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(?:\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
        "time": r"\b(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)\b",
        "invoice_number": r"(?:invoice|inv|invoice\s*#|no\.|number)[:\s]*([A-Z0-9\-]+)",
        "amount": r"\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        "total": r"(?:total|amount|grand\s*total|balance\s*due)[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        "subtotal": r"(?:subtotal|sub\s*total|net)[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        "tax": r"(?:tax|vat|gst|pst|hst)[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)",
        "payment_method": r"(?:payment|paid\s*by|method)[:\s]*(cash|credit|debit|card|visa|mastercard|amex|check|cheque|bank\s*transfer)",
        "riel": r"(៛\s*\d{1,3}(?:,\d{3})*)",
    }

    def extract(self, text: str) -> InvoiceData:
        """Extract structured data from OCR text"""
        lines = text.split("\n")

        # Initialize data
        merchant_name = None
        merchant_address = None
        merchant_phone = None
        invoice_number = None
        invoice_date = None
        invoice_time = None
        items: List[LineItem] = []
        payment = PaymentInfo()

        # Extract phone numbers
        phones = re.findall(self.PATTERNS["phone"], text)
        if phones:
            merchant_phone = phones[0]

        # Extract dates
        dates = re.findall(self.PATTERNS["date"], text)
        if dates:
            invoice_date = dates[0]

        # Extract times
        times = re.findall(self.PATTERNS["time"], text)
        if times:
            invoice_time = times[0]

        # Extract invoice number
        inv_match = re.search(self.PATTERNS["invoice_number"], text, re.IGNORECASE)
        if inv_match:
            invoice_number = inv_match.group(1).strip()

        # Extract amounts
        subtotal_match = re.search(self.PATTERNS["subtotal"], text, re.IGNORECASE)
        if subtotal_match:
            payment.subtotal = self._parse_amount(subtotal_match.group(1))

        tax_match = re.search(self.PATTERNS["tax"], text, re.IGNORECASE)
        if tax_match:
            payment.tax = self._parse_amount(tax_match.group(1))

        total_match = re.search(self.PATTERNS["total"], text, re.IGNORECASE)
        if total_match:
            payment.total = self._parse_amount(total_match.group(1))

        # Extract payment method
        payment_match = re.search(self.PATTERNS["payment_method"], text, re.IGNORECASE)
        if payment_match:
            payment.method = payment_match.group(1).title()

        # Extract line items (heuristic approach)
        items = self._extract_line_items(lines)

        # Extract merchant name (first non-empty line that's not a number/date)
        for line in lines:
            line = line.strip()
            if line and not self._is_numeric_line(line) and len(line) > 2:
                merchant_name = line
                break

        # Extract merchant address (lines containing street indicators)
        street_indicators = [
            "street",
            "st",
            "avenue",
            "ave",
            "boulevard",
            "blvd",
            "road",
            "rd",
            "drive",
            "dr",
            "lane",
            "ln",
        ]
        for line in lines:
            line_lower = line.lower()
            if any(ind in line_lower for ind in street_indicators):
                merchant_address = line.strip()
                break

        return InvoiceData(
            merchant_name=merchant_name,
            merchant_address=merchant_address,
            merchant_phone=merchant_phone,
            invoice_number=invoice_number,
            invoice_date=invoice_date,
            invoice_time=invoice_time,
            items=items,
            payment=payment
            if payment.subtotal or payment.tax or payment.total
            else None,
            dynamic_fields=self._extract_dynamic_fields(lines, text),
            raw_text=text,
        )

    def _extract_dynamic_fields(self, lines: List[str], text: str) -> Dict[str, Any]:
        """Build a flexible JSON map from OCR lines and common invoice patterns."""
        fields: Dict[str, Any] = {}

        def to_key(raw_key: str) -> str:
            key = raw_key.lower().strip()
            key = re.sub(r"[^a-z0-9]+", "_", key)
            key = re.sub(r"_+", "_", key).strip("_")
            return key or "field"

        def add_field(key: str, value: Any) -> None:
            if value is None:
                return
            value_str = str(value).strip()
            if not value_str:
                return

            if key not in fields:
                fields[key] = value_str
                return

            existing = fields[key]
            if isinstance(existing, list):
                if value_str not in existing:
                    existing.append(value_str)
            elif existing != value_str:
                fields[key] = [existing, value_str]

        # Parse generic "Label: Value" pairs.
        for line in lines:
            clean_line = line.strip()
            if not clean_line or ":" not in clean_line:
                continue

            left, right = clean_line.split(":", 1)
            key = to_key(left)
            add_field(key, right.strip())

        # Add common normalized values to make downstream usage easier.
        phone = re.findall(self.PATTERNS["phone"], text)
        if phone:
            add_field("phone", phone[0])

        date_match = re.findall(self.PATTERNS["date"], text)
        if date_match:
            add_field("date", date_match[0])

        time_match = re.findall(self.PATTERNS["time"], text)
        if time_match:
            add_field("time", time_match[0])

        inv_match = re.search(self.PATTERNS["invoice_number"], text, re.IGNORECASE)
        if inv_match:
            add_field("invoice_number", inv_match.group(1))

        total_match = re.search(self.PATTERNS["total"], text, re.IGNORECASE)
        if total_match:
            add_field("total", total_match.group(1))

        subtotal_match = re.search(self.PATTERNS["subtotal"], text, re.IGNORECASE)
        if subtotal_match:
            add_field("subtotal", subtotal_match.group(1))

        tax_match = re.search(self.PATTERNS["tax"], text, re.IGNORECASE)
        if tax_match:
            add_field("tax", tax_match.group(1))

        return fields

    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string to float"""
        # Remove currency symbols and commas
        cleaned = re.sub(r"[,$€£¥៛]", "", amount_str)
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def _is_numeric_line(self, line: str) -> bool:
        """Check if line is primarily numeric"""
        digits = sum(c.isdigit() for c in line)
        return digits > len(line) * 0.5

    def _extract_line_items(self, lines: List[str]) -> List[LineItem]:
        """Extract line items from text lines"""
        items = []

        # Look for lines that look like line items
        # Pattern: description followed by price
        item_pattern = r"^(.+?)\s+(\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*$"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip header-like lines
            if any(
                word in line.lower()
                for word in ["subtotal", "total", "tax", "invoice", "date", "payment"]
            ):
                continue

            # Try to match item pattern
            match = re.match(item_pattern, line)
            if match:
                name = match.group(1).strip()
                price = self._parse_amount(match.group(2))

                # Skip if name is too short or looks like a number
                if len(name) > 2 and not name.isdigit():
                    items.append(LineItem.create(name=name, price=price))

        # Limit to reasonable number of items
        return items[:20]
