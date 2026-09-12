"""Text cleanup for OCR output before documents are chunked."""

import re
import unicodedata


_INVISIBLE_CHARACTERS = (
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\ufeff",
    "\u061c",
)

# These are recurring recognition errors found in the inspected OCR output.
# Keep this list conservative: cleanup must not rewrite valid policy content.
_OCR_CORRECTIONS = {
    "لحتوبات": "المحتويات",
    "المو افقات": "الموافقات",
    "المو جودات": "الموجودات",
    "المستود عات": "المستودعات",
    "أصحول": "أصول",
    "أصحولياً": "أصولياً",
    "الإضسافة": "الإضافة",
    "اعدادهذ": "إعداده",
    "حساي": "حساب",
    "عاى": "على",
    "نظا": "نظام",
    "التو صيات": "التوصيات",
    "استيفا تواقيع": "استيفاء تواقيع",
    "محاضرالاستلام": "محاضر الاستلام",
    "مراكزالعمل": "مراكز العمل",
    "الطلبيا ت": "الطلبيات",
    "أمنا المستودعات": "أمناء المستودعات",
}


def _replace_ocr_errors(text: str) -> str:
    for incorrect, corrected in _OCR_CORRECTIONS.items():
        pattern = (
            rf"(?<![\u0600-\u06ff]){re.escape(incorrect)}"
            rf"(?![\u0600-\u06ff])"
        )
        text = re.sub(pattern, corrected, text)
    return text


def preprocess_text(text: str) -> str:
    """Normalize whitespace and correct known, recurring OCR artifacts."""
    text = unicodedata.normalize("NFC", text)

    for character in _INVISIBLE_CHARACTERS:
        text = text.replace(character, "")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u0640", "")  # Arabic tatweel
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[.]{3,}", "...", text)
    text = re.sub(r"\s+([،؛:,.!?؟])", r"\1", text)
    text = _replace_ocr_errors(text)

    return text.strip()


# This name makes the pipeline call site explicit while keeping the
# implementation reusable for text loaded from other OCR sources.
clean_ocr_text = preprocess_text
