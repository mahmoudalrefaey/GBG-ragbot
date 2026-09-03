from pathlib import Path
from io import BytesIO
import re
import unicodedata

import pymupdf
import pytesseract
from PIL import Image


DPI = 300
OCR_LANG = "ara"

TOP_CROP = 0.12
BOTTOM_CROP = 0.91
LEFT_CROP = 0.03
RIGHT_CROP = 0.97


def normalize_arabic_text(text: str) -> str:
    """
    Basic cleanup only.
    Does not spell-correct Arabic.
    """

    text = unicodedata.normalize("NFC", text)

    invisible_chars = [
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\ufeff",
        "\u061c",
    ]

    for char in invisible_chars:
        text = text.replace(char, "")

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([،؛:,.!?؟])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def render_page(page):
    pix = page.get_pixmap(
        dpi=DPI,
        colorspace=pymupdf.csRGB,
        alpha=False,
    )

    return Image.open(
        BytesIO(pix.tobytes("png"))
    )


def crop_content(image: Image.Image) -> Image.Image:
    width, height = image.size

    left = int(width * LEFT_CROP)
    top = int(height * TOP_CROP)

    right = int(width * RIGHT_CROP)
    bottom = int(height * BOTTOM_CROP)

    return image.crop(
        (left, top, right, bottom)
    )


def ocr_page(image: Image.Image) -> str:
    text = pytesseract.image_to_string(
        image,
        lang=OCR_LANG,
        config="--oem 3 --psm 6",
    )

    return normalize_arabic_text(text)


def process_pdf(pdf_path: str | Path):
    pdf_path = Path(pdf_path)

    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        print(
            f"OCR: {pdf_path.name} "
            f"[{page_number}/{len(doc)}]"
        )

        image = render_page(page)
        image = crop_content(image)

        text = ocr_page(image)

        if not text:
            continue

        pages.append({
            "page": page_number,
            "text": text,
        })

    doc.close()

    return pages