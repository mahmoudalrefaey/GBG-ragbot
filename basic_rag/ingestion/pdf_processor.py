from io import BytesIO
from pathlib import Path

import easyocr
import pymupdf
import numpy as np
from PIL import Image

from basic_rag.ingestion.preprocessor import preprocess_text


DPI = 250
TOP_CROP = 0.10
BOTTOM_CROP = 0.95


_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _reader

    if _reader is None:
        import torch

        use_gpu = torch.cuda.is_available()
        print(
            "Loading EasyOCR model for Arabic and English "
            f"(GPU: {'yes' if use_gpu else 'no'})..."
        )
        _reader = easyocr.Reader(["ar", "en"], gpu=use_gpu)
        print("EasyOCR model loaded.")

    return _reader


def render_page(page: pymupdf.Page) -> Image.Image:
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

    top = int(height * TOP_CROP)
    bottom = int(height * BOTTOM_CROP)

    return image.crop((0, top, width, bottom))


def ocr_page(image: Image.Image) -> str:
    results = _get_reader().readtext(
        np.asarray(image),
        detail=0,
        paragraph=True,
    )

    return preprocess_text("\n\n".join(results))


def process_pdf(pdf_path: str | Path) -> list[dict[str, object]]:
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

        print("Running EasyOCR...")
        text = ocr_page(image)

        if not text:
            continue

        pages.append({
            "page": page_number,
            "text": text,
        })

    doc.close()

    return pages