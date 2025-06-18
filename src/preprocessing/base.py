"""
Base preprocessor for document optimization
"""

import io
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
from PIL import Image
from pdf2image import convert_from_path

from ..config.settings import settings
from ..core.exceptions import PreprocessingError, InvalidDocumentError


class BasePreprocessor(ABC):
    """Base class for document preprocessing"""

    def __init__(self):
        self.max_size_kb = settings.IMAGE_RESIZE_THRESHOLD_KB
        self.resize_dim = settings.IMAGE_RESIZE_DIMENSION
        self.image_quality = settings.IMAGE_QUALITY
        self.pdf_dpi = settings.PDF_DPI

    def validate_document(self, file_path: str) -> None:
        """Validate document format and size"""
        path = Path(file_path)

        if not path.exists():
            raise InvalidDocumentError(f"File not found: {file_path}")

        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise InvalidDocumentError(
                f"File too large: {file_size_mb:.1f}MB (max: {settings.MAX_FILE_SIZE_MB}MB)"
            )

        # Check file format
        file_ext = path.suffix.lower().lstrip(".")
        if file_ext not in settings.SUPPORTED_FORMATS:
            raise InvalidDocumentError(
                f"Unsupported format: {file_ext}. Supported: {settings.SUPPORTED_FORMATS}"
            )

    def _prepare_image_content(self, file_path: str) -> Dict[str, Any]:
        """Prepare image content with optimization"""
        path = Path(file_path)
        mime_type, _ = mimetypes.guess_type(path)
        file_size_kb = path.stat().st_size / 1024

        try:
            if mime_type and path.stat().st_size > self.max_size_kb * 1024:
                if "image" in mime_type:
                    return self._optimize_image(path, file_size_kb)
                elif "pdf" in mime_type:
                    return self._convert_pdf_to_image(path, file_size_kb)

            # Use original file for small files
            return {
                "mime_type": mime_type or "application/octet-stream",
                "data": path.read_bytes(),
                "optimized": False,
                "original_size_kb": file_size_kb,
                "final_size_kb": file_size_kb,
            }

        except Exception as e:
            raise PreprocessingError(
                f"Failed to prepare document {file_path}: {str(e)}"
            )

    def _optimize_image(self, path: Path, original_size_kb: float) -> Dict[str, Any]:
        """Optimize image size and quality"""
        print(f"    Optimizing image {path.name} ({original_size_kb:.1f} KB)...")

        with Image.open(path) as img:
            # Apply resizing
            img.thumbnail((self.resize_dim, self.resize_dim))
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save optimized image
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG", quality=self.image_quality)
            img_byte_arr.seek(0)

            optimized_data = img_byte_arr.read()
            final_size_kb = len(optimized_data) / 1024

            print(
                f"      Optimized to {final_size_kb:.1f} KB ({final_size_kb/original_size_kb:.1%} of original)"
            )

            return {
                "mime_type": "image/jpeg",
                "data": optimized_data,
                "optimized": True,
                "original_size_kb": original_size_kb,
                "final_size_kb": final_size_kb,
            }

    def _convert_pdf_to_image(
        self, path: Path, original_size_kb: float
    ) -> Dict[str, Any]:
        """Convert PDF first page to optimized image"""
        print(f"    Converting PDF {path.name} ({original_size_kb:.1f} KB) to image...")

        try:
            # Convert first page to image
            images = convert_from_path(
                str(path), first_page=1, last_page=1, dpi=self.pdf_dpi, fmt="RGB"
            )

            if not images:
                raise PreprocessingError("PDF conversion produced no images")

            first_page_img = images[0]
            print(
                f"      Converted to image: {first_page_img.size[0]}x{first_page_img.size[1]} pixels"
            )

            # Apply optimization
            first_page_img.thumbnail((self.resize_dim, self.resize_dim))
            if first_page_img.mode != "RGB":
                first_page_img = first_page_img.convert("RGB")

            img_byte_arr = io.BytesIO()
            first_page_img.save(img_byte_arr, format="JPEG", quality=self.image_quality)
            img_byte_arr.seek(0)

            optimized_data = img_byte_arr.read()
            final_size_kb = len(optimized_data) / 1024

            print(
                f"      Optimized to {final_size_kb:.1f} KB ({final_size_kb/original_size_kb:.1%} of original)"
            )

            return {
                "mime_type": "image/jpeg",
                "data": optimized_data,
                "optimized": True,
                "original_size_kb": original_size_kb,
                "final_size_kb": final_size_kb,
                "conversion": "pdf_to_image",
            }

        except Exception as e:
            raise PreprocessingError(f"PDF to image conversion failed: {str(e)}")

    @abstractmethod
    def preprocess(self, file_path: str) -> Dict[str, Any]:
        """Preprocess document for specific use case"""
        pass
