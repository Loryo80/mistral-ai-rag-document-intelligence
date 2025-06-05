"""Extraction module for LegalDoc AI - handles PDF content extraction and processing"""

from .pdf_extractor import PDFExtractor, RETRY_OCR_CONFIG

__all__ = ["PDFExtractor"]