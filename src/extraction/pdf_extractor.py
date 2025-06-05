"""
Enhanced PDF Extraction Module for LegalDoc AI.

This module handles the extraction of text, tables, and metadata from legal and technical PDF documents
using Mistral OCR with fallback to other OCR engines if needed. Features include:
- Language detection for multilingual document support
- Chemical compound and technical parameter extraction
- Enhanced entity and relationship identification
- Structured data extraction with document classification
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import base64
from mistralai import Mistral
from pdf2image import convert_from_path
import pytesseract
from dotenv import load_dotenv
import PyPDF2
import concurrent.futures
import time
import json
import re
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_OCR_MODEL = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-2505")

def is_retryable_ocr_exception(exception):
    import requests
    retryable = (
        ConnectionError,
        TimeoutError,
        requests.exceptions.RequestException,
        Exception,  # Optionally catch all for now
    )
    return isinstance(exception, retryable)

RETRY_OCR_CONFIG = dict(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(4),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

class PDFExtractor:
    """Extract text and structure from legal PDF documents using enhanced Mistral OCR."""

    def __init__(self, api_key: Optional[str] = None, db=None, user_id=None):
        """
        Initialize the PDFExtractor with API credentials.

        Args:
            api_key: Mistral API key (defaults to environment variable)
            db: Database instance for logging API usage (optional)
            user_id: User ID for logging (optional)
        """
        self.api_key = api_key or MISTRAL_API_KEY
        if not self.api_key:
            logger.warning("No Mistral API key provided. Fallback OCR will be used.")
            self.client = None
        else:
            try:
                # Initialize the Mistral client for version 1.8.0+
                self.client = Mistral(api_key=self.api_key)
                logger.info("✅ Mistral OCR client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Mistral client: {e}")
                self.client = None
        
        self.fallback_enabled = True
        self.db = db
        self.user_id = user_id
        self.ocr_model = MISTRAL_OCR_MODEL

    def _encode_pdf_base64(self, pdf_path: str) -> str:
        """Encode the PDF file to a base64 string."""
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')

    def _encode_image_to_base64(self, image) -> str:
        """Encode PIL image to base64 string."""
        import io
        import base64
        
        # Save image to a temporary bytes buffer
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        img_byte_array.seek(0)
        
        # Encode the image to base64
        base64_image = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
        return base64_image

    def extract_from_pdf(self, pdf_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract content from a PDF file, optimized for legal documents.

        Args:
            pdf_path: Path to the PDF file
            document_id: Optional document ID for API usage logging

        Returns:
            Dictionary containing extracted text, tables, images, and metadata
        """
        logger.info(f"Processing PDF: {pdf_path}")
        # Use Mistral OCR best practice: upload file and use signed URL for large PDFs
        if self.api_key and self.client:
            try:
                return self._extract_with_mistral(pdf_path, document_id)
            except Exception as e:
                logger.warning(f"Mistral OCR failed: {e}. Falling back to alternative method.")
                if not self.fallback_enabled:
                    raise
        # Fallback to traditional OCR
        return self._extract_with_fallback(pdf_path)

    def _detect_document_language(self, pdf_path: str) -> str:
        """Detect the primary language of the document."""
        try:
            # Check if langdetect is available
            try:
                import langdetect
            except ImportError:
                logger.warning("langdetect module not available, defaulting to English")
                return "en"
            
            # Extract a sample text from the first few pages for language detection
            sample_text = ""
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                # Get text from first 3 pages or fewer if document is shorter
                for page_num in range(min(3, len(reader.pages))):
                    page = reader.pages[page_num]
                    page_text = page.extract_text() or ""
                    sample_text += page_text + "\n\n"
                    if len(sample_text) > 1000:  # 1000 chars should be enough for detection
                        break
            
            if not sample_text.strip():
                logger.warning("No text extracted for language detection, defaulting to English")
                return "en"
            
            # Detect language
            try:
                lang = langdetect.detect(sample_text)
                logger.info(f"Detected document language: {lang}")
                return lang
            except Exception as e:
                logger.warning(f"Language detection failed: {e}, defaulting to English")
                return "en"
        except Exception as e:
            logger.warning(f"Error in language detection: {e}. Defaulting to English.")
            return "en"

    @retry(**RETRY_OCR_CONFIG)
    def _extract_with_mistral(self, pdf_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Use Mistral OCR API to extract content from PDF using best practices for legal documents."""
        
        logger.info("Converting PDF to images for Mistral OCR processing")
        try:
            # Detect document language for better OCR prompting
            primary_language = self._detect_document_language(pdf_path)
            self._current_language = primary_language  # Store for logging
            logger.info(f"Detected document language: {primary_language}")
            
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            logger.info(f"PDF converted to {len(images)} images")
            
            all_text = ""
            all_tables = []
            all_structured_data = {"sections": [], "technical_specs": [], "safety_info": []}
            
            for i, image in enumerate(images):
                try:
                    # Enhanced prompting for different page types based on content detection
                    if i == 0:  # First page - likely contains title, product info, overview
                        prompt = """Extract all text from this agricultural/chemical document page. Focus on:
                        1. Product name, brand, and manufacturer information
                        2. Document type (SDS, Technical Datasheet, Product Specification, etc.)
                        3. Key identification numbers (CAS, EINECS, registration numbers)
                        4. Active ingredients and concentrations
                        
                        If tables are present, represent them as structured JSON:
                        {"table_type": "product_overview|specifications|composition", "headers": ["Header1", "Header2"], "rows": [["Value1", "Value2"]], "caption": "Optional caption"}
                        
                        Identify section headers clearly and preserve document structure."""
                        
                    elif any(keyword in all_text.lower() for keyword in ['hazard', 'safety', 'precaution', 'emergency']):
                        # Safety-focused page
                        prompt = """Extract all safety and hazard information from this document page:
                        1. GHS classifications and pictograms
                        2. Hazard statements (H-codes) and precautionary statements (P-codes)
                        3. Emergency procedures and first aid measures
                        4. Personal protective equipment (PPE) requirements
                        5. Environmental hazards and precautions
                        
                        For tables with safety data, use JSON format:
                        {"table_type": "safety_data|hazard_classification|emergency_procedures", "headers": [...], "rows": [...]}
                        
                        Extract exact regulatory codes and classification numbers."""
                        
                    elif any(keyword in all_text.lower() for keyword in ['specification', 'properties', 'analysis', 'composition']):
                        # Technical specifications page
                        prompt = """Extract technical specifications and analytical data:
                        1. Physical and chemical properties (pH, density, viscosity, etc.)
                        2. Composition and active ingredient percentages
                        3. Quality specifications and test methods
                        4. Storage conditions and stability data
                        
                        For specification tables, use JSON format:
                        {"table_type": "technical_specs|composition|quality_parameters", "headers": ["Parameter", "Value", "Unit", "Method"], "rows": [...]}
                        
                        Preserve exact numerical values with units."""
                        
                    elif any(keyword in all_text.lower() for keyword in ['application', 'use', 'dosage', 'rate']):
                        # Application information page
                        prompt = """Extract application and usage information:
                        1. Target crops, pests, diseases, or weeds
                        2. Application rates and dosages (kg/ha, L/ha, ppm, etc.)
                        3. Application methods (foliar, soil, seed treatment, fertigation)
                        4. Timing of application (growth stages, pre/post-emergence)
                        5. Mixing instructions and compatibility
                        
                        For application tables, use JSON format:
                        {"table_type": "application_rates|target_crops|mixing_instructions", "headers": [...], "rows": [...]}"""
                        
                    else:
                        # General page processing
                        prompt = """Extract all text and data from this agricultural/chemical document page.
                        If tables are present, represent them as structured JSON objects:
                        {"table_type": "general|specifications|composition|application|safety", "headers": ["Header1", "Header2"], "rows": [["Row1Col1", "Row1Col2"]], "caption": "Optional caption"}
                        
                        Identify section titles, subsections, and preserve document hierarchy.
                        Extract technical specifications, safety information, and application data completely."""

                    # Process with Mistral OCR API using correct 1.8.0 syntax
                    response = self.client.ocr.process(
                        model=self.ocr_model,
                        document={
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{self._encode_image_to_base64(image)}"
                        }
                    )
                    
                    # Extract text from OCR response format (Mistral 1.8.0)
                    page_text = ""
                    if hasattr(response, 'pages') and response.pages:
                        # Get the first page's markdown content (OCRPageObject)
                        page_obj = response.pages[0]
                        if hasattr(page_obj, 'markdown'):
                            page_text = page_obj.markdown
                        else:
                            # Fallback to string representation
                            page_text = str(page_obj)
                    else:
                        page_text = str(response)
                    
                    all_text += f"\n--- Page {i+1} ---\n{page_text}\n"
                    
                    # Extract structured data from this page
                    page_structured_data = self._extract_structured_data_from_page(page_text, i+1)
                    
                    # Merge structured data
                    for key in all_structured_data.keys():
                        if key in page_structured_data:
                            all_structured_data[key].extend(page_structured_data[key])
                    
                    # Extract tables from JSON format in the response
                    page_tables = self._extract_json_tables_from_text(page_text, i+1)
                    all_tables.extend(page_tables)
                    
                    # Log successful extraction with document ID
                    self._log_api_usage("mistral", "ocr", len(page_text.split()), response, document_id, i+1)
                    
                except Exception as e:
                    logger.error(f"Error processing page {i+1} with Mistral: {e}")
                    all_text += f"\n--- Page {i+1} (Error) ---\nFailed to extract content\n"
            
            # Combine all text
            combined_text = all_text.strip()
            
            # Extract tables from the combined text
            tables = all_tables
            
            # Merge structured data from all pages
            metadata = self._extract_metadata(pdf_path)
            merged_structured_data = self._merge_structured_data([all_structured_data], metadata)
            
            result = {
                "text": combined_text,
                "tables": tables,
                "images": [],
                "metadata": metadata,
                "structured_data": merged_structured_data,
                "source": pdf_path,
                "entities": merged_structured_data.get("entities", []),
                "relationships": merged_structured_data.get("relationships", [])
            }
            
            logger.info(f"Mistral OCR extraction complete: {len(combined_text)} characters, {len(tables)} tables, {len(merged_structured_data.get('entities', []))} entities")
            return result
            
        except Exception as e:
            logger.error(f"Error in Mistral OCR extraction: {e}")
            raise
    
    def _extract_json_tables_from_text(self, text: str, page_number: int) -> List[Dict[str, Any]]:
        """Extract JSON-formatted tables from Mistral's response."""
        tables = []
        
        # Look for JSON table structures in the response
        import re
        json_pattern = r'\{[^{}]*"table_type"[^{}]*"headers"[^{}]*"rows"[^{}]*\}'
        
        try:
            matches = re.finditer(json_pattern, text, re.DOTALL)
            for i, match in enumerate(matches):
                try:
                    table_json = json.loads(match.group())
                    tables.append({
                        "id": f"page_{page_number}_table_{i+1}",
                        "page_number": page_number,
                        "table_type": table_json.get("table_type", "general"),
                        "headers": table_json.get("headers", []),
                        "rows": table_json.get("rows", []),
                        "caption": table_json.get("caption", ""),
                        "json_content": table_json
                    })
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            logger.warning(f"Error extracting JSON tables from page {page_number}: {e}")
        
        # Fallback to markdown table extraction if no JSON tables found
        if not tables:
            markdown_tables = self._extract_tables_from_markdown(text)
            for i, md_table in enumerate(markdown_tables):
                tables.append({
                    "id": f"page_{page_number}_md_table_{i+1}",
                    "page_number": page_number,
                    "table_type": "markdown_extracted",
                    "headers": md_table.get("headers", []),
                    "rows": md_table.get("rows", []),
                    "caption": "",
                    "markdown_content": md_table
                })
        
        return tables

    def _extract_structured_data_from_page(self, page_text: str, page_number: int) -> Dict[str, List[Dict[str, Any]]]:
        """Enhanced extraction of structured data elements from page text."""
        structured_data = {"sections": [], "technical_specs": [], "safety_info": []}
        
        lines = page_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Enhanced section header detection for agricultural documents
            section_patterns = [
                r'^(?:SECTION\s+)?(\d+)\.?\s*([A-Z][A-Z\s&/,-]+)$',  # SECTION 1. IDENTIFICATION
                r'^(\d+)\.(\d+)\.?\s*([A-Z][A-Z\s&/,-]+)$',  # 1.1. PRODUCT IDENTIFIER
                r'^([A-Z][A-Z\s&/,-]{10,})$',  # ALL CAPS HEADERS
                r'^([A-Z][a-z][^:]{15,}):?\s*$',  # Title case headers
            ]
            
            is_header = False
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous section
                    if current_section and current_content:
                        structured_data["sections"].append({
                            "title": current_section,
                            "content": '\n'.join(current_content),
                            "page": page_number,
                            "section_type": self._classify_section_type(current_section)
                        })
                    
                    # Start new section
                    if len(match.groups()) >= 2:
                        current_section = f"{match.group(1)}. {match.group(2)}" if match.group(1).isdigit() else match.group(1)
                    else:
                        current_section = match.group(1)
                    current_content = []
                    is_header = True
                    break
            
            if not is_header:
                current_content.append(line)
                
                # Extract technical specifications on the fly
                spec_match = re.search(r'([A-Za-z\s]+):\s*([0-9.,\s-]+)\s*([A-Za-z/%°]+)?', line)
                if spec_match:
                    structured_data["technical_specs"].append({
                        "parameter": spec_match.group(1).strip(),
                        "value": spec_match.group(2).strip(),
                        "unit": spec_match.group(3).strip() if spec_match.group(3) else "",
                        "page": page_number,
                        "context": line
                    })
                
                # Extract safety codes (GHS, H-codes, P-codes)
                safety_codes = re.findall(r'\b(GHS\d+|H\d{3}|P\d{3}|EUH\d{3})\b', line)
                for code in safety_codes:
                    structured_data["safety_info"].append({
                        "code": code,
                        "type": "GHS" if code.startswith("GHS") else "H-code" if code.startswith("H") else "P-code" if code.startswith("P") else "EUH-code",
                        "page": page_number,
                        "context": line
                    })
        
        # Save last section
        if current_section and current_content:
            structured_data["sections"].append({
                "title": current_section,
                "content": '\n'.join(current_content),
                "page": page_number,
                "section_type": self._classify_section_type(current_section)
            })
        
        return structured_data

    def _classify_section_type(self, section_title: str) -> str:
        """Classify section type for agricultural/chemical documents."""
        title_lower = section_title.lower()
        
        if any(keyword in title_lower for keyword in ['identification', 'product', 'overview']):
            return "product_identification"
        elif any(keyword in title_lower for keyword in ['hazard', 'classification', 'danger']):
            return "hazard_classification"
        elif any(keyword in title_lower for keyword in ['composition', 'ingredient', 'component']):
            return "composition"
        elif any(keyword in title_lower for keyword in ['first aid', 'emergency', 'medical']):
            return "emergency_measures"
        elif any(keyword in title_lower for keyword in ['handling', 'storage', 'precaution']):
            return "handling_storage"
        elif any(keyword in title_lower for keyword in ['physical', 'chemical', 'properties']):
            return "physical_chemical_properties"
        elif any(keyword in title_lower for keyword in ['stability', 'reactivity']):
            return "stability_reactivity"
        elif any(keyword in title_lower for keyword in ['toxicological', 'health', 'exposure']):
            return "toxicological_information"
        elif any(keyword in title_lower for keyword in ['ecological', 'environmental', 'ecotoxicity']):
            return "ecological_information"
        elif any(keyword in title_lower for keyword in ['disposal', 'waste']):
            return "disposal_considerations"
        elif any(keyword in title_lower for keyword in ['transport', 'shipping']):
            return "transport_information"
        elif any(keyword in title_lower for keyword in ['regulatory', 'legal', 'classification']):
            return "regulatory_information"
        elif any(keyword in title_lower for keyword in ['application', 'use', 'instruction']):
            return "application_instructions"
        else:
            return "general"
    
    def _merge_structured_data(self, page_data_list: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Merge structured data from multiple pages and extract relationships."""
        merged_data = {
            "entities": [],
            "relationships": [],
            "key_value_pairs": [],
            "document_title": "",
            "document_type": "",
            "document_date": "",
            "document_number": "",
            "document_classification": "unknown"
        }
        
        # First, collect all entities
        all_entities = []
        all_key_value_pairs = []
        
        for page_data in page_data_list:
            all_entities.extend(page_data.get("entities", []))
            all_key_value_pairs.extend(page_data.get("key_value_pairs", []))
        
        # Deduplicate entities
        unique_entities = []
        seen = set()
        for entity in all_entities:
            key = (entity["type"], entity["value"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        merged_data["entities"] = unique_entities
        merged_data["key_value_pairs"] = all_key_value_pairs
        
        # Extract important document metadata from entities
        # Document title - usually from first page
        title_entities = [e for e in unique_entities if e["type"] == "title" or e["type"] == "name"]
        if title_entities and len(title_entities[0]["value"]) > 5:
            merged_data["document_title"] = title_entities[0]["value"]
        
        # Document type - look for common document type indicators
        doc_type_indicators = ["specification", "certificate", "report", "analysis", "safety data sheet", "sds", 
                             "msds", "technical data", "data sheet", "brochure"]
        for entity in unique_entities:
            entity_value_lower = entity["value"].lower()
            if any(indicator in entity_value_lower for indicator in doc_type_indicators):
                merged_data["document_type"] = entity["value"]
                break
        
        # Document date - most recent date from first page is usually the document date
        date_entities = [e for e in unique_entities if e["type"] == "date"]
        if date_entities:
            # Sort by page number (earlier pages more likely to have document date)
            date_entities.sort(key=lambda x: x.get("page", 999))
            merged_data["document_date"] = date_entities[0]["value"]
        
        # Document number - look for identifiers
        id_entities = [e for e in unique_entities if e["type"] == "identifier" or e["type"] == "id" or e["type"] == "reference"]
        if id_entities:
            merged_data["document_number"] = id_entities[0]["value"]
            
        # Document classification - determine document's primary category
        doc_classification = "unknown"
        
        text_content = " ".join([e["value"] for e in unique_entities])
        text_content = text_content.lower()
        
        if any(term in text_content for term in ["safety data sheet", "sds", "msds", "hazard", "precaution"]):
            doc_classification = "safety_data_sheet"
        elif any(term in text_content for term in ["specification", "technical data", "standard", "requirements"]):
            doc_classification = "technical_specification"
        elif any(term in text_content for term in ["certificate", "certification", "certified", "compliance"]):
            doc_classification = "certificate"
        elif any(term in text_content for term in ["brochure", "marketing", "product overview"]):
            doc_classification = "marketing_material"
            
        merged_data["document_classification"] = doc_classification
        
        # Extract relationships between entities
        # 1. Product-property relationships
        product_entities = [e for e in unique_entities if e["type"] in ["product", "material", "substance", "chemical_compound"]]
        parameter_entities = [e for e in unique_entities if e["type"] in ["parameter", "property", "specification", "value"]]
        
        for product_entity in product_entities:
            # Look for properties that might be related to this product
            for param_entity in parameter_entities:
                # Check if they're on the same page or adjacent pages
                if abs(product_entity.get("page", 0) - param_entity.get("page", 0)) <= 1:
                    merged_data["relationships"].append({
                        "type": "has_property",
                        "entity_1": product_entity["value"],
                        "entity_1_type": product_entity["type"],
                        "entity_2": param_entity["value"],
                        "entity_2_type": param_entity["type"],
                        "context": f"Product property relationship from page {product_entity.get('page', '?')}"
                    })
        
        # 2. Regulatory relationships
        regulatory_entities = [e for e in unique_entities if e["type"] == "regulatory"]
        
        for product_entity in product_entities:
            for reg_entity in regulatory_entities:
                # Regulatory entities often apply to products
                merged_data["relationships"].append({
                    "type": "regulatory_compliance",
                    "entity_1": product_entity["value"],
                    "entity_1_type": product_entity["type"],
                    "entity_2": reg_entity["value"],
                    "entity_2_type": reg_entity["type"],
                    "context": f"Regulatory information applicable to product"
                })
        
        # 3. Organization-product relationships
        org_entities = [e for e in unique_entities if e["type"] == "organization"]
        
        if org_entities and product_entities:
            # Assuming the primary organization in the document is the manufacturer/supplier
            primary_org = org_entities[0]["value"]
            
            for product_entity in product_entities:
                merged_data["relationships"].append({
                    "type": "manufactures",
                    "entity_1": primary_org,
                    "entity_1_type": "organization",
                    "entity_2": product_entity["value"],
                    "entity_2_type": product_entity["type"],
                    "context": f"Organization likely manufactures/supplies this product"
                })
        
        # 4. Related products/compounds
        if len(product_entities) > 1:
            for i, product1 in enumerate(product_entities):
                for j, product2 in enumerate(product_entities[i+1:], i+1):
                    merged_data["relationships"].append({
                        "type": "related_product",
                        "entity_1": product1["value"],
                        "entity_1_type": product1["type"],
                        "entity_2": product2["value"],
                        "entity_2_type": product2["type"],
                        "context": f"Products mentioned in same document"
                    })
        
        # Add document metadata
        merged_data["metadata"] = metadata
        
        return merged_data
    
    def _extract_tables_from_markdown(self, markdown_text: str) -> List[Dict[str, Any]]:
        """Extract tables from markdown formatted text with enhanced structure preservation."""
        tables = []
        lines = markdown_text.split('\n')
        current_table = []
        in_table = False
        table_metadata = {}
        
        for i, line in enumerate(lines):
            # Look for table captions or titles (often appear before tables)
            if not in_table and i > 0 and i+1 < len(lines) and '|' in lines[i+1]:
                if lines[i].strip() and '|' not in lines[i]:
                    table_metadata = {"caption": lines[i].strip()}
            
            # Table detection with improved pattern matching
            if ('|' in line and '-+-' in line.replace(' ', '')) or ('|' in line and line.count('|') > 2):
                in_table = True
                current_table.append(line)
            elif in_table and ('|' not in line or line.strip() == ''):
                # End of table
                if current_table:
                    # Extract header and determine column types
                    headers = []
                    if len(current_table) > 1:
                        header_row = current_table[0]
                        headers = [h.strip() for h in header_row.split('|') if h.strip()]
                    
                    table_data = {
                        "content": '\n'.join(current_table),
                        "format": "markdown",
                        "headers": headers,
                        "row_count": len(current_table) - 2 if len(current_table) > 2 else 0,
                        "column_count": len(headers)
                    }
                    
                    # Add any metadata like captions
                    if table_metadata:
                        table_data.update(table_metadata)
                    
                    tables.append(table_data)
                    
                current_table = []
                in_table = False
                table_metadata = {}
        
        # Handle the last table if there is one
        if current_table:
            headers = []
            if len(current_table) > 1:
                header_row = current_table[0]
                headers = [h.strip() for h in header_row.split('|') if h.strip()]
            
            table_data = {
                "content": '\n'.join(current_table),
                "format": "markdown",
                "headers": headers,
                "row_count": len(current_table) - 2 if len(current_table) > 2 else 0,
                "column_count": len(headers)
            }
            
            if table_metadata:
                table_data.update(table_metadata)
                
            tables.append(table_data)
        
        return tables
    
    def _extract_with_fallback(self, pdf_path: str) -> Dict[str, Any]:
        """
        Fallback extraction method using PyPDF2 and Tesseract.
        Optimized for legal documents: attempts to extract tables and images if possible.
        """
        logger.info("Using fallback extraction method")
        extracted_text = ""
        try:
            with open(pdf_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text() or ""
                    extracted_text += page_text + "\n\n"
        except Exception as e:
            logger.warning(f"PyPDF2 text extraction failed: {e}")
        
        # If PyPDF2 extraction yielded minimal text, use Tesseract OCR
        if len(extracted_text.strip()) < 100:
            logger.info("Text extraction yielded minimal content. Using Tesseract OCR.")
            extracted_text = self._extract_with_tesseract(pdf_path)
        
        metadata = self._extract_metadata(pdf_path)
        
        # Process text for structured data even in fallback mode
        structured_data = {"entities": [], "relationships": []}
        try:
            # Extract structured data from each page
            all_structured_data = []
            # Split text by pages (rough approximation)
            pages = extracted_text.split("\n\n\n")
            for i, page_text in enumerate(pages):
                if page_text.strip():
                    page_data = self._extract_structured_data_from_page(page_text, i+1)
                    all_structured_data.append(page_data)
            
            structured_data = self._merge_structured_data(all_structured_data, metadata)
        except Exception as e:
            logger.warning(f"Failed to extract structured data in fallback mode: {e}")
        
        return {
            "text": extracted_text,
            "tables": self._extract_tables_from_markdown(extracted_text),
            "images": [],
            "metadata": metadata,
            "structured_data": structured_data,
            "entities": structured_data.get("entities", []),
            "relationships": structured_data.get("relationships", []),
            "source": pdf_path
        }
    
    def _extract_with_tesseract(self, pdf_path: str) -> str:
        """Extract text from PDF using Tesseract OCR, parallelized for speed."""
        logger.info("Converting PDF to images for OCR processing")
        try:
            images = convert_from_path(pdf_path)
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            return ""
        
        extracted_texts = [None] * len(images)
        
        def ocr_page(i_image):
            i, image = i_image
            logger.info(f"Processing page {i+1}/{len(images)} (Tesseract, parallel)")
            try:
                return i, pytesseract.image_to_string(image)
            except Exception as e:
                logger.error(f"Tesseract OCR failed for page {i+1}: {e}")
                return i, ""
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(ocr_page, enumerate(images)))
        
        for i, page_text in results:
            extracted_texts[i] = page_text
        
        logger.info(f"Tesseract OCR completed for {len(images)} pages in {time.time()-start_time:.2f}s (parallel)")
        return "\n\n".join(filter(None, extracted_texts))
    
    def _extract_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract enhanced metadata from PDF document with intelligent content analysis."""
        metadata = {
            "num_pages": 0,
            "title": "",
            "author": "",
            "subject": "",
            "creator": "",
            "producer": "",
            "creation_date": "",
            # Enhanced metadata fields
            "file_size_mb": 0.0,
            "document_category": "unknown",
            "primary_language": "en",
            "content_keywords": [],
            "document_structure": {},
            "quality_indicators": {},
            "compliance_info": {},
            "technical_specs": {},
            "business_context": {}
        }
        
        try:
            # Basic file information
            import os
            file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Convert to MB
            metadata["file_size_mb"] = round(file_size, 2)
            
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                
                metadata["num_pages"] = len(reader.pages)
                
                # Extract basic PDF metadata
                if reader.metadata:
                    metadata.update({
                        "title": reader.metadata.get("/Title", ""),
                        "author": reader.metadata.get("/Author", ""),
                        "subject": reader.metadata.get("/Subject", ""),
                        "creator": reader.metadata.get("/Creator", ""),
                        "producer": reader.metadata.get("/Producer", ""),
                        "creation_date": str(reader.metadata.get("/CreationDate", ""))
                    })
                
                # Extract and analyze text content for enhanced metadata
                sample_text = ""
                for page_num in range(min(3, len(reader.pages))):
                    page = reader.pages[page_num]
                    page_text = page.extract_text() or ""
                    sample_text += page_text + "\n"
                
                if sample_text.strip():
                    # Enhance metadata with content analysis
                    metadata.update(self._analyze_document_content(sample_text, pdf_path))
                
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
        
        return metadata
    
    def _analyze_document_content(self, text: str, pdf_path: str) -> Dict[str, Any]:
        """Analyze document content to generate enhanced metadata."""
        analysis = {
            "document_category": "unknown",
            "primary_language": "en",
            "content_keywords": [],
            "document_structure": {},
            "quality_indicators": {},
            "compliance_info": {},
            "technical_specs": {},
            "business_context": {}
        }
        
        text_lower = text.lower()
        filename = os.path.basename(pdf_path).lower()
        
        # Document categorization
        analysis["document_category"] = self._categorize_document(text_lower, filename)
        
        # Language detection
        analysis["primary_language"] = self._detect_document_language(pdf_path)
        
        # Extract key content indicators
        analysis["content_keywords"] = self._extract_content_keywords(text_lower)
        
        # Document structure analysis
        analysis["document_structure"] = self._analyze_document_structure(text)
        
        # Quality indicators
        analysis["quality_indicators"] = self._assess_document_quality(text)
        
        # Compliance and regulatory information
        analysis["compliance_info"] = self._extract_compliance_info(text_lower)
        
        # Technical specifications
        analysis["technical_specs"] = self._extract_technical_specs(text_lower)
        
        # Business context
        analysis["business_context"] = self._extract_business_context(text_lower, filename)
        
        return analysis
    
    def _categorize_document(self, text: str, filename: str) -> str:
        """Categorize document based on content and filename."""
        categories = {
            "safety_data_sheet": ["sds", "msds", "safety data sheet", "hazard", "risk assessment"],
            "technical_specification": ["specification", "technical data", "parameters", "properties", "datasheet"],
            "legal_contract": ["agreement", "contract", "terms", "conditions", "party", "clause"],
            "compliance_certificate": ["certificate", "conformity", "compliance", "standard", "iso", "gmp"],
            "product_brochure": ["brochure", "product information", "marketing", "features", "benefits"],
            "research_report": ["research", "study", "analysis", "findings", "methodology", "results"],
            "financial_document": ["financial", "budget", "cost", "invoice", "payment", "accounting"],
            "manufacturing_record": ["batch", "lot", "production", "manufacturing", "quality control"],
            "regulatory_submission": ["submission", "regulatory", "approval", "license", "permit"]
        }
        
        # Check filename first
        for category, keywords in categories.items():
            if any(keyword in filename for keyword in keywords):
                return category
        
        # Check content
        for category, keywords in categories.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text)
            if keyword_count >= 2:  # At least 2 keywords match
                return category
        
        return "general_document"
    
    def _extract_content_keywords(self, text: str) -> List[str]:
        """Extract important keywords from document content."""
        keywords = []
        
        # Chemical and pharmaceutical terms
        chemical_patterns = [
            r'\b\w+\s*®\b',  # Trademarked products
            r'\bCAS\s*\w*\s*\d+[-]\d+[-]\d+\b',  # CAS numbers
            r'\b\w+\s*\(\w+\)\b',  # Chemical names with abbreviations
            r'\b(?:mg|kg|ml|°C|pH|ppm|%)\b'  # Units
        ]
        
        for pattern in chemical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend([match.strip() for match in matches[:5]])  # Limit to 5 per pattern
        
        # Technical standards
        standards = re.findall(r'\b(?:ISO|ASTM|EN|DIN|USP|EP|JP|CP)\s*[-]?\s*\w*\s*\d+\b', text, re.IGNORECASE)
        keywords.extend(standards[:5])
        
        # Company names (capitalized words, potentially with symbols)
        companies = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*®|\s*™)?\b', text)
        # Filter out common words and keep only likely company names
        company_keywords = [comp for comp in set(companies) if len(comp) > 3 and comp not in ['The', 'This', 'That', 'Page']]
        keywords.extend(company_keywords[:3])
        
        return list(set(keywords))[:15]  # Return unique keywords, max 15
    
    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structural elements of the document."""
        structure = {
            "has_table_of_contents": False,
            "section_count": 0,
            "has_tables": False,
            "has_references": False,
            "has_appendices": False,
            "estimated_reading_time_minutes": 0
        }
        
        # Check for table of contents
        if re.search(r'table\s+of\s+contents|contents|index', text, re.IGNORECASE):
            structure["has_table_of_contents"] = True
        
        # Count sections (numbered headings)
        sections = re.findall(r'\n\s*\d+[\.\)]\s+[A-Z]', text)
        structure["section_count"] = len(sections)
        
        # Check for tables
        if re.search(r'\|.*\||\btable\b|\btab\b', text, re.IGNORECASE):
            structure["has_tables"] = True
        
        # Check for references
        if re.search(r'references|bibliography|citations', text, re.IGNORECASE):
            structure["has_references"] = True
        
        # Check for appendices
        if re.search(r'appendix|annex', text, re.IGNORECASE):
            structure["has_appendices"] = True
        
        # Estimate reading time (average 200 words per minute)
        word_count = len(text.split())
        structure["estimated_reading_time_minutes"] = max(1, round(word_count / 200))
        
        return structure
    
    def _assess_document_quality(self, text: str) -> Dict[str, Any]:
        """Assess various quality indicators of the document."""
        quality = {
            "text_extraction_confidence": "high",
            "has_complete_sentences": True,
            "character_count": len(text),
            "word_count": len(text.split()),
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "contains_special_characters": False,
            "language_consistency": True
        }
        
        # Check for OCR quality indicators
        special_char_count = len(re.findall(r'[^\w\s\.,;:!?\-()[\]{}"\'/]', text))
        if special_char_count > len(text) * 0.02:  # More than 2% special chars
            quality["contains_special_characters"] = True
            quality["text_extraction_confidence"] = "medium"
        
        # Check for complete sentences
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = [s for s in sentences if len(s.strip().split()) >= 3]
        if len(complete_sentences) < len(sentences) * 0.7:  # Less than 70% complete
            quality["has_complete_sentences"] = False
            quality["text_extraction_confidence"] = "low"
        
        return quality
    
    def _extract_compliance_info(self, text: str) -> Dict[str, Any]:
        """Extract compliance and regulatory information."""
        compliance = {
            "standards_mentioned": [],
            "regulatory_bodies": [],
            "certifications": [],
            "compliance_statements": []
        }
        
        # Standards
        standards = re.findall(r'\b(?:ISO|ASTM|EN|DIN|USP|EP|JP|CP|GMP|FDA|EMA)\s*[-]?\s*\w*\s*\d*\b', text, re.IGNORECASE)
        compliance["standards_mentioned"] = list(set(standards))[:10]
        
        # Regulatory bodies
        bodies = re.findall(r'\b(?:FDA|EMA|MHRA|ANVISA|PMDA|NMPA|WHO)\b', text, re.IGNORECASE)
        compliance["regulatory_bodies"] = list(set(bodies))
        
        # Certifications
        certs = re.findall(r'\b(?:certificate|certified|certification|accredited|approved)\s+\w+', text, re.IGNORECASE)
        compliance["certifications"] = list(set(certs))[:5]
        
        # Compliance statements
        statements = re.findall(r'(?:complies with|conforms to|meets|accordance with)[^.]*', text, re.IGNORECASE)
        compliance["compliance_statements"] = statements[:3]
        
        return compliance
    
    def _extract_technical_specs(self, text: str) -> Dict[str, Any]:
        """Extract technical specifications and parameters."""
        specs = {
            "parameters_found": [],
            "units_used": [],
            "ranges_specified": [],
            "test_methods": []
        }
        
        # Parameters with values
        params = re.findall(r'(\w+(?:\s+\w+)*)\s*[:=]\s*([0-9.,]+(?:\s*[-–]\s*[0-9.,]+)?\s*\w*)', text)
        specs["parameters_found"] = [f"{param}: {value}" for param, value in params[:10]]
        
        # Units
        units = re.findall(r'\b\d+(?:\.\d+)?\s*([a-zA-Z]+/[a-zA-Z]+|°C|pH|%|ppm|mg|kg|ml|L|mm|cm|m)\b', text)
        specs["units_used"] = list(set(units))[:10]
        
        # Ranges
        ranges = re.findall(r'\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*\w*', text)
        specs["ranges_specified"] = ranges[:5]
        
        # Test methods
        methods = re.findall(r'(?:test|method|procedure|analysis)\s+[A-Z][^.]*', text, re.IGNORECASE)
        specs["test_methods"] = methods[:3]
        
        return specs
    
    def _extract_business_context(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract business and commercial context."""
        context = {
            "company_names": [],
            "product_names": [],
            "document_purpose": "unknown",
            "target_audience": "general",
            "commercial_terms": []
        }
        
        # Company names (capitalized sequences)
        companies = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*(?:Inc|Ltd|LLC|Corp|GmbH|SA)\.?)?', text)
        context["company_names"] = list(set([comp for comp in companies if len(comp) > 2]))[:5]
        
        # Product names (trademark symbols or all caps)
        products = re.findall(r'\b\w+\s*®|\b[A-Z]{2,}\b', text)
        context["product_names"] = list(set(products))[:5]
        
        # Document purpose
        if any(word in text for word in ["specification", "datasheet"]):
            context["document_purpose"] = "technical_reference"
        elif any(word in text for word in ["safety", "hazard"]):
            context["document_purpose"] = "safety_information"
        elif any(word in text for word in ["agreement", "contract"]):
            context["document_purpose"] = "legal_binding"
        elif any(word in text for word in ["certificate", "compliance"]):
            context["document_purpose"] = "compliance_verification"
        
        # Target audience
        if any(word in text for word in ["technical", "specification", "analysis"]):
            context["target_audience"] = "technical_professionals"
        elif any(word in text for word in ["safety", "warning", "precaution"]):
            context["target_audience"] = "safety_personnel"
        elif any(word in text for word in ["legal", "contract", "terms"]):
            context["target_audience"] = "legal_professionals"
        
        # Commercial terms
        commercial = re.findall(r'\b(?:price|cost|fee|payment|invoice|purchase|sale|order)\b[^.]*', text, re.IGNORECASE)
        context["commercial_terms"] = commercial[:3]
        
        return context
    
    def save_to_supabase(self, extraction_result: Dict[str, Any], project_id: str, filename: str) -> Dict[str, Any]:
        """
        Save extracted PDF data to Supabase database.
        
        Args:
            extraction_result: The result from extract_from_pdf
            project_id: The project ID to associate this document with
            filename: Original filename
            
        Returns:
            Dictionary with document ID and other storage information
        """
        if not self.db:
            raise ValueError("Database connection required for storage. Initialize PDFExtractor with db parameter.")
            
        # Make sure we have a user ID
        if not self.user_id:
            logger.warning("No user ID provided for document association. This may cause issues with row-level security.")
        
        try:
            # Verify project exists and user has access
            try:
                project = self.db.fetch_project_by_id(project_id)
                if not project:
                    raise ValueError(f"Project with ID {project_id} not found or user doesn't have access")
                logger.info(f"Verified project exists: {project.get('name')}")
            except Exception as e:
                logger.error(f"Error verifying project: {e}")
                raise ValueError(f"Could not verify project access: {e}")
                
            # Extract metadata
            metadata = extraction_result.get("metadata", {})
            num_pages = metadata.get("num_pages", 0)
            
            # Add structured data to metadata
            metadata["structured_data"] = extraction_result.get("structured_data", {})
            
            # Add user_id to metadata for tracking
            if self.user_id:
                metadata["user_id"] = self.user_id
            
            # Compute file hash for versioning
            import hashlib
            text_content = extraction_result.get("text", "")
            if not text_content:
                logger.warning("No text content found in extraction result, using filename for hash")
                text_content = filename
                
            file_hash = hashlib.sha256(text_content.encode()).hexdigest()
            
            # Check if document already exists with this hash
            existing_docs = self.db.fetch_documents_by_hash(file_hash)
            existing_in_project = [doc for doc in existing_docs if doc.get("project_id") == project_id]
            
            if existing_in_project:
                logger.info(f"Document with hash {file_hash} already exists in project {project_id}. Skipping re-insertion.")
                return {
                    "success": True,
                    "document_info": {
                        "filename": filename,
                        "document_id": existing_in_project[0]["id"],
                        "project_id": project_id,
                        "user_id": self.user_id,
                        "pages": existing_in_project[0].get("num_pages"),
                        "entities_count": len(extraction_result.get("entities", [])),
                        "chunks_count": None,
                        "version": existing_in_project[0].get("version", 1),
                        "hash": file_hash,
                        "message": "Document already exists in this project. No new version created."
                    }
                }
            
            # Get latest version for this filename in this project
            latest_version = self.db.fetch_latest_version_for_filename(filename, project_id)
            new_version = (latest_version or 0) + 1
            
            # Insert document metadata with hash, version, and project_id
            doc_id = self.db.insert_document(
                filename=filename,
                num_pages=num_pages,
                metadata=metadata,
                project_id=project_id,
                version=new_version,
                file_hash=file_hash
            )
            
            if not doc_id:
                raise ValueError("Failed to insert document - no document ID returned")
            
            # Process text into chunks if we have text (using the text_processor module)
            chunks = []
            chunk_ids = []
            if "text" in extraction_result and len(extraction_result["text"]) > 0:
                # Simple chunking by paragraphs
                paragraphs = extraction_result["text"].split("\n\n")
                for i, para in enumerate(paragraphs):
                    if para.strip():  # Only save non-empty paragraphs
                        # Determine approximate page number
                        page_num = 1  # Default
                        if num_pages > 1:
                            page_num = min(1 + (i * num_pages // len(paragraphs)), num_pages)
                            
                        chunk_metadata = {
                            "source": filename,
                            "chunk_index": i,
                            "total_chunks": len(paragraphs)
                        }
                        
                        # Insert chunk
                        try:
                            chunk_id = self.db.insert_chunk(
                                document_id=doc_id,
                                chunk_text=para.strip(),
                                chunk_order=i,
                                page_number=page_num,
                                metadata=chunk_metadata
                            )
                            chunk_ids.append(chunk_id)
                            chunks.append(para.strip())
                        except Exception as chunk_error:
                            logger.error(f"Error inserting chunk {i}: {chunk_error}")
            
            # Insert entities
            entity_count = 0
            for entity in extraction_result.get("entities", []):
                try:
                    self.db.insert_entity(
                        document_id=doc_id,
                        chunk_id=None,  # We're not associating with specific chunks for now
                        entity_type=entity.get("type", "unknown"),
                        entity_value=entity.get("value", ""),
                        metadata={"source_doc": filename, "page": entity.get("page", 0)}
                    )
                    entity_count += 1
                except Exception as entity_error:
                    logger.error(f"Error inserting entity: {entity_error}")
            
            # Insert relationships
            relationship_count = 0
            for rel in extraction_result.get("relationships", []):
                try:
                    self.db.insert_relationship(
                        document_id=doc_id,
                        chunk_id=None,  # We're not associating with specific chunks for now
                        entity_1=rel.get("entity_1", ""),
                        entity_2=rel.get("entity_2", ""),
                        relationship_type=rel.get("type", "unknown"),
                        context=rel.get("context", ""),
                        metadata={"source_doc": filename}
                    )
                    relationship_count += 1
                except Exception as rel_error:
                    logger.error(f"Error inserting relationship: {rel_error}")
            
            logger.info(f"Saved document to Supabase: {doc_id} with {entity_count} entities, {relationship_count} relationships, and {len(chunks)} chunks")
            
            return {
                "success": True,
                "document_info": {
                    "filename": filename,
                    "document_id": doc_id,
                    "project_id": project_id,
                    "user_id": self.user_id,
                    "pages": num_pages,
                    "entities_count": entity_count,
                    "relationships_count": relationship_count,
                    "chunks_count": len(chunks),
                    "version": new_version,
                    "hash": file_hash
                }
            }
            
        except Exception as e:
            logger.error(f"Error saving to Supabase: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def _log_api_usage(self, api_provider: str, api_type: str, tokens_used: int, response, document_id: Optional[str] = None, page_number: Optional[int] = None) -> None:
        """Log API usage to the database."""
        if not self.db:
            return
            
        try:
            # Extract token information from OCR response (Mistral 1.8.0)
            input_tokens = None
            output_tokens = None
            total_tokens = tokens_used
            
            if hasattr(response, 'usage_info') and response.usage_info:
                # OCRUsageInfo structure may have different fields than chat usage
                usage_info = response.usage_info
                input_tokens = getattr(usage_info, 'input_tokens', None)
                output_tokens = getattr(usage_info, 'output_tokens', None)
                total_tokens = getattr(usage_info, 'total_tokens', None) or tokens_used
                
            # Calculate cost using the existing method
            if api_provider == "mistral":
                cost_usd = self._calculate_mistral_ocr_cost(self.ocr_model, total_tokens, input_tokens, output_tokens)
            else:
                cost_usd = 0.0
            
            # Prepare request payload
            request_payload = {
                "model": self.ocr_model,
                "language": getattr(self, '_current_language', 'auto'),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
            
            # Add page number if provided
            if page_number is not None:
                request_payload["page"] = page_number
                
            # Prepare response metadata for OCR API (Mistral 1.8.0)
            extracted_chars = 0
            pages_processed = 1
            
            if hasattr(response, 'pages') and response.pages:
                # Sum up all markdown content length from all pages
                extracted_chars = sum(len(page.markdown) for page in response.pages if hasattr(page, 'markdown'))
            else:
                extracted_chars = len(str(response))
                
            if hasattr(response, 'usage_info') and response.usage_info:
                pages_processed = getattr(response.usage_info, 'pages_processed', 1)
                
            response_metadata = {
                "extracted_chars": extracted_chars,
                "mistral_response_id": getattr(response, 'id', None),
                "pages_processed": pages_processed,
                "model_used": getattr(response, 'model', self.ocr_model)
            }
            
            # Add page number to response metadata if provided
            if page_number is not None:
                response_metadata["page_number"] = page_number
            
            # Log to database
            self.db.log_api_usage(
                user_id=self.user_id,
                document_id=document_id,
                api_provider=api_provider,
                api_type=api_type,
                tokens_used=total_tokens,
                cost_usd=cost_usd,
                request_payload=request_payload,
                response_metadata=response_metadata
            )
            logger.info(f"Logged API usage: {api_provider} {api_type}, {total_tokens} tokens, ${cost_usd:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to log API usage: {e}")

    def _calculate_mistral_ocr_cost(self, model: str, tokens_used: int, input_tokens: int = None, output_tokens: int = None) -> float:
        """
        Calculate Mistral OCR cost based on model and token usage with current 2024 pricing.
        
        Args:
            model: The Mistral model name
            tokens_used: Total tokens used
            input_tokens: Input tokens (if available)
            output_tokens: Output tokens (if available)
            
        Returns:
            Cost in USD
        """
        if not tokens_used:
            return 0.0
        
        # Current Mistral pricing (December 2024) - prices per 1M tokens
        pricing = {
            # OCR models (dedicated OCR API)
            "mistral-ocr-2505": 0.15,             # $0.15 per 1M tokens (official OCR model)
            "mistral-ocr-2505-completion": 0.15,  # Alternative name
            "mistral-ocr-latest": 0.15,           # Legacy name fallback
            
            # Vision models (for OCR via chat API)
            "pixtral-12b": 0.40,       # $0.40 per 1M tokens
            "pixtral-12b-2409": 0.40,  # $0.40 per 1M tokens
            "pixtral-large": 1.20,     # $1.20 per 1M tokens (estimated)
            
            # Chat models (if used for OCR)
            "mistral-tiny": 0.25,      # $0.25 per 1M tokens
            "mistral-small": 2.00,     # $2.00 per 1M tokens
            "mistral-medium": 6.00,    # $6.00 per 1M tokens
            "mistral-large": 24.00,    # $24.00 per 1M tokens
        }
        
        # Normalize model name
        model_lower = model.lower().strip()
        
        # Find matching pricing
        cost_per_million = None
        for key, price in pricing.items():
            if key in model_lower or model_lower.startswith(key):
                cost_per_million = price
                break
        
        # Calculate cost
        if cost_per_million is not None:
            cost = (tokens_used / 1_000_000) * cost_per_million
            logger.debug(f"Mistral OCR cost calculation: {tokens_used} tokens × ${cost_per_million}/1M = ${cost:.6f}")
            return cost
        
        # Fallback pricing for unknown models
        fallback_rate = 1.00  # $1.00 per 1M tokens
        cost = (tokens_used / 1_000_000) * fallback_rate
        logger.warning(f"Unknown Mistral OCR model '{model}', using fallback rate ${fallback_rate}/1M tokens = ${cost:.6f}")
        return cost


if __name__ == "__main__":
    # Example usage
    extractor = PDFExtractor()
    results = extractor.extract_from_pdf(r"C:\Users\Yas\Desktop\Legal AI\Roquette_PSPE_Y078_PEARLITOL CR H - EXP_000000202191_EN.pdf")
    print(f"Extracted {len(results['text'])} characters of text")
    print(f"Found {len(results['tables'])} tables")
    print(f"Found {len(results.get('entities', []))} entities")
    print(f"Metadata: {results['metadata']}")
