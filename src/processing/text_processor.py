"""
Text Processing and Embedding Module for Agricultural AI.

This module handles text cleaning, chunking, entity extraction, and embedding generation
for agricultural documents including research papers, technical specifications, and product data.
"""

import os
import re
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mistralai import Mistral
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log

# Configure logging first before any usage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import the enhanced normalizer after logger is configured
try:
    from .enhanced_entity_normalizer import EnhancedEntityNormalizer
    NORMALIZER_AVAILABLE = True
    logger.info("Enhanced entity normalizer import successful")
except ImportError as e:
    logger.warning(f"Enhanced entity normalizer not available: {e}")
    NORMALIZER_AVAILABLE = False

# Import the entity quality filter
try:
    from .entity_quality_filter import EntityQualityFilter
    QUALITY_FILTER_AVAILABLE = True
    logger.info("Entity quality filter import successful")
except ImportError as e:
    logger.warning(f"Entity quality filter not available: {e}")
    QUALITY_FILTER_AVAILABLE = False

# Load environment variables
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")
RELATIONSHIP_MODEL = os.getenv("RELATIONSHIP_MODEL", "gpt-4.1-nano")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))

# Load agricultural entity configuration
def load_agricultural_entities():
    """Load agricultural entity configuration from config file."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "agro_entities.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load agricultural entities config: {e}")
    
    # Fallback to basic agricultural entities
    return {
        "entity_types": {
            "CROP": {"examples": ["corn", "wheat", "tomato", "soybean"]},
            "PRODUCT": {"examples": ["NPK 20-20-20", "GLYCOLYS", "fertilizer"]},
            "NUTRIENT": {"examples": ["nitrogen", "phosphorus", "potassium"]},
            "ORGANISM": {"examples": ["Bacillus subtilis", "mycorrhizae"]},
            "CONDITION": {"examples": ["pH 6.5", "sandy loam", "drought"]},
            "METRIC": {"examples": ["yield increase", "kg/ha"]},
            "METHOD": {"examples": ["foliar application", "soil incorporation"]},
            "TIMING": {"examples": ["flowering", "pre-emergence"]}
        },
        "relationship_types": {
            "increases_yield": {"description": "Product increases crop yield"},
            "applied_to": {"description": "Product applied to crop"},
            "effective_against": {"description": "Product effective against pest/disease"},
            "compatible_with": {"description": "Products are compatible"}
        }
    }

AGRO_ENTITIES = load_agricultural_entities()

def is_retryable_mistral_exception(exception):
    import requests
    retryable = (
        ConnectionError,
        TimeoutError,
        requests.exceptions.RequestException,
        Exception,  # Optionally catch all for now
    )
    return isinstance(exception, retryable)

def is_retryable_openai_exception(exception):
    import openai
    import requests
    retryable = (
        ConnectionError,
        TimeoutError,
        requests.exceptions.RequestException,
        openai.OpenAIError,
        Exception,  # Optionally catch all for now
    )
    return isinstance(exception, retryable)

RETRY_MISTRAL_CONFIG = dict(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(4),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
RETRY_OPENAI_CONFIG = dict(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(4),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

class TextProcessor:
    """Process and embed text from agricultural documents."""

    def __init__(self, api_key: Optional[str] = None, db=None, user_id=None):
        """
        Initialize the TextProcessor with API credentials.

        Args:
            api_key: Mistral API key (defaults to environment variable)
            db: Database instance for logging API usage (optional)
            user_id: User ID for logging (optional)
        """
        self.api_key = api_key or MISTRAL_API_KEY
        if not self.api_key:
            raise ValueError("Mistral API key is required for text processing and embedding")
        self.client = Mistral(api_key=self.api_key)
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # OpenAI client for NER/relationship extraction
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI-based NER/relationship extraction")
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.relationship_model = RELATIONSHIP_MODEL
        self.db = db
        self.user_id = user_id
        self.agro_entities = AGRO_ENTITIES
        
        # Initialize enhanced entity normalizer
        if NORMALIZER_AVAILABLE:
            try:
                self.entity_normalizer = EnhancedEntityNormalizer()
                logger.info("Enhanced entity normalizer initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize entity normalizer: {e}")
                self.entity_normalizer = None
        else:
            self.entity_normalizer = None
        
        # Initialize entity quality filter
        if QUALITY_FILTER_AVAILABLE:
            try:
                # Configure quality filter for agricultural/technical documents
                quality_config = {
                    'min_confidence': 0.7,
                    'min_semantic_score': 0.6,
                    'min_final_score': 0.65,
                    'max_entity_words': 6,
                    'min_entity_length': 2,
                    'preserve_compounds': True,
                    'language_detection': True,
                    'strict_mode': False
                }
                self.quality_filter = EntityQualityFilter(quality_config)
                logger.info("Entity quality filter initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize quality filter: {e}")
                self.quality_filter = None
        else:
            self.quality_filter = None
        
        if os.getenv("EMBEDDING_MODEL") != EMBEDDING_MODEL:
            logger.warning(f"Embedding model forced to '{EMBEDDING_MODEL}'. Please update your .env if needed.")
        if os.getenv("RELATIONSHIP_MODEL") != RELATIONSHIP_MODEL:
            logger.warning(f"Relationship extraction model forced to '{RELATIONSHIP_MODEL}'. Please update your .env if needed.")

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing noise and normalizing.

        Args:
            text: Raw text extracted from the PDF

        Returns:
            Cleaned text
        """
        # Remove page headers/footers (often contains page numbers, doc IDs, etc.)
        # This is a simplified example; real implementation would be more sophisticated
        lines = text.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            # Skip short lines that are likely headers/footers
            if len(line.strip()) < 5:
                continue
                
            # Skip lines that match common header/footer patterns
            if re.match(r'^Page \d+ of \d+$', line.strip()):
                continue
            
            # Skip lines with just numbers (likely page numbers)
            if re.match(r'^\d+$', line.strip()):
                continue
            
            # Remove excessive whitespace
            line = re.sub(r'\s+', ' ', line).strip()
            if line:
                cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove other common noise patterns
        cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned_text)  # Replace single newlines with spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with single space
        
        return cleaned_text.strip()

    def chunk_text(self, text: str, extracted_metadata: Dict[str, Any] = None) -> List[str]:
        """
        Split text into smaller, semantically meaningful chunks for agricultural documents.
        Returns list of chunk text strings (maintains backward compatibility).
        """
        # Get structured chunks with metadata
        chunks_with_metadata = self._chunk_text_with_metadata(text, extracted_metadata)
        
        # Return just the text for backward compatibility
        return [chunk["text"] for chunk in chunks_with_metadata]

    def _chunk_text_with_metadata(self, text: str, extracted_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into smaller, semantically meaningful chunks for agricultural documents.
        Each chunk is a dictionary with 'text' and 'metadata'.
        """
        chunks_with_metadata = []
        if extracted_metadata is None:
            extracted_metadata = {}

        # Get structured elements from pdf_extractor.py
        structured_data = extracted_metadata.get("structured_data", {})
        sections = structured_data.get("sections", [])
        tables = extracted_metadata.get("tables", [])
        technical_specs = structured_data.get("technical_specs", [])
        
        logger.info(f"Processing text with {len(sections)} sections, {len(tables)} tables, {len(technical_specs)} technical specs")

        # 1. Process Tables as Individual Chunks
        for i, table_data in enumerate(tables):
            table_content = table_data.get("json_content")
            if table_content:
                # Create structured table representation
                table_text = self._format_table_as_text(table_content)
                chunks_with_metadata.append({
                    "text": table_text,
                    "metadata": {
                        "type": "table",
                        "table_type": table_data.get("table_type", "general"),
                        "page": table_data.get("page_number", 1),
                        "table_id": table_data.get("id", f"table_{i}"),
                        "semantic_content": "structured_data"
                    }
                })
            elif table_data.get("markdown_content"):
                # Handle markdown tables
                md_table = table_data.get("markdown_content")
                table_text = self._format_markdown_table_as_text(md_table)
                chunks_with_metadata.append({
                    "text": table_text,
                    "metadata": {
                        "type": "table",
                        "table_type": "markdown_extracted",
                        "page": table_data.get("page_number", 1),
                        "table_id": table_data.get("id", f"md_table_{i}"),
                        "semantic_content": "structured_data"
                    }
                })

        # 2. Process Sections as Semantic Chunks
        processed_sections = []
        for section in sections:
            section_content = section.get("content", "")
            section_type = section.get("section_type", "general")
            
            # For large sections, split them further but respect semantic boundaries
            if len(section_content) > 2000:  # Large section threshold
                # Split by paragraphs first, then by sentences if needed
                paragraphs = section_content.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    if len(current_chunk + para) < 1500:  # Chunk size limit
                        current_chunk += para + "\n\n"
                    else:
                        # Save current chunk
                        if current_chunk.strip():
                            chunks_with_metadata.append({
                                "text": current_chunk.strip(),
                                "metadata": {
                                    "type": "section",
                                    "section_type": section_type,
                                    "section_title": section.get("title", ""),
                                    "page": section.get("page", 1),
                                    "semantic_content": "section_content",
                                    "chunk_part": len([c for c in chunks_with_metadata if c["metadata"].get("section_title") == section.get("title", "")])
                                }
                            })
                        current_chunk = para + "\n\n"
                
                # Add remaining content
                if current_chunk.strip():
                    chunks_with_metadata.append({
                        "text": current_chunk.strip(),
                        "metadata": {
                            "type": "section",
                            "section_type": section_type,
                            "section_title": section.get("title", ""),
                            "page": section.get("page", 1),
                            "semantic_content": "section_content",
                            "chunk_part": len([c for c in chunks_with_metadata if c["metadata"].get("section_title") == section.get("title", "")])
                        }
                    })
            else:
                # Small section - keep as single chunk
                chunks_with_metadata.append({
                    "text": section_content,
                    "metadata": {
                        "type": "section",
                        "section_type": section_type,
                        "section_title": section.get("title", ""),
                        "page": section.get("page", 1),
                        "semantic_content": "section_content"
                    }
                })
            
            processed_sections.append(section.get("title", ""))

        # 3. Process Technical Specifications as Focused Chunks
        if technical_specs:
            # Group related technical specs
            spec_groups = self._group_technical_specifications(technical_specs)
            for group_name, specs in spec_groups.items():
                spec_text = self._format_specifications_as_text(specs)
                chunks_with_metadata.append({
                    "text": spec_text,
                    "metadata": {
                        "type": "technical_specifications",
                        "spec_category": group_name,
                        "page": specs[0].get("page", 1) if specs else 1,
                        "spec_count": len(specs),
                        "semantic_content": "technical_parameters"
                    }
                })

        # 4. Fallback: Process Remaining Text with Traditional Chunking
        if not chunks_with_metadata:  # If no structured content was processed
            logger.info("No structured content found, using traditional text splitting")
            raw_chunks = self.text_splitter.split_text(text)
            for i, chunk_text in enumerate(raw_chunks):
                chunks_with_metadata.append({
                    "text": chunk_text,
                    "metadata": {
                        "type": "text_block",
                        "page": extracted_metadata.get("metadata", {}).get("num_pages", 1) // 2,
                        "chunk_index": i,
                        "semantic_content": "unstructured_text"
                    }
                })
        
        # 5. Ensure we have meaningful chunks
        filtered_chunks = []
        for chunk in chunks_with_metadata:
            # Filter out very short or empty chunks
            if len(chunk["text"].strip()) > 50:  # Minimum chunk size
                filtered_chunks.append(chunk)
        
        logger.info(f"Created {len(filtered_chunks)} semantic chunks from {len(chunks_with_metadata)} initial chunks")
        return filtered_chunks
    
    def _chunk_text_entity_aware(self, text: str, entities: List[Dict[str, Any]]) -> List[str]:
        """
        Create entity-aware chunks that preserve compound entities and technical specifications.
        
        Args:
            text: Text to chunk
            entities: List of extracted entities with positional information
            
        Returns:
            List of entity-aware text chunks
        """
        if not entities:
            return self.text_splitter.split_text(text)
        
        # Create entity boundaries to avoid splitting compound entities
        entity_boundaries = []
        for entity in entities:
            # Check if entity has positional information
            start_pos = entity.get('start_char', entity.get('start_pos'))
            end_pos = entity.get('end_char', entity.get('end_pos'))
            
            if start_pos is not None and end_pos is not None:
                entity_boundaries.append({
                    'start': start_pos,
                    'end': end_pos,
                    'entity': entity,
                    'is_compound': entity.get('metadata', {}).get('is_compound', False)
                })
        
        # Sort boundaries by start position
        entity_boundaries.sort(key=lambda x: x['start'])
        
        # Create chunks while respecting entity boundaries
        chunks = []
        current_pos = 0
        chunk_start = 0
        
        while current_pos < len(text):
            # Find the next safe split point (chunk size limit)
            target_end = min(current_pos + CHUNK_SIZE, len(text))
            
            # Look for entity boundaries near the target end
            safe_split_point = target_end
            
            # Check if we would split any compound entities
            for boundary in entity_boundaries:
                if boundary['start'] <= target_end <= boundary['end']:
                    # We would split an entity
                    if boundary['is_compound']:
                        # For compound entities, move split point to avoid breaking them
                        if target_end - boundary['start'] < boundary['end'] - target_end:
                            # Closer to start, split before entity
                            safe_split_point = boundary['start']
                        else:
                            # Closer to end, split after entity
                            safe_split_point = boundary['end']
                    break
            
            # Ensure we don't create too small chunks
            if safe_split_point - chunk_start < CHUNK_SIZE // 2 and safe_split_point < len(text):
                # Look for next natural break after the entity
                next_break = text.find('. ', safe_split_point)
                if next_break != -1 and next_break - chunk_start < CHUNK_SIZE * 1.5:
                    safe_split_point = next_break + 2
                else:
                    safe_split_point = target_end
            
            # Create chunk
            chunk_text = text[chunk_start:safe_split_point].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            # Handle overlap
            if safe_split_point < len(text):
                # Find overlap start point
                overlap_start = max(chunk_start, safe_split_point - CHUNK_OVERLAP)
                chunk_start = overlap_start
                current_pos = safe_split_point
            else:
                break
        
        logger.info(f"Created {len(chunks)} entity-aware chunks (preserved {len([b for b in entity_boundaries if b['is_compound']])} compound entities)")
        return chunks

    def _format_table_as_text(self, table_json: Dict) -> str:
        """Format JSON table as structured text."""
        headers = table_json.get("headers", [])
        rows = table_json.get("rows", [])
        table_type = table_json.get("table_type", "Table")
        caption = table_json.get("caption", "")
        
        text_parts = []
        if caption:
            text_parts.append(f"Table: {caption}")
        else:
            text_parts.append(f"{table_type.replace('_', ' ').title()}")
        
        if headers and rows:
            # Create a structured text representation
            text_parts.append("\nHeaders: " + " | ".join(headers))
            text_parts.append("\nData:")
            for i, row in enumerate(rows):
                if len(row) == len(headers):
                    row_text = []
                    for j, (header, value) in enumerate(zip(headers, row)):
                        row_text.append(f"{header}: {value}")
                    text_parts.append(f"Row {i+1}: " + "; ".join(row_text))
                else:
                    text_parts.append(f"Row {i+1}: " + " | ".join(str(cell) for cell in row))
        
        return "\n".join(text_parts)

    def _format_markdown_table_as_text(self, md_table: Dict) -> str:
        """Format markdown table as structured text."""
        headers = md_table.get("headers", [])
        rows = md_table.get("rows", [])
        
        text_parts = ["Table (from markdown)"]
        if headers:
            text_parts.append("Headers: " + " | ".join(headers))
        if rows:
            text_parts.append("Data:")
            for i, row in enumerate(rows):
                text_parts.append(f"Row {i+1}: " + " | ".join(str(cell) for cell in row))
        
        return "\n".join(text_parts)

    def _group_technical_specifications(self, specs: List[Dict]) -> Dict[str, List[Dict]]:
        """Group technical specifications by category."""
        groups = {
            "physical_properties": [],
            "chemical_properties": [],
            "composition": [],
            "quality_parameters": [],
            "storage_conditions": [],
            "other": []
        }
        
        for spec in specs:
            parameter = spec.get("parameter", "").lower()
            
            if any(term in parameter for term in ["ph", "density", "viscosity", "melting", "boiling", "appearance", "color", "odor"]):
                groups["physical_properties"].append(spec)
            elif any(term in parameter for term in ["purity", "assay", "concentration", "content", "active ingredient"]):
                groups["composition"].append(spec)
            elif any(term in parameter for term in ["solubility", "stability", "reactivity", "decomposition"]):
                groups["chemical_properties"].append(spec)
            elif any(term in parameter for term in ["storage", "temperature", "humidity", "shelf life"]):
                groups["storage_conditions"].append(spec)
            elif any(term in parameter for term in ["limit", "maximum", "minimum", "specification", "test"]):
                groups["quality_parameters"].append(spec)
            else:
                groups["other"].append(spec)
        
        # Return only non-empty groups
        return {k: v for k, v in groups.items() if v}

    def _format_specifications_as_text(self, specs: List[Dict]) -> str:
        """Format technical specifications as structured text."""
        text_parts = []
        for spec in specs:
            parameter = spec.get("parameter", "")
            value = spec.get("value", "")
            unit = spec.get("unit", "")
            
            if unit:
                text_parts.append(f"{parameter}: {value} {unit}")
            else:
                text_parts.append(f"{parameter}: {value}")
        
        return "\n".join(text_parts)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract agricultural entities and relationships from text using rule-based approach.

        Args:
            text: Text to extract entities from

        Returns:
            Dictionary of entity types and values
        """
        entities = {
            "crops": [],
            "products": [],
            "nutrients": [],
            "organisms": [],
            "conditions": [],
            "metrics": [],
            "methods": [],
            "timing": []
        }
        
        # Extract crops (enhanced patterns for agricultural domain)
        crop_patterns = [
            r"(?:crop|cultivar|variety|cv\.)\s+([A-Z][a-z]+(?:\s+[a-z]+)*)",
            r"\b(corn|maize|wheat|soybean|tomato|rice|cotton|barley|oats)\b",
            r"\b([A-Z][a-z]+\s+mays|[A-Z][a-z]+\s+lycopersicum)\b"  # Scientific names
        ]
        for pattern in crop_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                crop = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                if crop not in entities["crops"]:
                    entities["crops"].append(crop)
        
        # Extract agricultural products
        product_patterns = [
            r"NPK\s+\d+-\d+-\d+",
            r"\b(?:fertilizer|fungicide|herbicide|insecticide|bio-stimulant)\b",
            r"\b[A-Z][A-Z0-9]*®?\b",  # Brand names
            r"Bacillus\s+\w+"
        ]
        for pattern in product_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match not in entities["products"]:
                    entities["products"].append(match)
        
        # Extract nutrients
        nutrient_patterns = [
            r"\b(?:nitrogen|phosphorus|potassium|calcium|magnesium|iron|zinc|manganese)\b",
            r"\b(?:N|P|K|Ca|Mg|Fe|Zn|Mn)\b(?:\s*content|\s*level|\s*deficiency)?",
            r"(?:available|total)\s+(?:nitrogen|phosphorus|potassium)"
        ]
        for pattern in nutrient_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match not in entities["nutrients"]:
                    entities["nutrients"].append(match)
        
        # Extract yield metrics and measurements
        metric_patterns = [
            r"\d+(?:\.\d+)?\s*%\s*(?:yield\s*)?increase",
            r"\d+(?:\.\d+)?\s*(?:kg|t|bu)\/ha",
            r"\d+(?:\.\d+)?\s*(?:kg|tonnes|bushels)\s*per\s*hectare",
            r"p\s*[<>=]\s*0\.\d+"
        ]
        for pattern in metric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["metrics"].extend(matches)
        
        # Extract application methods
        method_patterns = [
            r"(?:foliar|soil|seed)\s+(?:application|treatment|incorporation)",
            r"fertigation|broadcasting|banding"
        ]
        for pattern in method_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["methods"].extend(matches)
        
        return entities

    @retry(**RETRY_OPENAI_CONFIG)
    def extract_entities_and_relationships_openai(self, text: str, document_id: Optional[str] = None, document_type: str = "general") -> Dict[str, Any]:
        """
        Use OpenAI to extract entities and relationships from text based on document type.
        
        Args:
            text: The text to extract entities and relationships from
            document_id: Optional document ID for logging
            document_type: Type of document to customize extraction ("general", "legal", "technical", "chemical", "agricultural")
            
        Returns:
            A dict with 'entities' and 'relationships' keys.
        """
        # Select the appropriate prompt based on document type
        if document_type.lower() in ["agricultural", "agro", "agriculture", "research", "field_trial", "efficacy"]:
            # Agricultural document prompt using agro_entities.json structure
            entity_types = self.agro_entities.get("entity_types", {})
            relationship_types = self.agro_entities.get("relationship_types", {})
            
            # Build entity type descriptions
            entity_descriptions = []
            for entity_type, config in entity_types.items():
                examples = ", ".join(config.get("examples", [])[:3])  # Show first 3 examples
                entity_descriptions.append(f"- {entity_type}: {config.get('description', '')} (e.g., {examples})")
            
            # Build relationship type descriptions
            relationship_descriptions = []
            for rel_type, config in relationship_types.items():
                relationship_descriptions.append(f"- {rel_type}: {config.get('description', '')}")
            
            prompt = f"""
                You are a specialized agricultural document analysis assistant. Extract all agricultural entities and relationships from the following text, focusing on:
                
                **Entity Types to Extract:**
{chr(10).join(entity_descriptions)}
                
                **Relationship Types to Identify:**
{chr(10).join(relationship_descriptions)}
                
                **Special Instructions:**
                1. For CROP entities: Include both common names and scientific names (e.g., "corn" and "Zea mays")
                2. For PRODUCT entities: Extract fertilizers, bio-stimulants, pesticides with specific formulations (e.g., "NPK 20-20-20")
                3. For NUTRIENT entities: Include both element names and chemical symbols (e.g., "nitrogen" and "N")
                4. For ORGANISM entities: Include beneficial microorganisms with strain identifications when available
                5. For CONDITION entities: Extract environmental parameters like pH, temperature, soil types
                6. For METRIC entities: Capture yield data, statistical significance, application rates with units
                7. For METHOD entities: Extract application techniques and farming practices
                8. For TIMING entities: Include growth stages, application timing, and seasonal references
                
                **Validation Patterns:**
                - Yield metrics: Look for percentage increases, statistical significance (p-values)
                - NPK ratios: Format like "NPK 20-20-20" or "N-P-K 15-15-15"
                - pH values: Format like "pH 6.5" or "soil pH of 7.2"
                - Application rates: Look for units like "kg/ha", "L/ha", "g/m²"
                - Scientific names: Italicized or binomial nomenclature
                
                Return your answer as a JSON object with two keys: 'entities' (a list of objects with 'type', 'value', and 'context' if possible) and 'relationships' (a list of objects with 'type', 'entity_1', 'entity_2', and 'context').
                
                Example output structure:
                {{
                  "entities": [
                    {{"type": "CROP", "value": "corn", "context": "field trial on corn yield"}},
                    {{"type": "PRODUCT", "value": "NPK 20-20-20", "context": "fertilizer application"}},
                    {{"type": "METRIC", "value": "15% yield increase", "context": "statistically significant improvement"}}
                  ],
                  "relationships": [
                    {{"type": "increases_yield", "entity_1": "NPK 20-20-20", "entity_2": "corn", "context": "fertilizer application resulted in 15% yield increase"}},
                    {{"type": "applied_to", "entity_1": "NPK 20-20-20", "entity_2": "corn", "context": "fertilizer was applied to corn plots"}}
                  ]
                }}
                
                Text:
                """ + text
        elif document_type.lower() in ["chemical", "sds", "safety_data_sheet"]:
            prompt = (
                """
                You are a specialized chemical and technical document analysis assistant. Extract all named entities and relationships from the following text, paying special attention to:
                
                1. Chemical compounds and ingredients (with CAS numbers if present)
                2. Technical specifications and parameters with their units
                3. Safety classifications and hazard information
                4. Manufacturing processes and conditions
                5. Quality control parameters
                6. Regulatory information and compliance statements
                
                Return your answer as a JSON object with two keys: 'entities' (a list of objects with 'type', 'value', and 'context' if possible) and 'relationships' (a list of objects with 'type', 'entity_1', 'entity_2', and 'context').
                
                For chemical entities, identify their roles (active ingredient, excipient, etc.) and any associated CAS numbers.
                For parameters, include their units and acceptable ranges when specified.
                For safety information, note hazard classifications and their sources.
                
                Text:
                """
                + text
            )
        elif document_type.lower() in ["technical", "specification", "datasheet"]:
            prompt = (
                """
                You are a technical document analysis assistant. Extract all named entities and relationships from the following text, focusing on:
                
                1. Product names and identifiers
                2. Technical parameters and specifications with units
                3. Testing methods and standards
                4. Performance characteristics
                5. Application areas and usage scenarios
                6. Compatibility information
                
                Return your answer as a JSON object with two keys: 'entities' (a list of objects with 'type', 'value', and 'context' if possible) and 'relationships' (a list of objects with 'type', 'entity_1', 'entity_2', and 'context').
                
                For parameters, include the measured value, units, and acceptable ranges when specified.
                For standards, include the standard name, number, and issuing organization.
                For products, note their intended applications and key features.
                
                Text:
                """
                + text
            )
        else:
            # General/legal prompt (original)
            prompt = (
                """
                You are a legal document analysis assistant. Extract all named entities (parties, dates, clauses, amounts, locations, organizations, laws, etc.) and explicit relationships (e.g., who signed what, who is obligated to whom, what clause relates to which party) from the following legal text. 
                Return your answer as a JSON object with two keys: 'entities' (a list of objects with 'type', 'value', and 'span' if possible) and 'relationships' (a list of objects with 'type', 'entity_1', 'entity_2', and 'context').
                Only use information present in the text. Example:
                {
                  "entities": [
                    {"type": "party", "value": "Party A"},
                    {"type": "date", "value": "15 January 2025"},
                    {"type": "amount", "value": "$10,000"}
                  ],
                  "relationships": [
                    {"type": "signed", "entity_1": "Party A", "entity_2": "Agreement", "context": "Party A signed the Agreement on 15 January 2025."}
                  ]
                }
                Text:
                """
                + text
            )
            
        try:
            response = self.openai_client.chat.completions.create(
                model=self.relationship_model,
                messages=[
                    {"role": "system", "content": f"You are a specialized entity and relationship extraction assistant for {document_type} documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=4000
            )
            answer = response.choices[0].message.content
            # --- API Usage Logging ---
            if self.db:
                # Extract token counts for accurate cost calculation
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else None
                input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and response.usage else None
                output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else None
                
                # Calculate OpenAI cost with separate input/output pricing
                cost_usd = self._calculate_openai_cost(
                    self.relationship_model, 
                    tokens_used, 
                    input_tokens, 
                    output_tokens
                ) if tokens_used else None
                
                self.db.log_api_usage(
                    user_id=self.user_id,
                    document_id=document_id,
                    api_provider="openai",
                    api_type="ner",
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    request_payload={
                        "model": self.relationship_model, 
                        "prompt_len": len(prompt), 
                        "document_type": document_type,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens
                    },
                    response_metadata={
                        "openai_response_id": getattr(response, 'id', None),
                        "finish_reason": response.choices[0].finish_reason if response.choices else None
                    }
                )
            
            def clean_json_response(response_text: str) -> str:
                """Remove markdown code blocks and other formatting from OpenAI response."""
                if not response_text:
                    return response_text
                
                # Remove markdown code blocks (```json ... ```)
                response_text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
                response_text = re.sub(r'^```\s*$', '', response_text, flags=re.MULTILINE)
                response_text = re.sub(r'```$', '', response_text)
                
                # Remove any remaining markdown formatting
                response_text = response_text.strip()
                
                # Try to fix incomplete JSON
                response_text = fix_incomplete_json(response_text)
                
                return response_text
            
            def fix_incomplete_json(json_str: str) -> str:
                """Attempt to fix incomplete or truncated JSON responses."""
                try:
                    # First try to parse as-is
                    json.loads(json_str)
                    return json_str  # It's already valid
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed: {e}. Attempting to fix...")
                    
                    # Try to fix common issues
                    fixed_json = json_str.strip()
                    
                    # If the JSON is truncated in the middle of an object or array, try to close it properly
                    # Count open braces and brackets to see what needs closing
                    open_braces = fixed_json.count('{') - fixed_json.count('}')
                    open_brackets = fixed_json.count('[') - fixed_json.count(']')
                    
                    # Remove any trailing incomplete content that might be causing issues
                    # Look for patterns like incomplete strings or objects
                    lines = fixed_json.split('\n')
                    valid_lines = []
                    
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Skip lines that look incomplete (e.g., just '"type": "Incorporation Level",')
                        if (line.endswith(',') and 
                            (line.count('"') % 2 != 0 or  # Odd number of quotes
                             ':' in line and not ('{' in line or '[' in line or line.endswith('",') or line.endswith('",')))):
                            logger.warning(f"Skipping incomplete line: {line}")
                            break
                        
                        valid_lines.append(line)
                    
                    # Reconstruct from valid lines
                    fixed_json = '\n'.join(valid_lines)
                    
                    # Now try to close any unclosed structures
                    # Close any open objects
                    for _ in range(open_braces):
                        fixed_json += '\n}'
                    
                    # Close any open arrays  
                    for _ in range(open_brackets):
                        fixed_json += '\n]'
                    
                    # Remove trailing commas before closing braces/brackets
                    fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
                    
                    # Try to parse the fixed version
                    try:
                        json.loads(fixed_json)
                        logger.info("Successfully repaired truncated JSON")
                        return fixed_json
                    except json.JSONDecodeError as e2:
                        logger.warning(f"Could not repair JSON: {e2}")
                        # Return a minimal valid structure if all else fails
                        return '{"entities": [], "relationships": []}'
                
                except Exception:
                    # If anything else goes wrong, return minimal structure
                    return '{"entities": [], "relationships": []}'
            
            # Try to parse JSON from the cleaned answer
            try:
                cleaned_answer = clean_json_response(answer)
                result = json.loads(cleaned_answer)
                if "entities" not in result:
                    result["entities"] = []
                if "relationships" not in result:
                    result["relationships"] = []
                    
                logger.info(f"Successfully parsed OpenAI response: {len(result['entities'])} entities, {len(result['relationships'])} relationships")
                return result
            except Exception as e:
                logger.error(f"Failed to parse OpenAI NER/relationship JSON: {e}; raw answer: {answer}")
                logger.error(f"Cleaned answer was: {clean_json_response(answer)}")
                
                # Try a simpler prompt with fewer entities to avoid truncation
                logger.info("Attempting simplified entity extraction due to parsing failure...")
                try:
                    simple_prompt = f"""
                    Extract only the most important entities from this {document_type} document text. 
                    Return ONLY a valid JSON object with this exact structure:
                    {{
                        "entities": [
                            {{"type": "Product Name", "value": "product name", "context": "brief context"}},
                            {{"type": "Technical Parameter", "value": "parameter name", "context": "value"}}
                        ],
                        "relationships": []
                    }}
                    
                    Text to analyze (first 2000 characters):
                    {text[:2000]}
                    """
                    
                    retry_response = self.openai_client.chat.completions.create(
                        model=self.relationship_model,
                        messages=[
                            {"role": "system", "content": "You are an entity extraction assistant. Return only valid JSON."},
                            {"role": "user", "content": simple_prompt}
                        ],
                        temperature=0.0,
                        max_tokens=1500
                    )
                    
                    retry_answer = retry_response.choices[0].message.content
                    retry_cleaned = clean_json_response(retry_answer)
                    retry_result = json.loads(retry_cleaned)
                    
                    if "entities" not in retry_result:
                        retry_result["entities"] = []
                    if "relationships" not in retry_result:
                        retry_result["relationships"] = []
                    
                    logger.info(f"Simplified extraction successful: {len(retry_result['entities'])} entities")
                    return retry_result
                    
                except Exception as retry_e:
                    logger.error(f"Simplified extraction also failed: {retry_e}")
                
                # Try to extract partial data even if JSON is malformed
                partial_entities = self._extract_partial_entities_from_text(answer)
                if partial_entities:
                    logger.info(f"Extracted {len(partial_entities)} entities from partial JSON")
                    return {"entities": partial_entities, "relationships": []}
                
                return {"entities": [], "relationships": []}
        except Exception as e:
            logger.error(f"OpenAI NER/relationship extraction failed: {e}")
            return {"entities": [], "relationships": []}

    @retry(**RETRY_MISTRAL_CONFIG)
    def generate_embeddings(self, chunks: List[str], document_id: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for text chunks using Mistral's API.

        Args:
            chunks: List of text chunks to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        batch_size = EMBEDDING_BATCH_SIZE  # Now configurable
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            start_time = time.time()
            response = None
            try:
                response = self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    inputs=batch  # Changed from 'input' to 'inputs' as per Mistral API
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logger.info(f"Embedding batch {i//batch_size+1}: size={len(batch)}, time={time.time()-start_time:.2f}s")
                # --- API Usage Logging ---
                if self.db:
                    tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else None
                    
                    # Calculate Mistral cost with improved method
                    cost_usd = self._calculate_mistral_cost(EMBEDDING_MODEL, tokens_used) if tokens_used else None
                    
                    self.db.log_api_usage(
                        user_id=self.user_id,
                        document_id=document_id,
                        api_provider="mistral",
                        api_type="embedding",
                        tokens_used=tokens_used,
                        cost_usd=cost_usd,
                        request_payload={
                            "model": EMBEDDING_MODEL, 
                            "batch_size": len(batch),
                            "batch_number": i//batch_size+1,
                            "total_batches": (len(chunks) + batch_size - 1) // batch_size
                        },
                        response_metadata={
                            "embedding_count": len(batch_embeddings),
                            "embedding_dimension": len(batch_embeddings[0]) if batch_embeddings else None
                        }
                    )
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size+1}: {e}")
                if response and hasattr(response, 'data'):
                    fail_count = len(response.data)
                else:
                    fail_count = len(batch)
                embeddings.extend([[0.0] * 1024] * fail_count)  # Adjust dimension if needed
        return embeddings

    def process_document(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document through the entire pipeline.

        Args:
            extracted_data: Dictionary containing extracted text and metadata

        Returns:
            Dictionary with processed text, chunks, entities, and embeddings
        """
        # Clean the extracted text
        cleaned_text = self.clean_text(extracted_data["text"])
        
        # Determine document type for specialized processing
        document_type = self._detect_document_type(cleaned_text, extracted_data)
        logger.info(f"Detected document type: {document_type}")
        
        # Add the detected document type to metadata for future reference
        if "metadata" not in extracted_data:
            extracted_data["metadata"] = {}
        extracted_data["metadata"]["detected_document_type"] = document_type
        extracted_data["document_type"] = document_type  # Also add at the top level
        
        # Split text into chunks
        chunks = self.chunk_text(cleaned_text, extracted_data)
        
        # Use OpenAI for advanced NER/relationship extraction with document type
        ner_result = self.extract_entities_and_relationships_openai(
            cleaned_text, 
            extracted_data.get("document_id"),
            document_type
        )
        entities = ner_result.get("entities", [])
        relationships = ner_result.get("relationships", [])
        
        # Debug logging for entity extraction
        logger.info(f"OpenAI NER extraction completed for document type '{document_type}'")
        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
        if entities:
            logger.info(f"Sample entities: {entities[:3]}")
        if relationships:
            logger.info(f"Sample relationships: {relationships[:2]}")
        
        # Apply quality filtering to entities
        if self.quality_filter and entities:
            logger.info("Applying entity quality filtering...")
            original_count = len(entities)
            
            # Filter entities based on quality metrics
            filtered_entities, quality_metrics = self.quality_filter.filter_entities(
                entities, 
                document_type=document_type, 
                language="auto"
            )
            
            # Update entities with filtered results
            entities = filtered_entities
            
            # Generate quality statistics
            quality_stats = self.quality_filter.get_quality_statistics(quality_metrics)
            
            logger.info(f"Quality filtering completed: {original_count} → {len(entities)} entities")
            logger.info(f"Pass rate: {quality_stats.get('pass_rate', 0):.2%}")
            logger.info(f"Noise reduction: {quality_stats.get('quality_improvement', {}).get('noise_reduction', 0):.2%}")
            
            # Store quality metrics in metadata
            if "quality_metrics" not in extracted_data.get("metadata", {}):
                extracted_data.setdefault("metadata", {})["quality_metrics"] = {}
            extracted_data["metadata"]["quality_metrics"]["entity_filtering"] = quality_stats
        else:
            logger.info("Quality filtering skipped (filter not available or no entities)")
        
        # Generate embeddings for each chunk
        embeddings = self.generate_embeddings(chunks, extracted_data.get("document_id"))
        
        # Prepare results
        results = {
            "cleaned_text": cleaned_text,
            "chunks": chunks,
            "entities": entities,
            "relationships": relationships,
            "embeddings": embeddings,
            "metadata": extracted_data.get("metadata", {}),
            "source": extracted_data.get("source", "unknown"),
            "document_type": document_type
        }
        
        # Add additional processing for specific document types
        if document_type in ["agricultural", "agro", "agriculture", "research", "field_trial", "efficacy"]:
            # Extract agricultural-specific information
            crop_data = self._extract_agricultural_crops(cleaned_text, entities)
            efficacy_data = self._extract_efficacy_metrics(cleaned_text, entities)
            application_data = self._extract_application_methods(cleaned_text, entities)
            results["crop_data"] = crop_data
            results["efficacy_metrics"] = efficacy_data
            results["application_methods"] = application_data
        elif document_type in ["chemical", "sds", "safety_data_sheet"]:
            # Extract chemical-specific information
            chemical_compounds = self._extract_chemical_compounds(cleaned_text, entities)
            safety_info = self._extract_safety_information(cleaned_text)
            results["chemical_compounds"] = chemical_compounds
            results["safety_information"] = safety_info
        elif document_type in ["technical", "specification", "datasheet"]:
            # Extract technical parameters and specifications
            technical_params = self._extract_technical_parameters(cleaned_text, entities)
            results["technical_parameters"] = technical_params
        
        return results

    def _detect_document_type(self, text: str, extracted_data: Dict[str, Any]) -> str:
        """
        Determine the document type based on content and metadata.
        
        Args:
            text: The cleaned document text
            extracted_data: Original extraction data with metadata
            
        Returns:
            Document type string: "agricultural", "legal", "chemical", "technical", "general", etc.
        """
        text_lower = text.lower()
        filename = extracted_data.get("source", "").lower()
        
        # Check if document type is already in metadata
        if "document_type" in extracted_data:
            return extracted_data["document_type"]
        
        # Check for agricultural document indicators (HIGHEST PRIORITY)
        agricultural_keywords = [
            # Crops and agriculture
            "crop", "crops", "yield", "harvest", "cultivation", "agriculture", "agricultural",
            "farming", "field trial", "field study", "efficacy", "agronomic", "agronomy",
            # Specific crops
            "corn", "wheat", "tomato", "soybean", "rice", "cotton", "maize", "barley",
            "zea mays", "triticum", "solanum lycopersicum", "glycine max", "oryza sativa",
            # Products and treatments
            "fertilizer", "bio-stimulant", "biostimulant", "pesticide", "herbicide", "fungicide",
            "npk", "nitrogen", "phosphorus", "potassium", "nutrient", "nutrients",
            # Organisms and biology
            "bacillus subtilis", "rhizobium", "mycorrhizae", "trichoderma", "beneficial microorganisms",
            # Conditions and methods
            "soil ph", "field capacity", "application rate", "foliar application", "fertigation",
            "growth stage", "pre-emergence", "post-emergence", "flowering stage",
            # Research terminology
            "statistical significance", "p-value", "anova", "randomized block", "treatment effect",
            "yield increase", "percent increase", "kg/ha", "tonnes per hectare", "bushels per acre"
        ]
        
        # Count agricultural keywords
        agro_score = sum(1 for keyword in agricultural_keywords if keyword in text_lower)
        
        # Check for agricultural filename patterns
        agricultural_filename_patterns = [
            "agro", "agriculture", "crop", "yield", "fertilizer", "field_trial", 
            "efficacy", "biostimulant", "nutrient", "farming"
        ]
        
        filename_agro_score = sum(1 for pattern in agricultural_filename_patterns if pattern in filename)
        
        # Agricultural document threshold: 3+ keywords or specific filename patterns
        if agro_score >= 3 or filename_agro_score >= 1:
            return "agricultural"
            
        # Check for SDS/MSDS indicators
        if any(term in text_lower for term in ["safety data sheet", "sds", "msds", "hazard"]):
            return "safety_data_sheet"
            
        # Check for technical document indicators
        if any(term in text_lower for term in ["specification", "technical data", "parameters", "properties"]):
            return "technical"
            
        # Check for chemical product indicators
        if any(term in text_lower for term in ["cas no", "chemical", "ingredient", "formulation"]):
            return "chemical"
            
        # Check for legal document indicators
        if any(term in text_lower for term in ["agreement", "contract", "party", "clause", "legal", "law"]):
            return "legal"
            
        # Check filename patterns for non-agricultural documents
        if any(term in filename for term in ["sds", "msds", "safety"]):
            return "safety_data_sheet"
        if any(term in filename for term in ["spec", "technical", "datasheet"]):
            return "technical"
        if any(term in filename for term in ["contract", "agreement", "legal"]):
            return "legal"
            
        # Default type
        return "general"
        
    def _extract_chemical_compounds(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract chemical compounds and their properties from text."""
        compounds = []
        
        # First check existing entities for chemical compounds
        for entity in entities:
            if entity.get("type") in ["chemical_compound", "substance", "ingredient"]:
                compound = {
                    "name": entity.get("value", ""),
                    "type": entity.get("type", "chemical_compound"),
                    "context": entity.get("context", "")
                }
                
                # Look for CAS number
                try:
                    cas_match = re.search(r'CAS(?:\s+No)?\.?\:?\s*(\d{1,7}\-\d{2}\-\d{1})', text, re.IGNORECASE)
                    if cas_match and cas_match.group(1):
                        compound["cas_number"] = cas_match.group(1)
                except (IndexError, AttributeError):
                    # Skip if regex match fails
                    pass
                    
                compounds.append(compound)
                
        # Additional extraction with regex patterns for chemicals not caught by NER
        cas_patterns = [
            r'(\w+[\s\-]+\w+)(?:\s+|\()CAS(?:\s+No)?\.?\:?\s*(\d{1,7}\-\d{2}\-\d{1})',
            r'CAS(?:\s+No)?\.?\:?\s*(\d{1,7}\-\d{2}\-\d{1})(?:\s+|\()([^)]+)'
        ]
        
        for pattern in cas_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    if match.lastindex and match.lastindex >= 2:
                        # Safely get groups
                        group1 = match.group(1) if match.lastindex >= 1 else ""
                        group2 = match.group(2) if match.lastindex >= 2 else ""
                        
                        # Determine which is name and which is CAS number
                        if "CAS" not in group1:
                            name = group1
                            cas_num = group2
                        else:
                            name = group2
                            cas_num = group1
                        
                        # Check if this compound is already in our list
                        if not any(c.get("name", "").lower() == name.lower() for c in compounds):
                            compounds.append({
                                "name": name.strip(),
                                "cas_number": cas_num.strip(),
                                "type": "chemical_compound",
                                "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
                            })
                except (IndexError, AttributeError):
                    # Skip if regex match fails
                    continue
        
        return compounds
        
    def _extract_safety_information(self, text: str) -> Dict[str, Any]:
        """Extract safety information from SDS text."""
        safety_info = {
            "hazard_statements": [],
            "precautionary_statements": [],
            "storage_conditions": [],
            "handling_precautions": []
        }
        
        # Extract hazard statements (H-statements)
        hazard_pattern = r'(?:H\d{3})[^\n\.]+'
        for match in re.finditer(hazard_pattern, text, re.IGNORECASE):
            safety_info["hazard_statements"].append(match.group(0).strip())
            
        # Extract precautionary statements (P-statements)
        precaution_pattern = r'(?:P\d{3})[^\n\.]+'
        for match in re.finditer(precaution_pattern, text, re.IGNORECASE):
            safety_info["precautionary_statements"].append(match.group(0).strip())
            
        # Extract storage information
        storage_section = self._extract_section(text, ["storage", "storage conditions", "storage and handling"])
        if storage_section:
            # Split into bullet points or sentences
            for line in re.split(r'[•\n\.]', storage_section):
                if len(line.strip()) > 10:  # Ignore very short lines
                    safety_info["storage_conditions"].append(line.strip())
                    
        # Extract handling precautions
        handling_section = self._extract_section(text, ["handling", "handling precautions", "precautions"])
        if handling_section:
            for line in re.split(r'[•\n\.]', handling_section):
                if len(line.strip()) > 10:
                    safety_info["handling_precautions"].append(line.strip())
        
        return safety_info
        
    def _extract_technical_parameters(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract technical parameters and specifications."""
        parameters = {
            "physical_properties": [],
            "chemical_properties": [],
            "performance_parameters": [],
            "standards": []
        }
        
        # Extract from entities
        for entity in entities:
            if entity.get("type") in ["parameter", "specification", "property", "value"]:
                value = entity.get("value", "")
                
                # Determine parameter type based on context or name
                param_type = "physical_properties"  # Default
                context = entity.get("context", "").lower()
                
                if any(term in context for term in ["chemical", "composition", "purity", "ph"]):
                    param_type = "chemical_properties"
                elif any(term in context for term in ["performance", "efficiency", "output", "result"]):
                    param_type = "performance_parameters"
                elif any(term in context for term in ["standard", "test method", "astm", "iso"]):
                    param_type = "standards"
                
                parameters[param_type].append({
                    "name": entity.get("name", value),
                    "value": value,
                    "unit": entity.get("unit", self._extract_unit(value)),
                    "context": entity.get("context", "")
                })
        
        # Extract additional parameters with regex
        property_patterns = [
            # Physical properties
            r'(?:density|viscosity|melting point|boiling point|flash point|solubility)[^\n:]*[:]\s*([^\n]+)',
            # Chemical properties
            r'(?:ph value|purity|composition|concentration)[^\n:]*[:]\s*([^\n]+)',
            # Standards
            r'(?:iso|astm|en|din)\s+\d+[^\n\.]+'
        ]
        
        for pattern in property_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                full_text = match.group(0).strip()
                value = match.group(1).strip() if match.lastindex else full_text
                
                # Determine parameter type
                param_type = "physical_properties"  # Default
                if any(term in full_text.lower() for term in ["ph", "purity", "composition"]):
                    param_type = "chemical_properties"
                elif any(term in full_text.lower() for term in ["iso", "astm", "en", "din"]):
                    param_type = "standards"
                
                # Extract name (everything before the colon)
                name = full_text.split(':')[0].strip() if ':' in full_text else full_text
                
                # Add to appropriate parameter type
                parameters[param_type].append({
                    "name": name,
                    "value": value,
                    "unit": self._extract_unit(value),
                    "context": text[max(0, match.start()-30):min(len(text), match.end()+30)]
                })
        
        return parameters
    
    def _extract_section(self, text: str, section_names: List[str]) -> str:
        """
        Extract a section from text based on common section headers.
        
        Args:
            text: The document text
            section_names: List of possible section names to look for
            
        Returns:
            The extracted section text or empty string
        """
        for section_name in section_names:
            # Look for section headers (numbered, all caps, or with colons)
            patterns = [
                rf'\d+[\.\)]\s+{section_name}[^\n]*\n(.*?)(?:\n\d+[\.\)]|\Z)',  # Numbered sections
                rf'{section_name}[^\n]*?:([^\n]+(?:\n(?!\d+[\.\)]|\w+:)[^\n]+)*)',  # Section with colon
                rf'{section_name.upper()}[^\n]*\n(.*?)(?:\n[A-Z][A-Z\s]+:|\Z)'  # ALL CAPS sections
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    # Return the captured section content
                    return match.group(1).strip()
        
        return ""
    
    def _extract_unit(self, text: str) -> str:
        """Extract measurement unit from text if present."""
        unit_pattern = r'(\d+(?:\.\d+)?)\s*(mg|g|kg|ml|L|°C|pH|%|ppm|mPa\.s|cP|mm|cm|m|N/mm²|MPa|kPa)\b'
        match = re.search(unit_pattern, text)
        if match and match.lastindex >= 2:
            return match.group(2)
        return ""

    def _extract_partial_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract entity information from malformed JSON text using regex."""
        entities = []
        
        # Look for entity patterns in the text even if JSON is malformed
        # Pattern for entities like: {"type": "Product Name", "value": "PEARLITOL® CR H – EXP", "context": "..."}
        entity_pattern = r'\{\s*"type":\s*"([^"]+)"\s*,\s*"value":\s*"([^"]+)"(?:\s*,\s*"context":\s*"([^"]*)")?\s*(?:,\s*"[^"]*":\s*"[^"]*")*\s*\}'
        
        for match in re.finditer(entity_pattern, text, re.DOTALL):
            entity = {
                "type": match.group(1),
                "value": match.group(2),
                "context": match.group(3) if match.group(3) else ""
            }
            entities.append(entity)
        
        # Also try simpler patterns for key-value pairs
        simple_pattern = r'"type":\s*"([^"]+)"\s*,\s*"value":\s*"([^"]+)"'
        for match in re.finditer(simple_pattern, text):
            # Check if this entity is already captured by the more complete pattern
            entity_value = match.group(2)
            if not any(e.get("value") == entity_value for e in entities):
                entity = {
                    "type": match.group(1),
                    "value": entity_value,
                    "context": ""
                }
                entities.append(entity)
        
        return entities

    def _extract_agricultural_crops(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract agricultural crop information from text."""
        crop_data = {
            "crops_mentioned": [],
            "varieties": [],
            "growth_stages": [],
            "growing_conditions": []
        }
        
        # First check existing entities for crops
        for entity in entities:
            if entity.get("type") in ["CROP", "crop"]:
                crop_info = {
                    "name": entity.get("value", ""),
                    "scientific_name": self._normalize_crop_name(entity.get("value", "")),
                    "context": entity.get("context", ""),
                    "type": "extracted_entity"
                }
                crop_data["crops_mentioned"].append(crop_info)
        
        # Extract crop varieties and cultivars
        variety_patterns = [
            r'(?:variety|cultivar|cv\.)\s+([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-z]+)\s+cv\.\s+([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-z]+)\s+variety\s+([A-Z][a-zA-Z\s]+)'
        ]
        
        for pattern in variety_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if match.lastindex >= 2:
                    crop_name = match.group(1)
                    variety_name = match.group(2)
                else:
                    crop_name = "unknown"
                    variety_name = match.group(1)
                
                crop_data["varieties"].append({
                    "crop": crop_name.strip(),
                    "variety": variety_name.strip(),
                    "context": text[max(0, match.start()-30):min(len(text), match.end()+30)]
                })
        
        # Extract growth stages
        growth_stage_patterns = [
            r'(?:V\d+|R\d+)\s+stage',
            r'(?:vegetative|reproductive|flowering|grain filling|maturity)\s+stage',
            r'(?:pre-emergence|post-emergence|seedling|tillering|heading|harvest)'
        ]
        
        for pattern in growth_stage_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                crop_data["growth_stages"].append({
                    "stage": match.group(0).strip(),
                    "context": text[max(0, match.start()-30):min(len(text), match.end()+30)]
                })
        
        # Extract growing conditions
        condition_patterns = [
            r'soil\s+pH\s+(?:of\s+)?(\d+(?:\.\d+)?)',
            r'temperature\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*°?C',
            r'(?:sandy|clay|loam|silt)\s+soil',
            r'field\s+capacity\s+(?:of\s+)?(\d+(?:\.\d+)?)%?',
            r'moisture\s+content\s+(?:of\s+)?(\d+(?:\.\d+)?)%?'
        ]
        
        for pattern in condition_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                crop_data["growing_conditions"].append({
                    "condition": match.group(0).strip(),
                    "value": match.group(1).strip() if match.lastindex else "",
                    "context": text[max(0, match.start()-30):min(len(text), match.end()+30)]
                })
        
        return crop_data
    
    def _extract_efficacy_metrics(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract efficacy metrics and performance data from agricultural text."""
        efficacy_data = {
            "yield_improvements": [],
            "statistical_significance": [],
            "application_rates": [],
            "performance_metrics": []
        }
        
        # First check existing entities for metrics
        for entity in entities:
            if entity.get("type") in ["METRIC", "metric"]:
                value = entity.get("value", "")
                if "yield" in value.lower() and ("increase" in value.lower() or "improvement" in value.lower()):
                    efficacy_data["yield_improvements"].append({
                        "metric": value,
                        "context": entity.get("context", ""),
                        "type": "extracted_entity"
                    })
                elif any(term in value.lower() for term in ["p-value", "p<", "p>", "significance"]):
                    efficacy_data["statistical_significance"].append({
                        "metric": value,
                        "context": entity.get("context", ""),
                        "type": "extracted_entity"
                    })
        
        # Extract yield improvements
        yield_patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*(?:yield\s*)?(?:increase|improvement|gain)',
            r'yield\s+(?:increased|improved)\s+by\s+(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*(?:kg|t|bu)\/ha\s+(?:increase|improvement)',
            r'(\d+(?:\.\d+)?)\s+(?:fold|times)\s+(?:increase|improvement)'
        ]
        
        for pattern in yield_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                efficacy_data["yield_improvements"].append({
                    "metric": match.group(0).strip(),
                    "value": match.group(1),
                    "context": text[max(0, match.start()-50):min(len(text), match.end()+50)],
                    "type": "regex_extracted"
                })
        
        # Extract statistical significance
        stats_patterns = [
            r'p\s*[<>=]\s*0\.\d+',
            r'(?:statistically\s+)?significant\s+(?:at\s+)?(?:p\s*[<>=]\s*0\.\d+)?',
            r'confidence\s+interval\s+(?:of\s+)?(\d+)%',
            r'(?:ANOVA|t-test|chi-square)\s+(?:p\s*[<>=]\s*0\.\d+)?'
        ]
        
        for pattern in stats_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                efficacy_data["statistical_significance"].append({
                    "metric": match.group(0).strip(),
                    "context": text[max(0, match.start()-40):min(len(text), match.end()+40)],
                    "type": "regex_extracted"
                })
        
        # Extract application rates
        rate_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:kg|L|g|ml)\/ha',
            r'(?:applied\s+at\s+)?(\d+(?:\.\d+)?)\s*(?:kg|L|g|ml)\s+per\s+hectare',
            r'application\s+rate\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:kg|L|g|ml)\/ha'
        ]
        
        for pattern in rate_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                efficacy_data["application_rates"].append({
                    "rate": match.group(0).strip(),
                    "value": match.group(1),
                    "context": text[max(0, match.start()-40):min(len(text), match.end()+40)],
                    "type": "regex_extracted"
                })
        
        # Extract general performance metrics
        performance_patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s+(?:reduction|decrease)\s+in\s+([a-zA-Z\s]+)',
            r'(?:improved|increased|enhanced)\s+([a-zA-Z\s]+)\s+by\s+(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*(?:mg|g|kg)\s+per\s+(?:plant|m²|hectare)'
        ]
        
        for pattern in performance_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                efficacy_data["performance_metrics"].append({
                    "metric": match.group(0).strip(),
                    "context": text[max(0, match.start()-40):min(len(text), match.end()+40)],
                    "type": "regex_extracted"
                })
        
        return efficacy_data
    
    def _extract_application_methods(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract application methods and timing information from agricultural text."""
        application_data = {
            "application_methods": [],
            "timing": [],
            "equipment": [],
            "frequency": []
        }
        
        # First check existing entities
        for entity in entities:
            if entity.get("type") in ["METHOD", "method"]:
                application_data["application_methods"].append({
                    "method": entity.get("value", ""),
                    "context": entity.get("context", ""),
                    "type": "extracted_entity"
                })
            elif entity.get("type") in ["TIMING", "timing"]:
                application_data["timing"].append({
                    "timing": entity.get("value", ""),
                    "context": entity.get("context", ""),
                    "type": "extracted_entity"
                })
        
        # Extract application methods
        method_patterns = [
            r'(?:foliar|soil|seed|root)\s+(?:application|treatment|spray)',
            r'(?:broadcasting|banding|fertigation|injection)',
            r'(?:sprayed|applied|incorporated|mixed)\s+(?:with|into|on)',
            r'(?:drench|dipping|coating|dusting)\s+(?:application|treatment)?'
        ]
        
        for pattern in method_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                application_data["application_methods"].append({
                    "method": match.group(0).strip(),
                    "context": text[max(0, match.start()-30):min(len(text), match.end()+30)],
                    "type": "regex_extracted"
                })
        
        # Extract timing information
        timing_patterns = [
            r'(?:at|during|before|after)\s+(?:planting|seeding|emergence|flowering|harvest)',
            r'(?:pre-emergence|post-emergence|pre-plant|post-plant)',
            r'(?:V\d+|R\d+)\s+stage',
            r'(?:\d+)\s+(?:days|weeks)\s+(?:before|after)\s+(?:planting|emergence|flowering)',
            r'(?:early|mid|late)\s+(?:season|spring|summer|fall|autumn)'
        ]
        
        for pattern in timing_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                application_data["timing"].append({
                    "timing": match.group(0).strip(),
                    "context": text[max(0, match.start()-30):min(len(text), match.end()+30)],
                    "type": "regex_extracted"
                })
        
        # Extract equipment
        equipment_patterns = [
            r'(?:sprayer|spreader|injector|nozzle|boom)',
            r'(?:aerial|ground|tractor-mounted|hand-held)\s+(?:application|sprayer)',
            r'(?:pressure|volume|droplet)\s+(?:setting|size|rate)'
        ]
        
        for pattern in equipment_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                application_data["equipment"].append({
                    "equipment": match.group(0).strip(),
                    "context": text[max(0, match.start()-30):min(len(text), match.end()+30)],
                    "type": "regex_extracted"
                })
        
        # Extract frequency
        frequency_patterns = [
            r'(?:once|twice|three times)\s+per\s+(?:season|year|month|week)',
            r'(?:single|multiple|repeated)\s+(?:application|treatment)',
            r'(?:every|each)\s+(?:\d+)\s+(?:days|weeks|months)',
            r'(?:weekly|monthly|seasonal|annual)\s+(?:application|treatment)'
        ]
        
        for pattern in frequency_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                application_data["frequency"].append({
                    "frequency": match.group(0).strip(),
                    "context": text[max(0, match.start()-30):min(len(text), match.end()+30)],
                    "type": "regex_extracted"
                })
        
        return application_data
    
    def _normalize_crop_name(self, crop_name: str) -> str:
        """Normalize crop name to scientific name using agro_entities.json configuration."""
        if not crop_name:
            return ""
        
        crop_name_lower = crop_name.lower().strip()
        
        # Check normalization rules from agro_entities.json
        crop_config = self.agro_entities.get("entity_types", {}).get("CROP", {})
        normalization_rules = crop_config.get("normalization_rules", {})
        
        # Check scientific names mapping
        scientific_names = normalization_rules.get("scientific_names", {})
        if crop_name_lower in scientific_names:
            return scientific_names[crop_name_lower]
        
        # Check common aliases mapping
        common_aliases = normalization_rules.get("common_aliases", {})
        if crop_name_lower in common_aliases:
            # Get the canonical name, then look up scientific name
            canonical_name = common_aliases[crop_name_lower]
            if canonical_name in scientific_names:
                return scientific_names[canonical_name]
            return canonical_name
        
        # Return original if no normalization found
        return crop_name

    def _calculate_openai_cost(self, model: str, tokens_used: int, input_tokens: int = None, output_tokens: int = None) -> float:
        """
        Calculate OpenAI cost based on model and token usage with current 2024 pricing.
        
        Args:
            model: The OpenAI model name
            tokens_used: Total tokens used (fallback if input/output not available)
            input_tokens: Input tokens used (for more accurate pricing)
            output_tokens: Output tokens used (for more accurate pricing)
            
        Returns:
            Cost in USD
        """
        if not tokens_used and not (input_tokens and output_tokens):
            return 0.0
        
        # Current OpenAI pricing (December 2024) - prices per 1M tokens
        # Separate input and output pricing for accurate calculation
        pricing = {
            # GPT-4o models
            "gpt-4o": {
                "input": 2.50,   # $2.50 per 1M input tokens
                "output": 10.00  # $10.00 per 1M output tokens
            },
            "gpt-4o-mini": {
                "input": 0.15,   # $0.15 per 1M input tokens  
                "output": 0.60   # $0.60 per 1M output tokens
            },
            "gpt-4o-2024-08-06": {
                "input": 2.50,
                "output": 10.00
            },
            "gpt-4o-2024-05-13": {
                "input": 5.00,   # $5.00 per 1M input tokens
                "output": 15.00  # $15.00 per 1M output tokens
            },
            
            # GPT-4 Turbo models
            "gpt-4-turbo": {
                "input": 10.00,  # $10.00 per 1M input tokens
                "output": 30.00  # $30.00 per 1M output tokens
            },
            "gpt-4-turbo-2024-04-09": {
                "input": 10.00,
                "output": 30.00
            },
            "gpt-4-turbo-preview": {
                "input": 10.00,
                "output": 30.00
            },
            
            # GPT-4 models
            "gpt-4": {
                "input": 30.00,  # $30.00 per 1M input tokens
                "output": 60.00  # $60.00 per 1M output tokens
            },
            "gpt-4-0613": {
                "input": 30.00,
                "output": 60.00
            },
            "gpt-4-32k": {
                "input": 60.00,  # $60.00 per 1M input tokens
                "output": 120.00 # $120.00 per 1M output tokens
            },
            
            # GPT-3.5 Turbo models
            "gpt-3.5-turbo": {
                "input": 0.50,   # $0.50 per 1M input tokens
                "output": 1.50   # $1.50 per 1M output tokens
            },
            "gpt-3.5-turbo-0125": {
                "input": 0.50,
                "output": 1.50
            },
            "gpt-3.5-turbo-instruct": {
                "input": 1.50,   # $1.50 per 1M input tokens
                "output": 2.00   # $2.00 per 1M output tokens
            },
            
            # Custom/hypothetical models
            "gpt-4.1-nano": {
                "input": 0.10,   # Estimated pricing for nano model
                "output": 0.20
            }
        }
        
        # Normalize model name (remove version suffixes and clean up)
        model_lower = model.lower().strip()
        
        # Find matching pricing
        model_pricing = None
        for key, price_info in pricing.items():
            if key in model_lower or model_lower.startswith(key):
                model_pricing = price_info
                break
        
        # If no exact match, try partial matches
        if not model_pricing:
            for key, price_info in pricing.items():
                if any(part in model_lower for part in key.split('-')[:2]):  # Match first two parts
                    model_pricing = price_info
                    logger.info(f"Using partial match pricing for model '{model}' -> '{key}'")
                    break
        
        # Calculate cost
        if model_pricing and input_tokens is not None and output_tokens is not None:
            # Use separate input/output pricing (most accurate)
            input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
            output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
            total_cost = input_cost + output_cost
            
            logger.debug(f"OpenAI cost calculation: {input_tokens} input + {output_tokens} output tokens = ${total_cost:.6f}")
            return total_cost
            
        elif model_pricing and tokens_used:
            # Fallback: estimate 70% input, 30% output (typical ratio)
            estimated_input = int(tokens_used * 0.7)
            estimated_output = int(tokens_used * 0.3)
            
            input_cost = (estimated_input / 1_000_000) * model_pricing["input"]
            output_cost = (estimated_output / 1_000_000) * model_pricing["output"]
            total_cost = input_cost + output_cost
            
            logger.debug(f"OpenAI cost estimation: {tokens_used} total tokens (est. {estimated_input} input + {estimated_output} output) = ${total_cost:.6f}")
            return total_cost
        
        # Final fallback pricing if model not found
        if tokens_used:
            fallback_rate = 5.00  # $5.00 per 1M tokens (conservative estimate)
            cost = (tokens_used / 1_000_000) * fallback_rate
            logger.warning(f"Unknown OpenAI model '{model}', using fallback rate ${fallback_rate}/1M tokens = ${cost:.6f}")
            return cost
        
        return 0.0

    def _calculate_mistral_cost(self, model: str, tokens_used: int, input_tokens: int = None, output_tokens: int = None) -> float:
        """
        Calculate Mistral cost based on model and token usage with current 2024 pricing.
        
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
            # Embedding models
            "mistral-embed": 0.10,     # $0.10 per 1M tokens
            
            # Chat models (if used in future)
            "mistral-tiny": 0.25,      # $0.25 per 1M tokens
            "mistral-small": 2.00,     # $2.00 per 1M tokens
            "mistral-medium": 6.00,    # $6.00 per 1M tokens
            "mistral-large": 24.00,    # $24.00 per 1M tokens
            
            # Vision models (for OCR)
            "pixtral-12b": 0.40,       # $0.40 per 1M tokens (estimated)
            "pixtral-12b-2409": 0.40,  # $0.40 per 1M tokens
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
            logger.debug(f"Mistral cost calculation: {tokens_used} tokens × ${cost_per_million}/1M = ${cost:.6f}")
            return cost
        
        # Fallback pricing
        fallback_rate = 1.00  # $1.00 per 1M tokens
        cost = (tokens_used / 1_000_000) * fallback_rate
        logger.warning(f"Unknown Mistral model '{model}', using fallback rate ${fallback_rate}/1M tokens = ${cost:.6f}")
        return cost

    @retry(**RETRY_MISTRAL_CONFIG)
    def extract_entities_and_relationships_mistral(self, text_chunk: str, document_id: Optional[str] = None, document_type: str = "general") -> Dict[str, Any]:
        """
        Use Mistral with function calling to extract specialized agricultural entities and relationships.
        """
        if not self.client:
            logger.error("Mistral client not initialized for extraction.")
            return {"entities": [], "relationships": []}

        # Define agricultural-specific extraction tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_PRODUCT",
                    "description": "Extracts agricultural products, materials, fertilizers, pesticides, or active ingredients.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_name": {"type": "string", "description": "The name of the product or material."},
                            "product_type": {"type": "string", "description": "Type: fertilizer, pesticide, herbicide, fungicide, biostimulant, etc."},
                            "brand": {"type": "string", "description": "Brand or trade name, if available."}
                        },
                        "required": ["product_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_CHEMICAL_COMPOUND",
                    "description": "Extracts chemical compounds, active ingredients, CAS numbers, EINECS codes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "compound_name": {"type": "string", "description": "Name of the chemical compound."},
                            "cas_number": {"type": "string", "description": "CAS number, if available."},
                            "einecs_code": {"type": "string", "description": "EINECS code, if available."},
                            "concentration": {"type": "string", "description": "Concentration or percentage, if mentioned."}
                        },
                        "required": ["compound_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_SPECIFICATION",
                    "description": "Extracts technical specifications, parameters, and analytical values.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "parameter_name": {"type": "string", "description": "Name of the specification or parameter."},
                            "value": {"type": "string", "description": "The measured or specified value."},
                            "unit": {"type": "string", "description": "Unit of measurement (%, mg/kg, pH, °C, etc.)."},
                            "test_method": {"type": "string", "description": "Test method or standard, if mentioned."}
                        },
                        "required": ["parameter_name", "value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_APPLICATION",
                    "description": "Extracts application information: crops, pests, diseases, application methods.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_crop": {"type": "string", "description": "Target crop or plant."},
                            "target_pest": {"type": "string", "description": "Target pest, disease, or weed."},
                            "application_method": {"type": "string", "description": "Application method: foliar, soil, seed treatment, etc."},
                            "application_rate": {"type": "string", "description": "Application rate with units (kg/ha, L/ha, etc.)."},
                            "timing": {"type": "string", "description": "Application timing or growth stage."}
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_SAFETY_INFO",
                    "description": "Extracts safety information: GHS codes, hazard statements, PPE requirements.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hazard_code": {"type": "string", "description": "GHS hazard code (H200, H300, etc.)."},
                            "precautionary_code": {"type": "string", "description": "Precautionary statement code (P101, P200, etc.)."},
                            "ppe_requirement": {"type": "string", "description": "Personal protective equipment requirement."},
                            "hazard_statement": {"type": "string", "description": "Full hazard statement text."}
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_REGULATORY",
                    "description": "Extracts regulatory information: registration numbers, certifications, standards.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "regulation_type": {"type": "string", "description": "Type: registration, certification, standard, etc."},
                            "regulation_number": {"type": "string", "description": "Registration or certification number."},
                            "issuing_authority": {"type": "string", "description": "Regulatory authority (EPA, EU, etc.)."},
                            "standard_name": {"type": "string", "description": "Standard name (ISO, ASTM, etc.)."}
                        },
                        "required": ["regulation_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_STORAGE_CONDITION",
                    "description": "Extracts storage conditions and requirements.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "temperature_range": {"type": "string", "description": "Storage temperature range."},
                            "humidity_requirement": {"type": "string", "description": "Humidity requirements."},
                            "special_conditions": {"type": "string", "description": "Special storage conditions."},
                            "shelf_life": {"type": "string", "description": "Shelf life or expiration information."}
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_COMPANY_INFO",
                    "description": "Extracts company information: manufacturers, distributors, contacts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company_name": {"type": "string", "description": "Company or organization name."},
                            "role": {"type": "string", "description": "Role: manufacturer, distributor, supplier, etc."},
                            "contact_info": {"type": "string", "description": "Contact information: address, phone, email."}
                        },
                        "required": ["company_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_relationship",
                    "description": "Creates relationships between extracted entities.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_entity": {"type": "string", "description": "Source entity name."},
                            "target_entity": {"type": "string", "description": "Target entity name."},
                            "relationship_type": {"type": "string", "description": "Type: CONTAINS, TARGETS, APPLIES_TO, HAS_SPECIFICATION, etc."},
                            "context": {"type": "string", "description": "Context or additional information about the relationship."}
                        },
                        "required": ["source_entity", "target_entity", "relationship_type"],
                    },
                },
            }
        ]

        # Create specialized system prompt based on document type
        system_prompts = {
            "safety_data_sheet": "You are an expert in analyzing Safety Data Sheets (SDS) for agricultural and chemical products. Extract all relevant safety information, hazard classifications, chemical identifiers, and regulatory data.",
            "technical": "You are an expert in analyzing technical datasheets for agricultural products. Focus on technical specifications, analytical parameters, quality standards, and application information.",
            "chemical": "You are an expert in chemical analysis. Extract chemical compounds, concentrations, molecular information, and analytical data.",
            "product_specification": "You are an expert in product specifications for agricultural inputs. Focus on product details, composition, specifications, and application guidelines."
        }
        
        system_prompt = system_prompts.get(document_type, "You are an expert in analyzing agricultural and chemical industry documents. Extract all relevant entities and their relationships using the provided tools.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze the following text and extract all relevant agricultural/chemical information:\n\n{text_chunk}"}
        ]

        extracted_entities = []
        extracted_relationships = []

        try:
            response = self.client.chat.complete(
                model="mistral-large-latest",  # Use latest for better function calling
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1  # Low temperature for consistent extraction
            )

            # Process tool calls
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    # Map function calls to entity structures
                    if function_name == "extract_PRODUCT":
                        extracted_entities.append({
                            "type": "PRODUCT", 
                            "value": arguments.get("product_name"),
                            "metadata": {
                                "product_type": arguments.get("product_type"),
                                "brand": arguments.get("brand"),
                                "extraction_method": "mistral_function_calling"
                            },
                            "context": text_chunk[:200] + "..." if len(text_chunk) > 200 else text_chunk,
                            "confidence": 0.9
                        })
                    
                    elif function_name == "extract_CHEMICAL_COMPOUND":
                        extracted_entities.append({
                            "type": "CHEMICAL_COMPOUND",
                            "value": arguments.get("compound_name"),
                            "metadata": {
                                "cas_number": arguments.get("cas_number"),
                                "einecs_code": arguments.get("einecs_code"),
                                "concentration": arguments.get("concentration"),
                                "extraction_method": "mistral_function_calling"
                            },
                            "context": text_chunk[:200] + "..." if len(text_chunk) > 200 else text_chunk,
                            "confidence": 0.9
                        })
                    
                    elif function_name == "extract_SPECIFICATION":
                        extracted_entities.append({
                            "type": "SPECIFICATION",
                            "value": f"{arguments.get('parameter_name')}: {arguments.get('value')} {arguments.get('unit', '')}".strip(),
                            "metadata": {
                                "parameter_name": arguments.get("parameter_name"),
                                "value": arguments.get("value"),
                                "unit": arguments.get("unit"),
                                "test_method": arguments.get("test_method"),
                                "extraction_method": "mistral_function_calling"
                            },
                            "context": text_chunk[:200] + "..." if len(text_chunk) > 200 else text_chunk,
                            "confidence": 0.9
                        })
                    
                    elif function_name == "extract_APPLICATION":
                        app_parts = []
                        if arguments.get("target_crop"):
                            app_parts.append(f"Crop: {arguments.get('target_crop')}")
                        if arguments.get("target_pest"):
                            app_parts.append(f"Target: {arguments.get('target_pest')}")
                        if arguments.get("application_method"):
                            app_parts.append(f"Method: {arguments.get('application_method')}")
                        if arguments.get("application_rate"):
                            app_parts.append(f"Rate: {arguments.get('application_rate')}")
                        
                        extracted_entities.append({
                            "type": "APPLICATION",
                            "value": "; ".join(app_parts) if app_parts else "Application information",
                            "metadata": {
                                "target_crop": arguments.get("target_crop"),
                                "target_pest": arguments.get("target_pest"),
                                "application_method": arguments.get("application_method"),
                                "application_rate": arguments.get("application_rate"),
                                "timing": arguments.get("timing"),
                                "extraction_method": "mistral_function_calling"
                            },
                            "context": text_chunk[:200] + "..." if len(text_chunk) > 200 else text_chunk,
                            "confidence": 0.85
                        })
                    
                    elif function_name == "extract_SAFETY_INFO":
                        safety_parts = []
                        if arguments.get("hazard_code"):
                            safety_parts.append(f"Hazard: {arguments.get('hazard_code')}")
                        if arguments.get("precautionary_code"):
                            safety_parts.append(f"Precaution: {arguments.get('precautionary_code')}")
                        if arguments.get("ppe_requirement"):
                            safety_parts.append(f"PPE: {arguments.get('ppe_requirement')}")
                        
                        extracted_entities.append({
                            "type": "SAFETY_INFO",
                            "value": "; ".join(safety_parts) if safety_parts else arguments.get("hazard_statement", "Safety information"),
                            "metadata": {
                                "hazard_code": arguments.get("hazard_code"),
                                "precautionary_code": arguments.get("precautionary_code"),
                                "ppe_requirement": arguments.get("ppe_requirement"),
                                "hazard_statement": arguments.get("hazard_statement"),
                                "extraction_method": "mistral_function_calling"
                            },
                            "context": text_chunk[:200] + "..." if len(text_chunk) > 200 else text_chunk,
                            "confidence": 0.9
                        })
                    
                    elif function_name == "extract_REGULATORY":
                        reg_parts = []
                        if arguments.get("regulation_number"):
                            reg_parts.append(f"Number: {arguments.get('regulation_number')}")
                        if arguments.get("issuing_authority"):
                            reg_parts.append(f"Authority: {arguments.get('issuing_authority')}")
                        if arguments.get("standard_name"):
                            reg_parts.append(f"Standard: {arguments.get('standard_name')}")
                        
                        extracted_entities.append({
                            "type": "REGULATORY",
                            "value": f"{arguments.get('regulation_type')}: {'; '.join(reg_parts)}" if reg_parts else arguments.get('regulation_type', 'Regulatory information'),
                            "metadata": {
                                "regulation_type": arguments.get("regulation_type"),
                                "regulation_number": arguments.get("regulation_number"),
                                "issuing_authority": arguments.get("issuing_authority"),
                                "standard_name": arguments.get("standard_name"),
                                "extraction_method": "mistral_function_calling"
                            },
                            "context": text_chunk[:200] + "..." if len(text_chunk) > 200 else text_chunk,
                            "confidence": 0.85
                        })
                    
                    elif function_name == "extract_STORAGE_CONDITION":
                        storage_parts = []
                        if arguments.get("temperature_range"):
                            storage_parts.append(f"Temperature: {arguments.get('temperature_range')}")
                        if arguments.get("humidity_requirement"):
                            storage_parts.append(f"Humidity: {arguments.get('humidity_requirement')}")
                        if arguments.get("shelf_life"):
                            storage_parts.append(f"Shelf life: {arguments.get('shelf_life')}")
                        
                        extracted_entities.append({
                            "type": "STORAGE_CONDITION",
                            "value": "; ".join(storage_parts) if storage_parts else arguments.get("special_conditions", "Storage conditions"),
                            "metadata": {
                                "temperature_range": arguments.get("temperature_range"),
                                "humidity_requirement": arguments.get("humidity_requirement"),
                                "special_conditions": arguments.get("special_conditions"),
                                "shelf_life": arguments.get("shelf_life"),
                                "extraction_method": "mistral_function_calling"
                            },
                            "context": text_chunk[:200] + "..." if len(text_chunk) > 200 else text_chunk,
                            "confidence": 0.85
                        })
                    
                    elif function_name == "extract_COMPANY_INFO":
                        company_value = arguments.get("company_name")
                        if arguments.get("role"):
                            company_value += f" ({arguments.get('role')})"
                        
                        extracted_entities.append({
                            "type": "COMPANY_INFO",
                            "value": company_value,
                            "metadata": {
                                "company_name": arguments.get("company_name"),
                                "role": arguments.get("role"),
                                "contact_info": arguments.get("contact_info"),
                                "extraction_method": "mistral_function_calling"
                            },
                            "context": text_chunk[:200] + "..." if len(text_chunk) > 200 else text_chunk,
                            "confidence": 0.8
                        })
                    
                    elif function_name == "create_relationship":
                        extracted_relationships.append({
                            "source_entity": arguments.get("source_entity"),
                            "target_entity": arguments.get("target_entity"),
                            "relationship_type": arguments.get("relationship_type"),
                            "context": arguments.get("context", text_chunk[:200] + "..." if len(text_chunk) > 200 else text_chunk),
                            "confidence": 0.8,
                            "extraction_method": "mistral_function_calling"
                        })

            # Log API usage
            if self.db:
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else None
                cost_usd = self._calculate_mistral_cost("mistral-large-latest", tokens_used) if tokens_used else None
                self.db.log_api_usage(
                    user_id=self.user_id, 
                    document_id=document_id, 
                    api_provider="mistral",
                    api_type="ner_extraction", 
                    tokens_used=tokens_used, 
                    cost_usd=cost_usd,
                    request_payload={"model": "mistral-large-latest", "document_type": document_type},
                    response_metadata={
                        "entities_extracted": len(extracted_entities),
                        "relationships_extracted": len(extracted_relationships),
                        "response_id": getattr(response, 'id', None)
                    }
                )
            
            logger.info(f"Mistral NER extracted {len(extracted_entities)} entities and {len(extracted_relationships)} relationships")
            return {"entities": extracted_entities, "relationships": extracted_relationships}

        except Exception as e:
            logger.error(f"Mistral NER/relationship extraction failed: {e}")
            return {"entities": [], "relationships": []}

if __name__ == "__main__":
    # Example usage
    processor = TextProcessor()
    
    # Example extracted data
    example_data = {
        "text": "AGREEMENT\n\nThis Agreement is made on 15 January 2025, by and between Party A (the \"Vendor\") and Party B (the \"Client\").\n\nPage 1 of 10\n\n1. SCOPE OF WORK\nVendor agrees to provide services as described in Appendix A for a fee of $10,000.\n\n2. TERM\nThis Agreement shall commence on 01/20/2025 and continue until 01/20/2026.\n\nPage 2 of 10",
        "metadata": {"title": "Service Agreement", "num_pages": 10},
        "source": "example_contract.pdf"
    }
    
    results = processor.process_document(example_data)
    
    print(f"Cleaned text ({len(results['cleaned_text'])} chars)")
    print(f"Number of chunks: {len(results['chunks'])}")
    print(f"Entities found: {results['entities']}")
    print(f"Relationships found: {results['relationships']}")
    print(f"Number of embeddings: {len(results['embeddings'])}")
