# product_fiche_processor.py - Enhanced Text Processing for Product Fiches
import os
import json
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import asyncio
from pathlib import Path

from src.processing.text_processor import TextProcessor
from src.generation.llm_generator import LLMGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedEntity:
    """Represents an extracted entity with metadata."""
    text: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str
    normalized_value: Optional[str] = None
    unit: Optional[str] = None
    numeric_value: Optional[float] = None

@dataclass
class ExtractedRelationship:
    """Represents an extracted relationship between entities."""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    context: str

class ProductFicheProcessor(TextProcessor):
    """
    Enhanced text processor specifically designed for product fiches, datasheets,
    safety data sheets, and technical specifications.
    
    Features:
    - Document type detection (SDS, Technical Datasheet, Product Specification, Brochure)
    - Enhanced entity extraction with 8 specialized entity types
    - Table structure recognition and extraction
    - Technical specification parsing with units
    - Safety information extraction
    - Regulatory compliance information
    - Company and contact information extraction
    """
    
    def __init__(self, llm_generator: Optional[LLMGenerator] = None):
        """Initialize the ProductFicheProcessor with enhanced capabilities."""
        super().__init__(llm_generator)
        
        # Load product fiche configuration
        self.product_config = self._load_product_fiche_config()
        
        # Initialize specialized patterns
        self._init_specialized_patterns()
        
        # Document type detection patterns
        self.document_type_patterns = self._init_document_type_patterns()
        
        # Table extraction patterns
        self.table_patterns = self._init_table_patterns()
        
        logger.info("ProductFicheProcessor initialized with enhanced entity extraction")

    def _load_product_fiche_config(self) -> Dict[str, Any]:
        """Load product fiche entity configuration."""
        config_path = Path(__file__).parent.parent.parent / "config" / "product_fiche_entities.json"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Product fiche config not found at {config_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in product fiche config: {e}")
            return {}

    def _init_specialized_patterns(self):
        """Initialize specialized regex patterns for product fiche extraction."""
        self.specialized_patterns = {
            "cas_number": re.compile(r'\bCAS\s*[:\-]?\s*(\d{1,7}-\d{2}-\d)\b', re.IGNORECASE),
            "einecs_number": re.compile(r'\bEINECS?\s*[:\-]?\s*(\d{3}-\d{3}-\d)\b', re.IGNORECASE),
            "ph_value": re.compile(r'\bpH\s*[:\-]?\s*(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\b', re.IGNORECASE),
            "temperature": re.compile(r'\b(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*°C\b'),
            "percentage": re.compile(r'\b(\d+(?:\.\d+)?)\s*%\b'),
            "pressure": re.compile(r'\b(\d+(?:\.\d+)?)\s*(kPa|MPa|bar|psi)\b', re.IGNORECASE),
            "viscosity": re.compile(r'\b(\d+(?:\.\d+)?)\s*(cP|mPa\.s|Pa\.s)\b', re.IGNORECASE),
            "particle_size": re.compile(r'\b(\d+(?:\.\d+)?)\s*(µm|μm|microns?|nm|mm)\b', re.IGNORECASE),
            "molecular_weight": re.compile(r'\bmolecular\s+weight\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(g/mol|Da|kDa)?\b', re.IGNORECASE),
            "flash_point": re.compile(r'\bflash\s+point\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*°C\b', re.IGNORECASE),
            "melting_point": re.compile(r'\bmelting\s+point\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*°C\b', re.IGNORECASE),
            "boiling_point": re.compile(r'\bboiling\s+point\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*°C\b', re.IGNORECASE),
            "density": re.compile(r'\bdensity\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(g/cm³|g/ml|kg/m³)\b', re.IGNORECASE),
            "shelf_life": re.compile(r'\bshelf\s+life\s*[:\-]?\s*(\d+)\s*(months?|years?)\b', re.IGNORECASE),
            "ghs_code": re.compile(r'\b(GHS\d+|H\d{3}|P\d{3}|EUH\d{3})\b', re.IGNORECASE),
            "iso_standard": re.compile(r'\b(ISO\s*\d+(?:-\d+)?|EN\s*\d+(?:-\d+)?)\b', re.IGNORECASE),
            "regulatory_body": re.compile(r'\b(FDA|EMA|EFSA|USP|EP|JP|GMP|HACCP)\b', re.IGNORECASE)
        }

    def _init_document_type_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for document type detection."""
        return {
            "SDS": [
                r"safety\s+data\s+sheet",
                r"\bSDS\b",
                r"\bMSDS\b",
                r"material\s+safety",
                r"hazard\s+identification",
                r"first\s+aid\s+measures"
            ],
            "TECHNICAL_DATASHEET": [
                r"technical\s+datasheet",
                r"technical\s+data\s+sheet",
                r"product\s+information",
                r"technical\s+specifications",
                r"product\s+overview"
            ],
            "PRODUCT_SPECIFICATION": [
                r"product\s+specification",
                r"\bPSPE\b",
                r"specification\s+sheet",
                r"quality\s+parameters",
                r"testing\s+methods"
            ],
            "BROCHURE": [
                r"brochure",
                r"product\s+range",
                r"product\s+portfolio",
                r"specialty\s+products",
                r"applications\s+guide"
            ]
        }

    def _init_table_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize patterns for table structure recognition."""
        return {
            "table_header": re.compile(r'^\s*\|.*\|.*\|\s*$', re.MULTILINE),
            "table_separator": re.compile(r'^\s*\|[\s\-\|]*\|\s*$', re.MULTILINE),
            "table_row": re.compile(r'^\s*\|.*\|.*\|\s*$', re.MULTILINE),
            "specification_table": re.compile(r'(parameter|property|specification|characteristic)\s*\|\s*(value|limit|range)', re.IGNORECASE),
            "composition_table": re.compile(r'(component|ingredient|compound)\s*\|\s*(percentage|content|amount)', re.IGNORECASE),
            "safety_table": re.compile(r'(hazard|precaution|warning)\s*\|\s*(code|statement|description)', re.IGNORECASE)
        }

    def detect_document_type(self, text: str, filename: str = "") -> str:
        """
        Detect the type of product document based on content and filename.
        
        Args:
            text: Document text content
            filename: Document filename (optional)
            
        Returns:
            Document type: "SDS", "TECHNICAL_DATASHEET", "PRODUCT_SPECIFICATION", or "BROCHURE"
        """
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Check filename first for quick detection
        if any(keyword in filename_lower for keyword in ["sds", "safety", "msds"]):
            return "SDS"
        elif any(keyword in filename_lower for keyword in ["pspe", "specification"]):
            return "PRODUCT_SPECIFICATION"
        elif any(keyword in filename_lower for keyword in ["datasheet", "technical"]):
            return "TECHNICAL_DATASHEET"
        elif any(keyword in filename_lower for keyword in ["brochure", "range", "portfolio"]):
            return "BROCHURE"
        
        # Score each document type based on pattern matches
        type_scores = {}
        for doc_type, patterns in self.document_type_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            type_scores[doc_type] = score
        
        # Return the type with highest score
        if not type_scores or max(type_scores.values()) == 0:
            return "TECHNICAL_DATASHEET"  # Default
        
        detected_type = max(type_scores, key=type_scores.get)
        logger.info(f"Detected document type: {detected_type} (score: {type_scores[detected_type]})")
        return detected_type

    def extract_enhanced_entities(self, text: str, document_type: str = "TECHNICAL_DATASHEET") -> Dict[str, List[ExtractedEntity]]:
        """
        Extract entities with enhanced metadata and confidence scores.
        
        Args:
            text: Text to analyze
            document_type: Type of document for prioritized extraction
            
        Returns:
            Dictionary of entity types to lists of ExtractedEntity objects
        """
        entities = {}
        
        # Get extraction priorities for this document type
        priorities = self.product_config.get("extraction_priorities", {}).get(document_type, [])
        entity_types = self.product_config.get("entity_types", {})
        
        # Process each entity type in priority order
        for entity_type in priorities:
            if entity_type not in entity_types:
                continue
                
            entity_config = entity_types[entity_type]
            entities[entity_type] = self._extract_entities_by_type(text, entity_type, entity_config)
        
        # Extract remaining entity types not in priorities
        for entity_type, entity_config in entity_types.items():
            if entity_type not in entities:
                entities[entity_type] = self._extract_entities_by_type(text, entity_type, entity_config)
        
        return entities

    def _extract_entities_by_type(self, text: str, entity_type: str, entity_config: Dict[str, Any]) -> List[ExtractedEntity]:
        """Extract entities of a specific type with confidence scoring."""
        entities = []
        patterns = entity_config.get("patterns", [])
        context_keywords = entity_config.get("context_keywords", [])
        
        # Extract using regex patterns
        for pattern in patterns:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for match in regex.finditer(text):
                    entity_text = match.group().strip()
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Get surrounding context
                    context_start = max(0, start_pos - 100)
                    context_end = min(len(text), end_pos + 100)
                    context = text[context_start:context_end].strip()
                    
                    # Calculate confidence based on context keywords
                    confidence = self._calculate_confidence(context, context_keywords)
                    
                    # Normalize and validate entity
                    normalized_value, unit, numeric_value = self._normalize_entity(entity_text, entity_type)
                    
                    if self._validate_entity(entity_text, entity_type):
                        entities.append(ExtractedEntity(
                            text=entity_text,
                            entity_type=entity_type,
                            confidence=confidence,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            context=context,
                            normalized_value=normalized_value,
                            unit=unit,
                            numeric_value=numeric_value
                        ))
            except re.error as e:
                logger.warning(f"Invalid regex pattern for {entity_type}: {pattern} - {e}")
        
        # Remove duplicates and sort by confidence
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.confidence, reverse=True)
        
        return entities

    def _calculate_confidence(self, context: str, context_keywords: List[str]) -> float:
        """Calculate confidence score based on context keywords."""
        if not context_keywords:
            return 0.5  # Default confidence
        
        context_lower = context.lower()
        keyword_matches = sum(1 for keyword in context_keywords if keyword.lower() in context_lower)
        
        # Base confidence + bonus for keyword matches
        confidence = 0.3 + (keyword_matches / len(context_keywords)) * 0.7
        return min(confidence, 1.0)

    def _normalize_entity(self, entity_text: str, entity_type: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Normalize entity value and extract units/numeric values."""
        normalization_rules = self.product_config.get("normalization_rules", {}).get(entity_type, {})
        
        normalized = entity_text.strip()
        unit = None
        numeric_value = None
        
        # Apply normalization rules
        if "remove_symbols" in normalization_rules:
            for symbol in normalization_rules["remove_symbols"]:
                normalized = normalized.replace(symbol, "")
        
        if "standardize_case" in normalization_rules:
            case_rule = normalization_rules["standardize_case"]
            if case_rule == "lower":
                normalized = normalized.lower()
            elif case_rule == "upper":
                normalized = normalized.upper()
            elif case_rule == "title":
                normalized = normalized.title()
        
        if "normalize_units" in normalization_rules:
            unit_mapping = normalization_rules["normalize_units"]
            for old_unit, new_unit in unit_mapping.items():
                if old_unit.lower() in normalized.lower():
                    unit = new_unit
                    break
        
        # Extract numeric value and unit for specifications
        if entity_type == "SPECIFICATION":
            numeric_match = re.search(r'(\d+(?:\.\d+)?)', normalized)
            if numeric_match:
                try:
                    numeric_value = float(numeric_match.group(1))
                except ValueError:
                    pass
                
                # Extract unit
                unit_match = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z%°]+)', normalized)
                if unit_match and not unit:
                    unit = unit_match.group(2)
        
        return normalized, unit, numeric_value

    def _validate_entity(self, entity_text: str, entity_type: str) -> bool:
        """Validate extracted entity against validation rules."""
        validation_rules = self.product_config.get("validation_rules", {}).get(entity_type, {})
        
        if not validation_rules:
            return True
        
        # Check length constraints
        if "min_length" in validation_rules and len(entity_text) < validation_rules["min_length"]:
            return False
        if "max_length" in validation_rules and len(entity_text) > validation_rules["max_length"]:
            return False
        
        # Check forbidden characters
        if "forbidden_chars" in validation_rules:
            for char in validation_rules["forbidden_chars"]:
                if char in entity_text:
                    return False
        
        # Check pattern matching
        if "pattern" in validation_rules:
            if not re.match(validation_rules["pattern"], entity_text):
                return False
        
        # Check uppercase requirement
        if validation_rules.get("must_contain_uppercase", False):
            if not any(c.isupper() for c in entity_text):
                return False
        
        return True

    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities based on text and position overlap."""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x.start_pos)
        
        deduplicated = []
        for entity in entities:
            # Check if this entity overlaps significantly with any existing entity
            is_duplicate = False
            for existing in deduplicated:
                if existing.entity_type == entity.entity_type:
                    # Check text similarity
                    if entity.text.lower() == existing.text.lower():
                        is_duplicate = True
                        break
                    
                    # Check position overlap (more than 50% overlap)
                    overlap_start = max(entity.start_pos, existing.start_pos)
                    overlap_end = min(entity.end_pos, existing.end_pos)
                    overlap_length = max(0, overlap_end - overlap_start)
                    
                    entity_length = entity.end_pos - entity.start_pos
                    existing_length = existing.end_pos - existing.start_pos
                    
                    overlap_ratio = overlap_length / min(entity_length, existing_length)
                    if overlap_ratio > 0.5:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                deduplicated.append(entity)
        
        return deduplicated

    def extract_relationships(self, entities: Dict[str, List[ExtractedEntity]], text: str) -> List[ExtractedRelationship]:
        """Extract relationships between entities based on proximity and context."""
        relationships = []
        relationship_types = self.product_config.get("relationship_types", {})
        
        for rel_type, rel_config in relationship_types.items():
            source_types = rel_config.get("source_types", [])
            target_types = rel_config.get("target_types", [])
            
            # Find relationships between entities of specified types
            for source_type in source_types:
                if source_type not in entities:
                    continue
                    
                for target_type in target_types:
                    if target_type not in entities:
                        continue
                    
                    # Check each source-target pair
                    for source_entity in entities[source_type]:
                        for target_entity in entities[target_type]:
                            relationship = self._detect_relationship(
                                source_entity, target_entity, rel_type, text
                            )
                            if relationship:
                                relationships.append(relationship)
        
        return relationships

    def _detect_relationship(self, source_entity: ExtractedEntity, target_entity: ExtractedEntity, 
                           rel_type: str, text: str) -> Optional[ExtractedRelationship]:
        """Detect if a relationship exists between two entities."""
        # Calculate distance between entities
        distance = abs(source_entity.start_pos - target_entity.start_pos)
        
        # Only consider entities within reasonable proximity (500 characters)
        if distance > 500:
            return None
        
        # Get context around both entities
        context_start = min(source_entity.start_pos, target_entity.start_pos) - 50
        context_end = max(source_entity.end_pos, target_entity.end_pos) + 50
        context_start = max(0, context_start)
        context_end = min(len(text), context_end)
        context = text[context_start:context_end]
        
        # Check for relationship indicators based on type
        confidence = self._calculate_relationship_confidence(rel_type, context, source_entity, target_entity)
        
        if confidence > 0.3:  # Minimum confidence threshold
            return ExtractedRelationship(
                source_entity=source_entity.text,
                target_entity=target_entity.text,
                relationship_type=rel_type,
                confidence=confidence,
                context=context
            )
        
        return None

    def _calculate_relationship_confidence(self, rel_type: str, context: str, 
                                         source_entity: ExtractedEntity, target_entity: ExtractedEntity) -> float:
        """Calculate confidence for a specific relationship type."""
        context_lower = context.lower()
        
        # Relationship-specific indicators
        indicators = {
            "HAS_SPECIFICATION": ["specification", "specs", "parameter", "value", "property"],
            "HAS_APPLICATION": ["used for", "application", "purpose", "function", "intended"],
            "CONTAINS_COMPOUND": ["contains", "composed of", "ingredient", "component"],
            "HAS_SAFETY_INFO": ["safety", "hazard", "warning", "caution", "danger"],
            "REQUIRES_STORAGE": ["store", "storage", "keep", "maintain", "condition"],
            "REGULATED_BY": ["approved by", "regulated", "compliance", "standard", "certified"],
            "MANUFACTURED_BY": ["manufactured", "produced", "made by", "supplier", "company"],
            "SPECIFIED_AS": ["specified", "defined", "measured", "determined"]
        }
        
        rel_indicators = indicators.get(rel_type, [])
        indicator_matches = sum(1 for indicator in rel_indicators if indicator in context_lower)
        
        # Base confidence from indicator matches
        confidence = min(indicator_matches * 0.2, 0.6)
        
        # Proximity bonus (closer entities get higher confidence)
        distance = abs(source_entity.start_pos - target_entity.start_pos)
        proximity_bonus = max(0, (200 - distance) / 200) * 0.3
        
        # Entity confidence bonus
        entity_confidence_bonus = (source_entity.confidence + target_entity.confidence) / 2 * 0.1
        
        total_confidence = confidence + proximity_bonus + entity_confidence_bonus
        return min(total_confidence, 1.0)

    def extract_table_data(self, text: str) -> List[Dict[str, Any]]:
        """Extract and structure table data from the text."""
        tables = []
        
        # Look for markdown-style tables
        table_sections = self._find_table_sections(text)
        
        for table_text in table_sections:
            table_data = self._parse_table_section(table_text)
            if table_data:
                tables.append(table_data)
        
        return tables

    def _find_table_sections(self, text: str) -> List[str]:
        """Find table sections in the text."""
        table_sections = []
        lines = text.split('\n')
        
        current_table = []
        in_table = False
        
        for line in lines:
            line = line.strip()
            
            # Check if line looks like a table row
            if '|' in line and line.count('|') >= 2:
                current_table.append(line)
                in_table = True
            elif in_table and (not line or not line.startswith('|')):
                # End of table
                if current_table and len(current_table) >= 2:
                    table_sections.append('\n'.join(current_table))
                current_table = []
                in_table = False
        
        # Add final table if exists
        if current_table and len(current_table) >= 2:
            table_sections.append('\n'.join(current_table))
        
        return table_sections

    def _parse_table_section(self, table_text: str) -> Optional[Dict[str, Any]]:
        """Parse a table section into structured data."""
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return None
        
        # Extract headers from first line
        header_line = lines[0]
        if not header_line.startswith('|') or not header_line.endswith('|'):
            return None
        
        headers = [col.strip() for col in header_line.split('|')[1:-1]]
        
        # Skip separator line if present
        data_start_idx = 1
        if len(lines) > 1 and all(c in '|-: ' for c in lines[1]):
            data_start_idx = 2
        
        # Extract data rows
        rows = []
        for line in lines[data_start_idx:]:
            if line.startswith('|') and line.endswith('|'):
                row_data = [col.strip() for col in line.split('|')[1:-1]]
                if len(row_data) == len(headers):
                    row_dict = dict(zip(headers, row_data))
                    rows.append(row_dict)
        
        if not rows:
            return None
        
        # Determine table type
        table_type = self._classify_table(headers, rows)
        
        return {
            "type": table_type,
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "column_count": len(headers)
        }

    def _classify_table(self, headers: List[str], rows: List[Dict[str, str]]) -> str:
        """Classify the type of table based on headers and content."""
        headers_lower = [h.lower() for h in headers]
        
        # Check for specification tables
        if any(keyword in ' '.join(headers_lower) for keyword in 
               ['parameter', 'property', 'specification', 'value', 'limit', 'range']):
            return "specification"
        
        # Check for composition tables
        if any(keyword in ' '.join(headers_lower) for keyword in 
               ['component', 'ingredient', 'compound', 'percentage', 'content']):
            return "composition"
        
        # Check for safety tables
        if any(keyword in ' '.join(headers_lower) for keyword in 
               ['hazard', 'warning', 'precaution', 'ghs', 'safety']):
            return "safety"
        
        # Check for regulatory tables
        if any(keyword in ' '.join(headers_lower) for keyword in 
               ['standard', 'regulation', 'compliance', 'approval']):
            return "regulatory"
        
        return "general"

    def process_document(self, text: str, document_id: Optional[str] = None, 
                        filename: str = "") -> Dict[str, Any]:
        """
        Process a complete product fiche document with enhanced extraction.
        
        Args:
            text: Document text content
            document_id: Optional document identifier
            filename: Document filename
            
        Returns:
            Comprehensive extraction results
        """
        logger.info(f"Processing product fiche document: {filename}")
        
        # Detect document type
        document_type = self.detect_document_type(text, filename)
        
        # Extract entities with enhanced metadata
        entities = self.extract_enhanced_entities(text, document_type)
        
        # Extract relationships
        relationships = self.extract_relationships(entities, text)
        
        # Extract table data
        tables = self.extract_table_data(text)
        
        # Process with specialized extraction methods
        specifications = self._extract_technical_specifications(text)
        safety_info = self._extract_safety_information(text)
        regulatory_info = self._extract_regulatory_information(text)
        company_info = self._extract_company_information(text)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_extraction_quality(entities, relationships, tables)
        
        return {
            "document_id": document_id,
            "document_type": document_type,
            "filename": filename,
            "entities": {entity_type: [self._entity_to_dict(e) for e in entity_list] 
                        for entity_type, entity_list in entities.items()},
            "relationships": [self._relationship_to_dict(r) for r in relationships],
            "tables": tables,
            "specifications": specifications,
            "safety_info": safety_info,
            "regulatory_info": regulatory_info,
            "company_info": company_info,
            "quality_metrics": quality_metrics,
            "processing_metadata": {
                "processor_version": "ProductFicheProcessor_v1.0",
                "entity_types_extracted": len(entities),
                "total_entities": sum(len(entity_list) for entity_list in entities.values()),
                "total_relationships": len(relationships),
                "tables_extracted": len(tables),
                "document_type_confidence": 0.8  # Placeholder
            }
        }

    def _extract_technical_specifications(self, text: str) -> Dict[str, Any]:
        """Extract technical specifications using specialized patterns."""
        specs = {}
        
        for spec_name, pattern in self.specialized_patterns.items():
            matches = pattern.findall(text)
            if matches:
                specs[spec_name] = matches
        
        return specs

    def _extract_safety_information(self, text: str) -> Dict[str, Any]:
        """Extract safety-related information."""
        safety_info = {
            "ghs_codes": [],
            "hazard_statements": [],
            "precautionary_statements": [],
            "signal_words": []
        }
        
        # Extract GHS codes
        ghs_matches = self.specialized_patterns["ghs_code"].findall(text)
        safety_info["ghs_codes"] = list(set(ghs_matches))
        
        # Extract signal words
        signal_words = re.findall(r'\b(DANGER|WARNING|CAUTION)\b', text, re.IGNORECASE)
        safety_info["signal_words"] = list(set(signal_words))
        
        return safety_info

    def _extract_regulatory_information(self, text: str) -> Dict[str, Any]:
        """Extract regulatory and compliance information."""
        regulatory_info = {
            "standards": [],
            "regulatory_bodies": [],
            "certifications": []
        }
        
        # Extract ISO/EN standards
        standards = self.specialized_patterns["iso_standard"].findall(text)
        regulatory_info["standards"] = list(set(standards))
        
        # Extract regulatory bodies
        bodies = self.specialized_patterns["regulatory_body"].findall(text)
        regulatory_info["regulatory_bodies"] = list(set(bodies))
        
        return regulatory_info

    def _extract_company_information(self, text: str) -> Dict[str, Any]:
        """Extract company and contact information."""
        company_info = {
            "companies": [],
            "addresses": [],
            "contact_info": []
        }
        
        # Extract company names (simplified)
        company_pattern = re.compile(r'\b[A-Z][a-z]+\s+(?:GmbH|Ltd|Inc|Corp|SA|SAS|BV)\b')
        companies = company_pattern.findall(text)
        company_info["companies"] = list(set(companies))
        
        return company_info

    def _calculate_extraction_quality(self, entities: Dict[str, List[ExtractedEntity]], 
                                    relationships: List[ExtractedRelationship], 
                                    tables: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate quality metrics for the extraction."""
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        avg_entity_confidence = 0.0
        
        if total_entities > 0:
            total_confidence = sum(
                sum(entity.confidence for entity in entity_list)
                for entity_list in entities.values()
            )
            avg_entity_confidence = total_confidence / total_entities
        
        avg_relationship_confidence = 0.0
        if relationships:
            avg_relationship_confidence = sum(r.confidence for r in relationships) / len(relationships)
        
        return {
            "entity_count": total_entities,
            "relationship_count": len(relationships),
            "table_count": len(tables),
            "avg_entity_confidence": avg_entity_confidence,
            "avg_relationship_confidence": avg_relationship_confidence,
            "extraction_completeness": min(1.0, total_entities / 50),  # Assume 50 is good coverage
            "overall_quality": (avg_entity_confidence + avg_relationship_confidence) / 2
        }

    def _entity_to_dict(self, entity: ExtractedEntity) -> Dict[str, Any]:
        """Convert ExtractedEntity to dictionary."""
        return {
            "text": entity.text,
            "entity_type": entity.entity_type,
            "confidence": entity.confidence,
            "start_pos": entity.start_pos,
            "end_pos": entity.end_pos,
            "context": entity.context,
            "normalized_value": entity.normalized_value,
            "unit": entity.unit,
            "numeric_value": entity.numeric_value
        }

    def _relationship_to_dict(self, relationship: ExtractedRelationship) -> Dict[str, Any]:
        """Convert ExtractedRelationship to dictionary."""
        return {
            "source_entity": relationship.source_entity,
            "target_entity": relationship.target_entity,
            "relationship_type": relationship.relationship_type,
            "confidence": relationship.confidence,
            "context": relationship.context
        }

# Convenience function for easy import
def create_product_fiche_processor(llm_generator: Optional[LLMGenerator] = None) -> ProductFicheProcessor:
    """Create and return a ProductFicheProcessor instance."""
    return ProductFicheProcessor(llm_generator) 