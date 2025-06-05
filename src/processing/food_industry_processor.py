"""
Food Industry Processor for B2B Ingredients.

This module provides specialized processing for food industry documents including
ingredient specifications, food safety data sheets, nutritional information,
and regulatory compliance documents.
"""

import re
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from src.processing.product_fiche_processor import ProductFicheProcessor, ExtractedEntity, ExtractedRelationship

logger = logging.getLogger(__name__)

@dataclass
class FoodEntity(ExtractedEntity):
    """Extended entity class for food industry specific information."""
    allergen_info: Optional[Dict[str, Any]] = None
    nutritional_value: Optional[float] = None
    regulatory_status: Optional[str] = None
    food_grade: Optional[bool] = None
    shelf_life: Optional[str] = None

class FoodIndustryProcessor(ProductFicheProcessor):
    """Specialized processor for food industry B2B ingredients and products."""
    
    def __init__(self, db_client=None, llm_generator=None):
        """Initialize with food industry specific configurations."""
        # Initialize parent class (ProductFicheProcessor) with appropriate parameters
        # Pass None for llm_generator to parent since it will be handled separately
        super().__init__(None)
        
        # Store both db_client and llm_generator for future use
        self.db_client = db_client
        self.llm_generator = llm_generator
        
        # Load food industry configuration
        self.food_config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "food_industry_entities.json"
        )
        self.load_food_config()
        
        # Initialize food-specific patterns
        self._init_food_patterns()
        
        # Food safety keywords for enhanced classification
        self.food_safety_keywords = [
            'haccp', 'gmp', 'brc', 'sqf', 'ifs', 'fssc', 'gras', 'fda approved',
            'food grade', 'kosher', 'halal', 'organic', 'non-gmo'
        ]
    
    def load_food_config(self):
        """Load food industry entity configuration."""
        try:
            with open(self.food_config_path, 'r', encoding='utf-8') as f:
                self.food_config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load food industry config: {e}")
            self.food_config = {"entity_types": {}, "relationship_types": {}}
    
    def _init_food_patterns(self):
        """Initialize food industry specific regex patterns."""
        self.food_patterns = {
            # Nutritional information
            "vitamin_content": re.compile(r'\bvitamin\s+([A-K]\d*)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*([μmg/IU]+)', re.IGNORECASE),
            "mineral_content": re.compile(r'\b(calcium|iron|zinc|magnesium|potassium)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(mg|g|%)', re.IGNORECASE),
            "caloric_value": re.compile(r'\b(\d+(?:\.\d+)?)\s*(cal|kcal|kJ)(?:/(?:100)?g)?\b', re.IGNORECASE),
            
            # Food additives and E-numbers
            "e_number": re.compile(r'\bE\s*(\d{3,4})\b', re.IGNORECASE),
            "ins_number": re.compile(r'\bINS\s*(\d{3,4})\b', re.IGNORECASE),
            "gras_status": re.compile(r'\b(GRAS)\s*(?:status|approved|recognized)?\b', re.IGNORECASE),
            
            # Allergen information
            "allergen_declaration": re.compile(r'\b(?:contains?|may contain)\s+([^.]+)', re.IGNORECASE),
            "allergen_free": re.compile(r'\b(gluten|dairy|nut|soy|egg|fish|shellfish)-free\b', re.IGNORECASE),
            
            # Processing methods
            "processing_method": re.compile(r'\b(spray|freeze)\s+dr(?:y|ied)\b|\b(ferment|extract|distill)(?:ed|ation)\b', re.IGNORECASE),
            
            # Storage and shelf life
            "shelf_life": re.compile(r'\bshelf\s+life\s*[:\-]?\s*(\d+)\s*(days?|months?|years?)\b', re.IGNORECASE),
            "storage_temp": re.compile(r'\bstore\s+at\s+(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*°C\b', re.IGNORECASE),
            
            # Microbiological specifications
            "micro_spec": re.compile(r'\b(total\s+plate\s+count|yeast|mold|e\.?\s*coli|salmonella)\s*[:\-<]?\s*(\d+(?:\.\d+)?)\s*(cfu|mpn)/g\b', re.IGNORECASE),
            
            # Food applications
            "food_application": re.compile(r'\b(?:used?\s+in|suitable\s+for|application\s+in)\s+([^.]+)', re.IGNORECASE),
        }
    
    def extract_food_entities(self, text: str, document_type: str = "FOOD_DATASHEET") -> Dict[str, List[FoodEntity]]:
        """
        Extract food industry specific entities with enhanced metadata.
        
        Args:
            text: Text to analyze
            document_type: Type of food industry document
            
        Returns:
            Dictionary of entity types to lists of FoodEntity objects
        """
        entities = {}
        
        # Get food entity types
        food_entity_types = self.food_config.get("entity_types", {})
        
        # Process each food entity type
        for entity_type, entity_config in food_entity_types.items():
            entities[entity_type] = self._extract_food_entities_by_type(text, entity_type, entity_config)
        
        # Extract traditional product entities as well
        product_entities = super().extract_enhanced_entities(text, document_type)
        
        # Merge and enhance with food-specific information
        for entity_type, entity_list in product_entities.items():
            enhanced_entities = []
            for entity in entity_list:
                food_entity = self._enhance_entity_with_food_info(entity, text)
                enhanced_entities.append(food_entity)
            
            if entity_type in entities:
                entities[entity_type].extend(enhanced_entities)
            else:
                entities[entity_type] = enhanced_entities
        
        return entities
    
    def _extract_food_entities_by_type(self, text: str, entity_type: str, entity_config: Dict[str, Any]) -> List[FoodEntity]:
        """Extract entities of a specific food industry type."""
        entities = []
        patterns = entity_config.get("patterns", [])
        examples = entity_config.get("examples", [])
        
        # Extract using regex patterns
        for pattern in patterns:
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                for match in regex.finditer(text):
                    entity_text = match.group().strip()
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Get surrounding context
                    context_start = max(0, start_pos - 150)
                    context_end = min(len(text), end_pos + 150)
                    context = text[context_start:context_end].strip()
                    
                    # Calculate confidence
                    confidence = self._calculate_food_confidence(entity_text, entity_type, context)
                    
                    # Extract food-specific information
                    food_info = self._extract_food_specific_info(entity_text, context, entity_type)
                    
                    if confidence > 0.5:  # Threshold for food entities
                        entities.append(FoodEntity(
                            text=entity_text,
                            entity_type=entity_type,
                            confidence=confidence,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            context=context,
                            normalized_value=food_info.get("normalized_value", entity_text),
                            unit=food_info.get("unit"),
                            numeric_value=food_info.get("numeric_value"),
                            allergen_info=food_info.get("allergen_info"),
                            nutritional_value=food_info.get("nutritional_value"),
                            regulatory_status=food_info.get("regulatory_status"),
                            food_grade=food_info.get("food_grade"),
                            shelf_life=food_info.get("shelf_life")
                        ))
            except re.error as e:
                logger.warning(f"Invalid regex pattern for {entity_type}: {pattern}, error: {e}")
        
        # Extract using example matching
        for example in examples:
            example_lower = example.lower()
            text_lower = text.lower()
            
            start = 0
            while True:
                pos = text_lower.find(example_lower, start)
                if pos == -1:
                    break
                
                # Check word boundaries
                if ((pos == 0 or not text[pos-1].isalnum()) and 
                    (pos + len(example) >= len(text) or not text[pos + len(example)].isalnum())):
                    
                    context_start = max(0, pos - 100)
                    context_end = min(len(text), pos + len(example) + 100)
                    context = text[context_start:context_end].strip()
                    
                    confidence = self._calculate_food_confidence(example, entity_type, context)
                    food_info = self._extract_food_specific_info(example, context, entity_type)
                    
                    if confidence > 0.6:
                        entities.append(FoodEntity(
                            text=example,
                            entity_type=entity_type,
                            confidence=confidence,
                            start_pos=pos,
                            end_pos=pos + len(example),
                            context=context,
                            normalized_value=food_info.get("normalized_value", example),
                            unit=food_info.get("unit"),
                            numeric_value=food_info.get("numeric_value"),
                            allergen_info=food_info.get("allergen_info"),
                            nutritional_value=food_info.get("nutritional_value"),
                            regulatory_status=food_info.get("regulatory_status"),
                            food_grade=food_info.get("food_grade"),
                            shelf_life=food_info.get("shelf_life")
                        ))
                
                start = pos + 1
        
        return entities
    
    def _calculate_food_confidence(self, entity_text: str, entity_type: str, context: str) -> float:
        """Calculate confidence score for food industry entities."""
        confidence = 0.5  # Base confidence
        
        context_lower = context.lower()
        entity_lower = entity_text.lower()
        
        # Food industry specific confidence boosters
        food_indicators = {
            "FOOD_INGREDIENT": ["ingredient", "additive", "preservative", "stabilizer"],
            "FOOD_ADDITIVE_TYPE": ["function", "purpose", "used as", "acts as"],
            "FOOD_APPLICATION": ["used in", "suitable for", "application", "product"],
            "FOOD_SAFETY_STANDARD": ["approved", "certified", "compliant", "standard"],
            "NUTRITIONAL_COMPONENT": ["vitamin", "mineral", "nutrient", "supplement"],
            "ALLERGEN_INFO": ["contains", "allergen", "free", "may contain"]
        }
        
        indicators = food_indicators.get(entity_type, [])
        indicator_matches = sum(1 for indicator in indicators if indicator in context_lower)
        confidence += indicator_matches * 0.15
        
        # Specific pattern bonuses
        if entity_type == "FOOD_SAFETY_STANDARD":
            if any(keyword in context_lower for keyword in self.food_safety_keywords):
                confidence += 0.2
        
        if entity_type == "NUTRITIONAL_COMPONENT":
            if re.search(r'\d+\s*(?:mg|g|%|iu)', context_lower):
                confidence += 0.15
        
        if entity_type == "ALLERGEN_INFO":
            if "allergen" in context_lower or "contains" in context_lower:
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _extract_food_specific_info(self, entity_text: str, context: str, entity_type: str) -> Dict[str, Any]:
        """Extract food industry specific information from entity context."""
        info = {}
        
        # Extract nutritional values
        if entity_type == "NUTRITIONAL_COMPONENT":
            vitamin_match = self.food_patterns["vitamin_content"].search(context)
            if vitamin_match:
                info["nutritional_value"] = float(vitamin_match.group(2))
                info["unit"] = vitamin_match.group(3)
            
            mineral_match = self.food_patterns["mineral_content"].search(context)
            if mineral_match:
                info["nutritional_value"] = float(mineral_match.group(2))
                info["unit"] = mineral_match.group(3)
        
        # Extract regulatory status
        if entity_type == "FOOD_SAFETY_STANDARD":
            if self.food_patterns["gras_status"].search(context):
                info["regulatory_status"] = "GRAS approved"
            
            e_number_match = self.food_patterns["e_number"].search(context)
            if e_number_match:
                info["regulatory_status"] = f"E{e_number_match.group(1)}"
        
        # Extract allergen information
        if entity_type == "ALLERGEN_INFO":
            allergen_match = self.food_patterns["allergen_declaration"].search(context)
            if allergen_match:
                allergens = [a.strip() for a in allergen_match.group(1).split(',')]
                info["allergen_info"] = {"contains": allergens}
            
            free_match = self.food_patterns["allergen_free"].search(context)
            if free_match:
                info["allergen_info"] = {"free_from": free_match.group(1)}
        
        # Extract shelf life
        shelf_life_match = self.food_patterns["shelf_life"].search(context)
        if shelf_life_match:
            info["shelf_life"] = f"{shelf_life_match.group(1)} {shelf_life_match.group(2)}"
        
        # Determine if food grade
        if "food grade" in context.lower() or "food quality" in context.lower():
            info["food_grade"] = True
        
        return info
    
    def _enhance_entity_with_food_info(self, entity: ExtractedEntity, full_text: str) -> FoodEntity:
        """Enhance regular entity with food industry specific information."""
        # Convert to FoodEntity
        food_entity = FoodEntity(
            text=entity.text,
            entity_type=entity.entity_type,
            confidence=entity.confidence,
            start_pos=entity.start_pos,
            end_pos=entity.end_pos,
            context=entity.context,
            normalized_value=entity.normalized_value,
            unit=entity.unit,
            numeric_value=entity.numeric_value
        )
        
        # Add food-specific enhancements
        food_info = self._extract_food_specific_info(entity.text, entity.context, entity.entity_type)
        food_entity.allergen_info = food_info.get("allergen_info")
        food_entity.nutritional_value = food_info.get("nutritional_value")
        food_entity.regulatory_status = food_info.get("regulatory_status")
        food_entity.food_grade = food_info.get("food_grade")
        food_entity.shelf_life = food_info.get("shelf_life")
        
        return food_entity
    
    def extract_food_relationships(self, entities: Dict[str, List[FoodEntity]], text: str) -> List[ExtractedRelationship]:
        """Extract food industry specific relationships between entities."""
        relationships = []
        
        # Get food relationship types
        food_relationship_types = self.food_config.get("relationship_types", {})
        
        # Extract relationships using food-specific patterns
        for rel_type, rel_config in food_relationship_types.items():
            patterns = rel_config.get("patterns", [])
            
            for pattern in patterns:
                try:
                    regex = re.compile(f'([^.]+?)\\s+{pattern}\\s+([^.]+)', re.IGNORECASE)
                    
                    for match in regex.finditer(text):
                        source_text = match.group(1).strip()
                        target_text = match.group(2).strip()
                        
                        # Find matching entities
                        source_entity = self._find_matching_entity(source_text, entities)
                        target_entity = self._find_matching_entity(target_text, entities)
                        
                        if source_entity and target_entity:
                            confidence = self._calculate_food_relationship_confidence(
                                rel_type, match.group(0), source_entity, target_entity
                            )
                            
                            if confidence > 0.5:
                                relationships.append(ExtractedRelationship(
                                    source_entity=source_entity.text,
                                    target_entity=target_entity.text,
                                    relationship_type=rel_type,
                                    confidence=confidence,
                                    context=match.group(0)
                                ))
                
                except re.error as e:
                    logger.warning(f"Invalid relationship pattern for {rel_type}: {pattern}, error: {e}")
        
        return relationships
    
    def _calculate_food_relationship_confidence(self, rel_type: str, context: str, 
                                              source_entity: FoodEntity, target_entity: FoodEntity) -> float:
        """Calculate confidence for food industry specific relationships."""
        context_lower = context.lower()
        
        # Food relationship indicators
        food_indicators = {
            "USED_IN_FOOD": ["used in", "added to", "incorporated", "ingredient in"],
            "HAS_FUNCTION": ["functions as", "acts as", "serves as", "provides"],
            "APPROVED_FOR": ["approved for", "permitted", "GRAS for", "allowed"],
            "CONTAINS_ALLERGEN": ["contains", "source of", "derived from", "may contain"],
            "REPLACES_INGREDIENT": ["replaces", "substitute", "alternative", "instead of"],
            "ENHANCES_PROPERTY": ["enhances", "improves", "stabilizes", "extends", "preserves"]
        }
        
        indicators = food_indicators.get(rel_type, [])
        indicator_matches = sum(1 for indicator in indicators if indicator in context_lower)
        
        # Base confidence from indicators
        confidence = min(indicator_matches * 0.25, 0.7)
        
        # Entity type compatibility bonus
        compatible_pairs = {
            "USED_IN_FOOD": [("FOOD_INGREDIENT", "FOOD_APPLICATION"), ("NUTRITIONAL_COMPONENT", "FOOD_APPLICATION")],
            "HAS_FUNCTION": [("FOOD_INGREDIENT", "FOOD_ADDITIVE_TYPE"), ("NUTRITIONAL_COMPONENT", "FOOD_ADDITIVE_TYPE")],
            "APPROVED_FOR": [("FOOD_INGREDIENT", "FOOD_SAFETY_STANDARD"), ("NUTRITIONAL_COMPONENT", "FOOD_SAFETY_STANDARD")],
            "CONTAINS_ALLERGEN": [("FOOD_INGREDIENT", "ALLERGEN_INFO"), ("PRODUCT", "ALLERGEN_INFO")]
        }
        
        pairs = compatible_pairs.get(rel_type, [])
        for source_type, target_type in pairs:
            if (source_entity.entity_type == source_type and target_entity.entity_type == target_type):
                confidence += 0.2
                break
        
        # Food industry context bonus
        if any(keyword in context_lower for keyword in ["food", "ingredient", "additive", "preservative"]):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def process_food_document(self, text: str, document_id: Optional[str] = None, 
                             filename: str = "") -> Dict[str, Any]:
        """
        Process a complete food industry document.
        
        Args:
            text: Document text
            document_id: Optional document identifier
            filename: Document filename
            
        Returns:
            Comprehensive food industry document analysis
        """
        # Classify document type
        document_type = self._classify_food_document_type(text, filename)
        
        # Extract food entities
        entities = self.extract_food_entities(text, document_type)
        
        # Extract food relationships
        relationships = self.extract_food_relationships(entities, text)
        
        # Extract additional food-specific information
        nutritional_info = self._extract_nutritional_information(text)
        allergen_info = self._extract_allergen_information(text)
        regulatory_info = self._extract_food_regulatory_information(text)
        storage_info = self._extract_food_storage_information(text)
        
        # Extract tables (inherit from parent)
        tables = self._extract_tables_from_text(text)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_food_quality_metrics(entities, relationships, text)
        
        return {
            "document_id": document_id,
            "document_type": document_type,
            "filename": filename,
            "entities": {entity_type: [self._food_entity_to_dict(e) for e in entity_list] 
                        for entity_type, entity_list in entities.items()},
            "relationships": [self._relationship_to_dict(r) for r in relationships],
            "nutritional_information": nutritional_info,
            "allergen_information": allergen_info,
            "regulatory_information": regulatory_info,
            "storage_information": storage_info,
            "tables": tables,
            "quality_metrics": quality_metrics,
            "processing_metadata": {
                "processor_version": "FoodIndustryProcessor_v1.0",
                "entity_types_extracted": len(entities),
                "total_entities": sum(len(entity_list) for entity_list in entities.values()),
                "total_relationships": len(relationships),
                "food_specific_features": True,
                "document_type_confidence": 0.85
            }
        }
    
    def _classify_food_document_type(self, text: str, filename: str) -> str:
        """Classify the type of food industry document."""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Food Safety Data Sheet
        if any(term in text_lower for term in ['safety data sheet', 'sds', 'msds', 'haccp']):
            return "FOOD_SDS"
        
        # Nutritional Information
        if any(term in text_lower for term in ['nutrition facts', 'nutritional information', 'vitamin', 'mineral']):
            return "NUTRITIONAL_INFO"
        
        # Ingredient Specification
        if any(term in text_lower for term in ['ingredient specification', 'food grade', 'additive']):
            return "INGREDIENT_SPEC"
        
        # Certificate of Analysis
        if any(term in text_lower for term in ['certificate of analysis', 'coa', 'microbiological']):
            return "COA"
        
        # Allergen Declaration
        if any(term in text_lower for term in ['allergen', 'contains', 'may contain', 'free from']):
            return "ALLERGEN_DECLARATION"
        
        # Regulatory Compliance
        if any(term in text_lower for term in ['gras', 'fda approved', 'efsa', 'regulation']):
            return "REGULATORY_COMPLIANCE"
        
        return "FOOD_DATASHEET"
    
    def _extract_nutritional_information(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive nutritional information."""
        nutritional_info = {
            "vitamins": [],
            "minerals": [],
            "macronutrients": [],
            "calories": None,
            "serving_size": None
        }
        
        # Extract vitamins
        vitamin_matches = self.food_patterns["vitamin_content"].findall(text)
        for vitamin, amount, unit in vitamin_matches:
            nutritional_info["vitamins"].append({
                "vitamin": vitamin,
                "amount": float(amount),
                "unit": unit
            })
        
        # Extract minerals
        mineral_matches = self.food_patterns["mineral_content"].findall(text)
        for mineral, amount, unit in mineral_matches:
            nutritional_info["minerals"].append({
                "mineral": mineral,
                "amount": float(amount),
                "unit": unit
            })
        
        # Extract caloric value
        calorie_matches = self.food_patterns["caloric_value"].findall(text)
        if calorie_matches:
            amount, unit = calorie_matches[0]
            nutritional_info["calories"] = {
                "amount": float(amount),
                "unit": unit
            }
        
        return nutritional_info
    
    def _extract_allergen_information(self, text: str) -> Dict[str, Any]:
        """Extract allergen declarations and free-from claims."""
        allergen_info = {
            "contains": [],
            "may_contain": [],
            "free_from": [],
            "allergen_statement": None
        }
        
        # Extract allergen declarations
        allergen_matches = self.food_patterns["allergen_declaration"].findall(text)
        for allergen_text in allergen_matches:
            allergens = [a.strip() for a in allergen_text.split(',')]
            if "may contain" in text.lower():
                allergen_info["may_contain"].extend(allergens)
            else:
                allergen_info["contains"].extend(allergens)
        
        # Extract allergen-free claims
        free_matches = self.food_patterns["allergen_free"].findall(text)
        allergen_info["free_from"] = list(set(free_matches))
        
        return allergen_info
    
    def _extract_food_regulatory_information(self, text: str) -> Dict[str, Any]:
        """Extract food industry regulatory information."""
        regulatory_info = {
            "gras_status": False,
            "fda_approved": False,
            "efsa_approved": False,
            "e_numbers": [],
            "ins_numbers": [],
            "certifications": []
        }
        
        # Check GRAS status
        if self.food_patterns["gras_status"].search(text):
            regulatory_info["gras_status"] = True
        
        # Check FDA approval
        if re.search(r'\bfda\s+approved\b', text, re.IGNORECASE):
            regulatory_info["fda_approved"] = True
        
        # Check EFSA approval
        if re.search(r'\befsa\s+approved\b', text, re.IGNORECASE):
            regulatory_info["efsa_approved"] = True
        
        # Extract E-numbers
        e_numbers = self.food_patterns["e_number"].findall(text)
        regulatory_info["e_numbers"] = [f"E{num}" for num in e_numbers]
        
        # Extract INS numbers
        ins_numbers = self.food_patterns["ins_number"].findall(text)
        regulatory_info["ins_numbers"] = [f"INS{num}" for num in ins_numbers]
        
        # Extract certifications
        cert_patterns = [
            r'\b(kosher|halal|organic|non-gmo)\s+certified\b',
            r'\b(haccp|brc|sqf|ifs|fssc)\s+(?:certified|approved)?\b'
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            regulatory_info["certifications"].extend(matches)
        
        return regulatory_info
    
    def _extract_food_storage_information(self, text: str) -> Dict[str, Any]:
        """Extract food-specific storage conditions."""
        storage_info = {
            "temperature": None,
            "humidity": None,
            "shelf_life": None,
            "special_conditions": []
        }
        
        # Extract storage temperature
        temp_match = self.food_patterns["storage_temp"].search(text)
        if temp_match:
            storage_info["temperature"] = temp_match.group(1) + "°C"
        
        # Extract shelf life
        shelf_match = self.food_patterns["shelf_life"].search(text)
        if shelf_match:
            storage_info["shelf_life"] = f"{shelf_match.group(1)} {shelf_match.group(2)}"
        
        # Extract special storage conditions
        special_conditions = []
        if re.search(r'\brefrigerat', text, re.IGNORECASE):
            special_conditions.append("refrigerated")
        if re.search(r'\bfreez', text, re.IGNORECASE):
            special_conditions.append("frozen")
        if re.search(r'\bdry\s+place\b', text, re.IGNORECASE):
            special_conditions.append("dry storage")
        if re.search(r'\bprotect\s+from\s+light\b', text, re.IGNORECASE):
            special_conditions.append("light-protected")
        
        storage_info["special_conditions"] = special_conditions
        
        return storage_info
    
    def _extract_tables_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables from text using basic pattern matching."""
        tables = []
        
        # Split text into lines for analysis
        lines = text.split('\n')
        current_table = []
        in_table = False
        table_id = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect potential table rows (lines with multiple separators)
            separators = ['|', '\t', '  ', ',']
            separator_count = max(line.count(sep) for sep in separators)
            
            # If line has multiple separators, might be a table row
            if separator_count >= 2 and len(line.split()) >= 3:
                if not in_table:
                    in_table = True
                    table_id += 1
                current_table.append(line)
            else:
                # End of table if we were in one
                if in_table and current_table:
                    # Process the completed table
                    table = self._process_table_lines(current_table, table_id)
                    if table:
                        tables.append(table)
                    current_table = []
                in_table = False
        
        # Handle last table if file ends while in table
        if current_table:
            table = self._process_table_lines(current_table, table_id)
            if table:
                tables.append(table)
        
        return tables
    
    def _process_table_lines(self, table_lines: List[str], table_id: int) -> Optional[Dict[str, Any]]:
        """Process a list of table lines into a structured table."""
        if len(table_lines) < 2:  # Need at least header and one data row
            return None
        
        # Determine the best separator
        separators = ['|', '\t', ',', '  ']
        best_separator = None
        max_consistency = 0
        
        for sep in separators:
            # Check consistency of column count across rows
            column_counts = [len(line.split(sep)) for line in table_lines]
            if len(set(column_counts)) == 1 and column_counts[0] > 1:  # All rows have same column count
                consistency = column_counts[0]
                if consistency > max_consistency:
                    max_consistency = consistency
                    best_separator = sep
        
        if not best_separator:
            return None
        
        # Parse the table
        rows = []
        for line in table_lines:
            row = [cell.strip() for cell in line.split(best_separator) if cell.strip()]
            if row:  # Only add non-empty rows
                rows.append(row)
        
        if len(rows) < 2:
            return None
        
        # First row is typically headers
        headers = rows[0]
        data_rows = rows[1:]
        
        return {
            "id": f"table_{table_id}",
            "headers": headers,
            "rows": data_rows,
            "row_count": len(data_rows),
            "column_count": len(headers),
            "table_type": "extracted",
            "separator_used": best_separator
        }
    
    def _calculate_food_quality_metrics(self, entities: Dict[str, List[FoodEntity]], 
                                       relationships: List[ExtractedRelationship], text: str) -> Dict[str, float]:
        """Calculate quality metrics specific to food industry documents."""
        metrics = {}
        
        # Entity extraction completeness for food industry
        food_entity_types = ["FOOD_INGREDIENT", "FOOD_ADDITIVE_TYPE", "FOOD_APPLICATION", 
                           "NUTRITIONAL_COMPONENT", "ALLERGEN_INFO"]
        extracted_types = set(entities.keys())
        food_types_found = len(extracted_types.intersection(food_entity_types))
        metrics["food_entity_coverage"] = food_types_found / len(food_entity_types)
        
        # Regulatory compliance completeness
        regulatory_keywords = ["gras", "fda", "efsa", "e-number", "haccp", "certified"]
        regulatory_mentions = sum(1 for keyword in regulatory_keywords 
                                if keyword in text.lower())
        metrics["regulatory_completeness"] = min(regulatory_mentions / 3, 1.0)
        
        # Allergen information completeness
        allergen_keywords = ["allergen", "contains", "may contain", "free from"]
        allergen_mentions = sum(1 for keyword in allergen_keywords 
                              if keyword in text.lower())
        metrics["allergen_completeness"] = min(allergen_mentions / 2, 1.0)
        
        # Overall food industry relevance
        food_keywords = ["food", "ingredient", "additive", "nutrition", "preservation"]
        food_mentions = sum(1 for keyword in food_keywords 
                          if keyword in text.lower())
        metrics["food_industry_relevance"] = min(food_mentions / 3, 1.0)
        
        return metrics
    
    def _food_entity_to_dict(self, entity: FoodEntity) -> Dict[str, Any]:
        """Convert FoodEntity to dictionary representation."""
        base_dict = self._entity_to_dict(entity)
        
        # Add food-specific fields
        base_dict.update({
            "allergen_info": entity.allergen_info,
            "nutritional_value": entity.nutritional_value,
            "regulatory_status": entity.regulatory_status,
            "food_grade": entity.food_grade,
            "shelf_life": entity.shelf_life
        })
        
        return base_dict
    
    def _find_matching_entity(self, text: str, entities: Dict[str, List[FoodEntity]]) -> Optional[FoodEntity]:
        """Find entity that matches the given text."""
        text_lower = text.lower().strip()
        
        for entity_list in entities.values():
            for entity in entity_list:
                if (entity.text.lower() in text_lower or 
                    text_lower in entity.text.lower() or
                    entity.normalized_value and entity.normalized_value.lower() in text_lower):
                    return entity
        
        return None 