"""
Compound Entity Extractor for Enhanced Legal AI System
Addresses over-granular extraction by recognizing meaningful compound entities
Based on 2024-2025 NER research findings for attention-based compound recognition
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass
from collections import defaultdict
import json

# Optional spaCy imports with fallback handling
try:
    import spacy
    from spacy.matcher import Matcher
    from spacy.tokens import Doc, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    Matcher = None
    Doc = None
    Span = None

logger = logging.getLogger(__name__)

@dataclass
class CompoundEntity:
    """Represents a compound entity with preserved relationships"""
    id: str
    value: str
    components: List[Dict[str, Any]]
    entity_type: str
    domain: str
    confidence: float
    start_pos: int
    end_pos: int
    technical_properties: Dict[str, Any]
    relationships: List[str]
    context: str
    pattern_type: str

@dataclass
class TechnicalSpecEntity:
    """Represents a complete technical specification as a single entity"""
    property: str
    value: str
    unit: str
    qualifier: str
    full_context: str
    confidence: float
    start_pos: int
    end_pos: int
    domain_specific_attributes: Dict[str, Any]

@dataclass
class ChemicalCompoundEntity:
    """Represents a chemical compound with its relationships preserved"""
    pattern_type: str
    full_match: str
    components: List[Dict[str, Any]]
    relationships: List[str]
    confidence: float
    cas_number: Optional[str]
    molecular_formula: Optional[str]
    compound_type: str

class PharmaceuticalCompoundAnalyzer:
    """Analyzes pharmaceutical text for compound patterns"""
    
    def __init__(self):
        self.spec_patterns = [
            # pH specifications
            r'(pH):\s*([0-9.,\-]+)(?:\s*[-–to]\s*([0-9.,\-]+))?\s*(?:\(([^)]+)\))?',
            # Loss on drying
            r'(Loss on drying):\s*([0-9.,]+)%?\s*(max|maximum|typical)?\s*(?:at\s+([0-9°C\s]+))?',
            # Density specifications
            r'(Bulk density|Tapped density):\s*([0-9.,\-]+)(?:\s*[-–to]\s*([0-9.,\-]+))?\s*(g/cm³|g/ml)',
            # Particle size
            r'(Particle size)\s*(?:\([^)]+\))?\s*:\s*([0-9.,\-]+)(?:\s*[-–to]\s*([0-9.,\-]+))?\s*(μm|microns)',
            # Composition with percentages
            r'Composition:\s*(\d+%)\s+([A-Za-z\s]+)\s*\+\s*(\d+%)\s+([A-Za-z\s\d]+)',
            # Heavy metals
            r'(Heavy metals):\s*([0-9.,]+)\s*(ppm|mg/kg)\s*(max|maximum)?',
            # Microbial count
            r'(Total aerobic count):\s*([0-9.,]+)\s*(CFU/g)\s*(max|maximum)?',
            # Assay specifications
            r'(Assay):\s*([0-9.,\-]+%?)(?:\s+\([^)]+\))?',
        ]
        
    def identify_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Identify pharmaceutical compound patterns in text"""
        patterns = []
        
        pattern_types = [
            'ph_specification', 'moisture_specification', 'density_specification', 
            'particle_size', 'composition', 'impurity_specification', 'microbial_specification', 'assay_specification'
        ]
        
        for i, pattern in enumerate(self.spec_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                patterns.append({
                    'pattern_id': i,
                    'match': match,
                    'type': pattern_types[i] if i < len(pattern_types) else 'specification',
                    'confidence': 0.9,
                    'full_text': match.group(0),
                    'groups': match.groups()
                })
        
        return patterns

class LegalCompoundAnalyzer:
    """Analyzes legal text for compound clause patterns"""
    
    def __init__(self):
        self.legal_patterns = [
            # Legal clauses with conditions - more flexible matching
            r'(shall|must|may)\s+([^.]{10,150}?)(?:\s+(provided that|unless|except|within))',
            # Obligation patterns - enhanced to catch more variations
            r'(Licensee|Party|Licensor|Company)\s+(shall|must|will|agrees to)\s+([^.]{15,150})',
            # Liability clauses - broader pattern
            r'(Liability|liable|responsible|damages?)\s+(?:shall\s+)?(?:not\s+)?(?:exceed|for)\s+([^.]{10,100})',
            # Time-based requirements
            r'(within\s+\d+\s+days?|before|after)\s+(?:of\s+)?([^.]{10,80})',
            # Performance requirements
            r'(Performance|standards?|requirements?)\s+(?:shall\s+)?(?:be\s+)?(?:maintained\s+)?(?:at\s+)?([^.]{10,100})',
            # Financial specifications
            r'(\$[\d,]+(?:\.\d{2})?)\s+([^.]{5,50})',
            # Definition patterns
            r'"([^"]+)"\s+means\s+([^.]+)',
        ]
        
    def identify_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Identify legal compound patterns in text"""
        patterns = []
        
        pattern_types = [
            'condition', 'obligation', 'liability', 'time_requirement', 
            'performance_requirement', 'financial_specification', 'definition'
        ]
        
        for i, pattern in enumerate(self.legal_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                patterns.append({
                    'pattern_id': i,
                    'match': match,
                    'type': pattern_types[i] if i < len(pattern_types) else 'legal_clause',
                    'confidence': 0.8,
                    'full_text': match.group(0),
                    'groups': match.groups()
                })
        
        return patterns

class FoodIndustryCompoundAnalyzer:
    """Analyzes food industry text for compound patterns and specifications"""
    
    def __init__(self):
        self.food_patterns = [
            # Nutritional information with values and units
            r'(Protein|Fat|Carbohydrate|Fiber|Sugar|Sodium|Calories|Energy):\s*([0-9.,]+)\s*(g|mg|kcal|kJ|%)\s*(?:per\s+([0-9]+g|100g|serving))?',
            
            # Ingredient compositions with percentages
            r'(Ingredients?):\s*([^.]{20,200})\s*(?:\(([^)]+)\))?',
            
            # Food safety specifications
            r'(Shelf life|Best before|Expiry):\s*([0-9]+)\s*(days?|months?|years?)\s*(?:from\s+([^.]+))?',
            
            # Temperature and storage conditions
            r'(Storage|Store|Keep)\s+(?:at\s+)?([^.]{10,80})\s*(?:at\s+([0-9\-°C\s]+))?',
            
            # Microbiological standards - enhanced for separate detection
            r'(Total plate count|Total viable count|E\.?\s*coli|Salmonella|Listeria|Yeast and molds?|Yeast|Mold):\s*([0-9.,<>]+|Absent)\s*(CFU/g|CFU/ml|per gram|per 25g)?\s*(?:(max|maximum))?',
            
            # Allergen information
            r'(Contains?|May contain|Allergens?):\s*([^.]{10,150})',
            
            # Physical properties
            r'(pH|Water activity|Moisture|Viscosity|Density):\s*([0-9.,\-]+)(?:\s*[-–to]\s*([0-9.,\-]+))?\s*(?:([%°C\w/]+))?',
            
            # Processing parameters
            r'(Processing temperature|Cooking time|Pasteurization):\s*([0-9.,\-]+)\s*([°C°F\s]*)\s*(?:for\s+([0-9.,]+)\s*(minutes?|hours?|seconds?))?',
            
            # Packaging specifications
            r'(Net weight|Volume|Package size):\s*([0-9.,]+)\s*(g|kg|ml|L|oz|lb)',
            
            # Quality standards and certifications
            r'(Organic|Non-GMO|Gluten-free|Kosher|Halal|FDA approved|USDA certified)\s*(?::\s*([^.]+))?',
            
            # Chemical additives and preservatives
            r'(Preservatives?|Additives?|E[0-9]+|BHA|BHT|Sorbic acid|Benzoic acid):\s*([^.]{5,100})',
            
            # Serving and portion information
            r'(Serving size|Portion):\s*([0-9.,]+)\s*(g|ml|cups?|pieces?|slices?)',
        ]
        
    def identify_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Identify food industry compound patterns in text"""
        patterns = []
        
        pattern_types = [
            'nutritional_specification', 'ingredient_composition', 'shelf_life_specification',
            'storage_condition', 'microbiological_standard', 'allergen_information',
            'physical_property', 'processing_parameter', 'packaging_specification',
            'quality_certification', 'chemical_additive', 'serving_specification'
        ]
        
        for i, pattern in enumerate(self.food_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                patterns.append({
                    'pattern_id': i,
                    'match': match,
                    'type': pattern_types[i] if i < len(pattern_types) else 'food_specification',
                    'confidence': 0.9,
                    'full_text': match.group(0),
                    'groups': match.groups()
                })
        
        return patterns

class AgriculturalCompoundAnalyzer:
    """Analyzes agricultural text for compound patterns (farming/crop production)"""
    
    def __init__(self):
        self.agro_patterns = [
            # Application rates with crops - more flexible
            r'(?:Application rate:\s*)?(\d+(?:\.\d+)?)\s*(kg/ha|L/ha|g/L)\s+(?:of\s+)?([A-Za-z\s]+)(?:\s+(?:on|for)\s+([A-Za-z\s]+))?',
            # Efficacy statements - enhanced pattern
            r'(?:Efficacy:\s*)?(?:Product\s+)?([A-Za-z\s]*)\s*(?:increases|improves|reduces)\s+([A-Za-z\s]+)\s+by\s+(\d+(?:\.\d+)?%)',
            # Chemical compositions - broader matching
            r'(?:Chemical composition:\s*)?([A-Za-z\s]+)\s*(?:\(|:)\s*(\d+(?:\.\d+)?%)\s+([A-Za-z\s]+)(?:\)|$)',
            # Safety intervals - more flexible
            r'(?:Safety interval:\s*)?(\d+)\s*(days?|hours?)\s+(?:before|after)\s+([A-Za-z\s]+)(?:\s+for\s+([A-Za-z\s]+))?',
            # Alternative efficacy pattern
            r'(?:Efficacy|Effect):\s*([^.]{10,100})',
            # Alternative application pattern  
            r'(?:Apply|Application):\s*([^.]{10,100})',
        ]
        
    def identify_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Identify agricultural compound patterns in text"""
        patterns = []
        
        pattern_types = [
            'application', 'efficacy', 'composition', 'safety', 
            'efficacy', 'application'
        ]
        
        for i, pattern in enumerate(self.agro_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                patterns.append({
                    'pattern_id': i,
                    'match': match,
                    'type': pattern_types[i] if i < len(pattern_types) else 'agricultural_spec',
                    'confidence': 0.85,
                    'full_text': match.group(0),
                    'groups': match.groups()
                })
        
        return patterns

class AttentionBasedNER:
    """Simulated attention-based NER for compound entity extraction"""
    
    def __init__(self):
        # Load spaCy model for basic NLP operations
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            logger.warning("spaCy not available. Install with: pip install spacy>=3.7.0")
            
    def extract_compounds(self, text: str, domain_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract compound entities using attention-like mechanisms"""
        compounds = []
        
        if not self.nlp:
            # Fallback to pattern-based extraction
            return self._pattern_based_extraction(text, domain_patterns)
        
        doc = self.nlp(text)
        
        # Identify compound entities based on patterns and NLP
        for pattern in domain_patterns:
            match = pattern['match']
            start_pos = match.start()
            end_pos = match.end()
            
            # Find corresponding spaCy span
            span = None
            for token in doc:
                if token.idx <= start_pos < token.idx + len(token.text):
                    # Find the span that covers the pattern
                    end_token = token
                    for t in doc[token.i:]:
                        if t.idx + len(t.text) >= end_pos:
                            end_token = t
                            break
                    span = doc[token.i:end_token.i + 1]
                    break
            
            if span:
                compound = {
                    'text': pattern['full_text'],
                    'span': span,
                    'pattern_type': pattern['type'],
                    'confidence': pattern['confidence'],
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'entities': self._extract_entities_from_span(span),
                    'relationships': self._extract_relationships_from_span(span, doc)
                }
                compounds.append(compound)
        
        return compounds
    
    def _pattern_based_extraction(self, text: str, domain_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback pattern-based extraction when spaCy is not available"""
        compounds = []
        
        for pattern in domain_patterns:
            match = pattern['match']
            compound = {
                'text': pattern['full_text'],
                'span': None,
                'pattern_type': pattern['type'],
                'confidence': pattern['confidence'],
                'start_pos': match.start(),
                'end_pos': match.end(),
                'entities': [{'text': g, 'label': 'COMPOUND_PART'} for g in pattern['groups'] if g],
                'relationships': []
            }
            compounds.append(compound)
        
        return compounds
    
    def _extract_entities_from_span(self, span) -> List[Dict[str, str]]:
        """Extract named entities from a spaCy span"""
        entities = []
        if hasattr(span, 'ents'):
            for ent in span.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        return entities
    
    def _extract_relationships_from_span(self, span, doc) -> List[str]:
        """Extract relationships within and around the span"""
        relationships = []
        
        # Look for dependency relationships
        if hasattr(span, '__iter__'):
            for token in span:
                if hasattr(token, 'dep_') and token.dep_ in ['nmod', 'amod', 'compound', 'appos']:
                    relationships.append(f"{token.text} --{token.dep_}--> {token.head.text}")
        
        return relationships

class CompoundPatternMatcher:
    """Pattern matcher for compound entities"""
    
    def __init__(self):
        self.technical_spec_patterns = {
            'measurement_range': r'([A-Za-z\s]+):\s*([0-9.,\-]+)\s*(?:to|-)\s*([0-9.,\-]+)\s*([%°C\w/]+)',
            'single_measurement': r'([A-Za-z\s]+):\s*([0-9.,\-]+)\s*([%°C\w/]+)(?:\s+(max|min|typical))?',
            'composition': r'(\d+%)\s+([A-Za-z\s]+)(?:\s*\+\s*(\d+%)\s+([A-Za-z\s]+))?',
            'grade_spec': r'([A-Za-z\s]+)\s+(grade|type|class)\s+([A-Za-z0-9\s]+)',
        }
        
    def match_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Match compound patterns in text"""
        matches = []
        
        for pattern_name, pattern in self.technical_spec_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append({
                    'pattern_name': pattern_name,
                    'match': match,
                    'confidence': 0.9,
                    'groups': match.groups()
                })
        
        return matches

class CompoundEntityExtractor:
    """
    Advanced compound entity extraction using attention-based mechanisms
    Addresses over-granular extraction by recognizing meaningful units
    """
    
    def __init__(self, model_path: str = None):
        self.attention_model = AttentionBasedNER()
        self.compound_patterns = CompoundPatternMatcher()
        self.domain_analyzers = {
            'pharmaceutical': PharmaceuticalCompoundAnalyzer(),
            'legal': LegalCompoundAnalyzer(),
            'food_industry': FoodIndustryCompoundAnalyzer(),
            'agricultural': AgriculturalCompoundAnalyzer()
        }
        
    def extract_compound_entities(self, text: str, domain: str = 'general') -> List[CompoundEntity]:
        """Extract meaningful compound entities instead of fragments"""
        
        # 1. Domain-specific compound pattern recognition
        if domain in self.domain_analyzers:
            domain_patterns = self.domain_analyzers[domain].identify_patterns(text)
        else:
            domain_patterns = self.compound_patterns.match_patterns(text)
        
        # 2. Attention-based sequence labeling for compound extraction
        attention_entities = self.attention_model.extract_compounds(text, domain_patterns)
        
        # 3. Multi-level feature fusion for hierarchical entities
        hierarchical_entities = self._fuse_hierarchical_features(attention_entities)
        
        # 4. Relation-aware entity grouping
        compound_entities = self._group_related_entities(hierarchical_entities, domain)
        
        return compound_entities
    
    def _fuse_hierarchical_features(self, attention_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fuse multi-level features for hierarchical entity recognition"""
        
        hierarchical = []
        
        for entity in attention_entities:
            # Add hierarchical structure
            entity['hierarchy'] = self._build_entity_hierarchy(entity)
            entity['semantic_level'] = self._determine_semantic_level(entity)
            entity['context_features'] = self._extract_context_features(entity)
            
            hierarchical.append(entity)
        
        return hierarchical
    
    def _build_entity_hierarchy(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Build hierarchical structure for entity"""
        hierarchy = {
            'main_concept': entity.get('text', ''),
            'sub_concepts': [],
            'attributes': {},
            'modifiers': []
        }
        
        # Extract sub-concepts from pattern groups or entities
        if 'groups' in entity:
            for i, group in enumerate(entity['groups']):
                if group:
                    if re.match(r'\d+', group):
                        hierarchy['attributes'][f'value_{i}'] = group
                    elif re.match(r'[A-Za-z\s]+', group):
                        hierarchy['sub_concepts'].append(group.strip())
        elif 'entities' in entity:
            # Extract from entities field (pattern-based extraction)
            for i, ent in enumerate(entity['entities']):
                if ent and ent.get('text'):
                    text = ent['text']
                    if re.match(r'\d+', text):
                        hierarchy['attributes'][f'value_{i}'] = text
                    elif re.match(r'[A-Za-z\s]+', text):
                        hierarchy['sub_concepts'].append(text.strip())
        
        return hierarchy
    
    def _determine_semantic_level(self, entity: Dict[str, Any]) -> str:
        """Determine the semantic level of the entity"""
        text = entity.get('text', '').lower()
        
        if any(word in text for word in ['specification', 'property', 'characteristic']):
            return 'technical_specification'
        elif any(word in text for word in ['composition', 'formula', 'mixture']):
            return 'chemical_composition'
        elif any(word in text for word in ['application', 'use', 'method']):
            return 'application_method'
        elif any(word in text for word in ['safety', 'hazard', 'warning']):
            return 'safety_information'
        else:
            return 'general_entity'
    
    def _extract_context_features(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contextual features around the entity"""
        return {
            'pattern_type': entity.get('pattern_type', 'unknown'),
            'confidence': entity.get('confidence', 0.5),
            'entity_count': len(entity.get('entities', [])),
            'relationship_count': len(entity.get('relationships', [])),
            'has_measurements': self._has_measurements(entity.get('text', '')),
            'has_percentages': self._has_percentages(entity.get('text', ''))
        }
    
    def _has_measurements(self, text: str) -> bool:
        """Check if text contains measurement values"""
        return bool(re.search(r'\d+(?:\.\d+)?\s*[%°C\w/]+', text))
    
    def _has_percentages(self, text: str) -> bool:
        """Check if text contains percentage values"""
        return bool(re.search(r'\d+(?:\.\d+)?%', text))
    
    def _group_related_entities(self, hierarchical_entities: List[Dict[str, Any]], domain: str) -> List[CompoundEntity]:
        """Group related entities into compound entities"""
        
        compound_entities = []
        
        for i, entity in enumerate(hierarchical_entities):
            # Create compound entity - use pattern type as entity type for domain-specific patterns
            entity_type = entity.get('pattern_type', entity.get('semantic_level', 'general'))
            
            compound = CompoundEntity(
                id=f"compound_{domain}_{i}",
                value=entity.get('text', ''),
                components=self._extract_components(entity),
                entity_type=entity_type,
                domain=domain,
                confidence=entity.get('confidence', 0.5),
                start_pos=entity.get('start_pos', 0),
                end_pos=entity.get('end_pos', 0),
                technical_properties=self._extract_technical_properties(entity),
                relationships=entity.get('relationships', []),
                context=entity.get('text', ''),
                pattern_type=entity.get('pattern_type', 'unknown')
            )
            
            compound_entities.append(compound)
        
        return compound_entities
    
    def _extract_components(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract components from hierarchical entity"""
        components = []
        
        hierarchy = entity.get('hierarchy', {})
        
        # Add sub-concepts as components
        for concept in hierarchy.get('sub_concepts', []):
            components.append({
                'type': 'sub_concept',
                'value': concept,
                'role': 'component'
            })
        
        # Add attributes as components
        for key, value in hierarchy.get('attributes', {}).items():
            components.append({
                'type': 'attribute',
                'value': value,
                'role': key
            })
        
        # If no components found, try to extract from pattern text directly
        if not components and entity.get('text'):
            text = entity.get('text', '')
            pattern_type = entity.get('pattern_type', '')
            
            # For legal entities without components, extract key parts
            if pattern_type in ['condition', 'performance_requirement']:
                # Extract numerical values
                numbers = re.findall(r'\d+(?:\.\d+)?', text)
                for num in numbers:
                    components.append({
                        'type': 'numerical_value',
                        'value': num,
                        'role': 'extracted_value'
                    })
                
                # Extract key words (more than 3 characters)
                words = re.findall(r'\b[A-Za-z]{4,}\b', text)[:3]  # Limit to 3 key words
                for word in words:
                    if word.lower() not in ['shall', 'must', 'will', 'that', 'such', 'this']:
                        components.append({
                            'type': 'key_term',
                            'value': word,
                            'role': 'key_concept'
                        })
        
        return components
    
    def _extract_technical_properties(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical properties from entity"""
        context_features = entity.get('context_features', {})
        
        return {
            'has_measurements': context_features.get('has_measurements', False),
            'has_percentages': context_features.get('has_percentages', False),
            'pattern_type': context_features.get('pattern_type', 'unknown'),
            'semantic_level': entity.get('semantic_level', 'general'),
            'entity_complexity': context_features.get('entity_count', 0) + context_features.get('relationship_count', 0)
        } 