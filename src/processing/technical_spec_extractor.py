"""
Technical Specification Extractor for Enhanced Legal AI System
Specialized extractor for technical specifications as complete units
Prevents fragmentation of property-value-context relationships
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

logger = logging.getLogger(__name__)

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
    specification_type: str
    related_properties: List[str]
    measurement_range: Optional[Tuple[str, str]]

@dataclass
class CompoundSpecification:
    """Represents a compound specification with multiple related properties"""
    main_property: str
    specifications: List[TechnicalSpecEntity]
    compound_context: str
    confidence: float
    specification_family: str
    preservation_score: float

class PharmaceuticalSpecExtractor:
    """Extract pharmaceutical technical specifications as unified entities"""
    
    def __init__(self):
        self.pharmaceutical_patterns = {
            # pH specifications with ranges
            'ph_specification': r'(pH):\s*([0-9.,]+)(?:\s*[-–to]\s*([0-9.,]+))?\s*(?:\(([^)]+)\))?',
            
            # Moisture/Loss on drying with conditions
            'moisture_specification': r'(Loss on drying|Moisture content|Water content):\s*([0-9.,]+)%?\s*(max|maximum|typical|NMT)?\s*(?:at\s+([0-9°C\s]+))?',
            
            # Density specifications
            'density_specification': r'(Bulk density|Tapped density|Apparent density):\s*([0-9.,]+)(?:\s*[-–to]\s*([0-9.,]+))?\s*(g/cm³|g/ml|kg/m³)',
            
            # Particle size with distribution
            'particle_size': r'(Particle size|D50|D90|Mean particle size):\s*([0-9.,]+)(?:\s*[-–to]\s*([0-9.,]+))?\s*(μm|microns|nm)\s*(?:\(([^)]+)\))?',
            
            # Assay specifications
            'assay_specification': r'(Assay|Content|Purity):\s*([0-9.,]+)(?:\s*[-–to]\s*([0-9.,]+))?\s*%?\s*(w/w|w/v|on dry basis)?',
            
            # Chemical composition
            'composition_specification': r'Composition:\s*(\d+%)\s+([A-Za-z\s]+)\s*\+\s*(\d+%)\s+([A-Za-z\s\d]+)',
            
            # Heavy metals and impurities
            'impurity_specification': r'(Heavy metals|Residual solvents|[A-Za-z\s]+impurities?):\s*([0-9.,]+)\s*(ppm|ppb|mg/kg|%)\s*(max|maximum|NMT)?',
            
            # Microbiological specifications
            'microbial_specification': r'(Total aerobic count|Yeast and molds|E\. coli|Salmonella):\s*([0-9.,]+|Absent)\s*(CFU/g|per gram)?',
        }
        
        self.context_indicators = [
            'specification', 'property', 'characteristic', 'parameter',
            'requirement', 'limit', 'range', 'typical', 'nominal'
        ]
        
    def extract_pharmaceutical_specs(self, text: str) -> List[TechnicalSpecEntity]:
        """Extract pharmaceutical specifications as complete entities"""
        specs = []
        
        for spec_type, pattern in self.pharmaceutical_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                spec = self._create_pharmaceutical_spec(match, spec_type, text)
                if spec:
                    specs.append(spec)
        
        # Group related specifications
        grouped_specs = self._group_related_specifications(specs)
        
        return grouped_specs
    
    def _create_pharmaceutical_spec(self, match: re.Match, spec_type: str, full_text: str) -> Optional[TechnicalSpecEntity]:
        """Create a pharmaceutical specification entity from match"""
        groups = match.groups()
        
        if len(groups) < 2:
            return None
        
        property_name = groups[0].strip()
        value = groups[1].strip()
        
        # Extract additional components based on pattern type
        unit = ""
        qualifier = ""
        range_end = None
        
        if spec_type == 'ph_specification':
            unit = "pH units"
            range_end = groups[2] if len(groups) > 2 and groups[2] else None
            qualifier = groups[3] if len(groups) > 3 and groups[3] else ""
            
        elif spec_type == 'moisture_specification':
            unit = "%"
            qualifier = groups[2] if len(groups) > 2 and groups[2] else ""
            condition = groups[3] if len(groups) > 3 and groups[3] else ""
            if condition:
                qualifier += f" at {condition}"
                
        elif spec_type == 'density_specification':
            unit = groups[3] if len(groups) > 3 else ""
            range_end = groups[2] if len(groups) > 2 and groups[2] else None
            
        elif spec_type == 'particle_size':
            unit = groups[3] if len(groups) > 3 else "μm"
            range_end = groups[2] if len(groups) > 2 and groups[2] else None
            qualifier = groups[4] if len(groups) > 4 and groups[4] else ""
            
        elif spec_type == 'assay_specification':
            unit = "%"
            range_end = groups[2] if len(groups) > 2 and groups[2] else None
            qualifier = groups[3] if len(groups) > 3 and groups[3] else ""
            
        elif spec_type == 'composition_specification':
            # Handle compound composition differently
            return self._create_composition_spec(match, full_text)
            
        elif spec_type == 'impurity_specification':
            unit = groups[2] if len(groups) > 2 else ""
            qualifier = groups[3] if len(groups) > 3 and groups[3] else ""
            
        elif spec_type == 'microbial_specification':
            unit = groups[2] if len(groups) > 2 and groups[2] else ""
        
        # Build full context
        context_start = max(0, match.start() - 50)
        context_end = min(len(full_text), match.end() + 50)
        full_context = full_text[context_start:context_end].strip()
        
        # Calculate confidence based on completeness
        confidence = self._calculate_spec_confidence(property_name, value, unit, qualifier)
        
        return TechnicalSpecEntity(
            property=property_name,
            value=value,
            unit=unit,
            qualifier=qualifier,
            full_context=full_context,
            confidence=confidence,
            start_pos=match.start(),
            end_pos=match.end(),
            domain_specific_attributes={
                'specification_type': spec_type,
                'pattern_matched': match.group(0),
                'has_range': range_end is not None,
                'has_condition': bool(qualifier),
                'pharmaceutical_category': self._classify_pharmaceutical_spec(property_name)
            },
            specification_type=spec_type,
            related_properties=[],
            measurement_range=(value, range_end) if range_end else None
        )
    
    def _create_composition_spec(self, match: re.Match, full_text: str) -> TechnicalSpecEntity:
        """Create a composition specification preserving component relationships"""
        groups = match.groups()
        
        # Build compound composition description
        composition_parts = []
        if groups[0] and groups[1]:  # First component
            composition_parts.append(f"{groups[0]} {groups[1].strip()}")
        if groups[2] and groups[3]:  # Second component
            composition_parts.append(f"{groups[2]} {groups[3].strip()}")
        
        composition_value = " + ".join(composition_parts)
        
        # Extract context
        context_start = max(0, match.start() - 30)
        context_end = min(len(full_text), match.end() + 30)
        full_context = full_text[context_start:context_end].strip()
        
        return TechnicalSpecEntity(
            property="Composition",
            value=composition_value,
            unit="w/w",
            qualifier="pharmaceutical grade",
            full_context=full_context,
            confidence=0.95,
            start_pos=match.start(),
            end_pos=match.end(),
            domain_specific_attributes={
                'specification_type': 'composition_specification',
                'component_count': len([g for g in groups if g]),
                'has_grade_specification': any('type' in str(g).lower() for g in groups if g),
                'pharmaceutical_category': 'composition'
            },
            specification_type='composition_specification',
            related_properties=[groups[1].strip() if groups[1] else "", groups[3].strip() if groups[3] else ""],
            measurement_range=None
        )
    
    def _calculate_spec_confidence(self, property_name: str, value: str, unit: str, qualifier: str) -> float:
        """Calculate confidence score for specification completeness"""
        confidence = 0.5  # Base confidence
        
        # Property name quality
        if len(property_name) > 2:
            confidence += 0.2
        if any(indicator in property_name.lower() for indicator in ['ph', 'moisture', 'density', 'size', 'assay']):
            confidence += 0.1
        
        # Value quality
        if re.match(r'^\d+(?:\.\d+)?$', value):
            confidence += 0.2
        
        # Unit presence
        if unit:
            confidence += 0.15
        
        # Qualifier presence
        if qualifier:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _classify_pharmaceutical_spec(self, property_name: str) -> str:
        """Classify pharmaceutical specification type"""
        prop_lower = property_name.lower()
        
        if 'ph' in prop_lower:
            return 'chemical_property'
        elif any(word in prop_lower for word in ['moisture', 'loss', 'water']):
            return 'physical_property'
        elif any(word in prop_lower for word in ['density', 'bulk', 'tapped']):
            return 'physical_property'
        elif any(word in prop_lower for word in ['particle', 'size', 'd50', 'd90']):
            return 'particle_property'
        elif any(word in prop_lower for word in ['assay', 'content', 'purity']):
            return 'chemical_property'
        elif any(word in prop_lower for word in ['heavy', 'metal', 'impurity']):
            return 'impurity_property'
        elif any(word in prop_lower for word in ['microbial', 'bacteria', 'yeast']):
            return 'microbiological_property'
        else:
            return 'general_property'
    
    def _group_related_specifications(self, specs: List[TechnicalSpecEntity]) -> List[TechnicalSpecEntity]:
        """Group specifications that are related to preserve relationships"""
        # For now, return specs as-is, but mark related properties
        for i, spec in enumerate(specs):
            related = []
            for j, other_spec in enumerate(specs):
                if i != j and self._are_specifications_related(spec, other_spec):
                    related.append(other_spec.property)
            spec.related_properties = related
        
        return specs
    
    def _are_specifications_related(self, spec1: TechnicalSpecEntity, spec2: TechnicalSpecEntity) -> bool:
        """Check if two specifications are related"""
        # Same category
        if (spec1.domain_specific_attributes.get('pharmaceutical_category') == 
            spec2.domain_specific_attributes.get('pharmaceutical_category')):
            return True
        
        # Physical properties are often related
        physical_props = ['density', 'particle', 'moisture']
        if (any(prop in spec1.property.lower() for prop in physical_props) and
            any(prop in spec2.property.lower() for prop in physical_props)):
            return True
        
        # Close proximity in text
        if abs(spec1.start_pos - spec2.start_pos) < 200:
            return True
        
        return False

class LegalSpecExtractor:
    """Extract legal specifications and requirements as unified entities"""
    
    def __init__(self):
        self.legal_patterns = {
            # Legal requirements with conditions
            'requirement_specification': r'(shall|must|will)\s+([^.]{20,150})\s+(within\s+\d+\s+days?|by\s+[^.]+|provided that)',
            
            # Performance standards
            'performance_specification': r'(performance|standard|requirement)\s+(?:of\s+)?([^.]{10,100})\s+(?:shall\s+)?(?:be\s+)?([^.]+)',
            
            # Liability limits
            'liability_specification': r'(liability|damages?)\s+(?:shall\s+)?(?:not\s+)?(?:exceed\s+)?([^.]{10,80})',
            
            # Time specifications
            'time_specification': r'(within|not later than|before|after)\s+(\d+)\s+(days?|months?|years?)\s+(?:of\s+)?([^.]+)',
            
            # Financial specifications - enhanced to catch more patterns
            'financial_specification': r'(\$\d+(?:,\d{3})*(?:\.\d{2})?)\s*([^.]{0,100})',
        }
    
    def extract_legal_specs(self, text: str) -> List[TechnicalSpecEntity]:
        """Extract legal specifications as complete entities"""
        specs = []
        
        for spec_type, pattern in self.legal_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                spec = self._create_legal_spec(match, spec_type, text)
                if spec:
                    specs.append(spec)
        
        return specs
    
    def _create_legal_spec(self, match: re.Match, spec_type: str, full_text: str) -> Optional[TechnicalSpecEntity]:
        """Create a legal specification entity from match"""
        groups = match.groups()
        
        if len(groups) < 2:
            return None
        
        # Extract context
        context_start = max(0, match.start() - 100)
        context_end = min(len(full_text), match.end() + 100)
        full_context = full_text[context_start:context_end].strip()
        
        if spec_type == 'requirement_specification':
            property_name = "Legal Requirement"
            value = f"{groups[0]} {groups[1]}"
            qualifier = groups[2] if len(groups) > 2 else ""
            unit = "legal obligation"
            
        elif spec_type == 'performance_specification':
            property_name = groups[0].title()
            value = groups[1] if len(groups) > 1 else ""
            qualifier = groups[2] if len(groups) > 2 else ""
            unit = "performance standard"
            
        elif spec_type == 'liability_specification':
            property_name = groups[0].title()
            value = groups[1] if len(groups) > 1 else ""
            qualifier = ""
            unit = "liability terms"
            
        elif spec_type == 'time_specification':
            property_name = "Time Requirement"
            value = f"{groups[1]} {groups[2]}"
            qualifier = f"{groups[0]} {groups[3] if len(groups) > 3 else ''}"
            unit = "time period"
            # Use 'time_specification' for consistency with test expectations
            spec_type = 'time_specification'
            
        elif spec_type == 'financial_specification':
            property_name = "Financial Specification"
            value = groups[0]  # This will be the $1,000,000 part
            unit = "currency"
            qualifier = groups[1] if len(groups) > 1 and groups[1] else ""
        
        else:
            return None
        
        confidence = self._calculate_legal_confidence(property_name, value, qualifier)
        
        return TechnicalSpecEntity(
            property=property_name,
            value=value,
            unit=unit,
            qualifier=qualifier,
            full_context=full_context,
            confidence=confidence,
            start_pos=match.start(),
            end_pos=match.end(),
            domain_specific_attributes={
                'specification_type': spec_type,
                'legal_category': self._classify_legal_spec(property_name, value),
                'has_time_component': 'time' in spec_type,
                'has_financial_component': 'financial' in spec_type
            },
            specification_type=spec_type,
            related_properties=[],
            measurement_range=None
        )
    
    def _calculate_legal_confidence(self, property_name: str, value: str, qualifier: str) -> float:
        """Calculate confidence for legal specification"""
        confidence = 0.6  # Base confidence
        
        # Legal keywords increase confidence
        legal_keywords = ['shall', 'must', 'will', 'liability', 'obligation', 'requirement']
        if any(keyword in value.lower() for keyword in legal_keywords):
            confidence += 0.2
        
        # Specificity increases confidence
        if len(value) > 20:
            confidence += 0.1
        
        # Qualifier adds context
        if qualifier:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _classify_legal_spec(self, property_name: str, value: str) -> str:
        """Classify legal specification type"""
        combined_text = f"{property_name} {value}".lower()
        
        if any(word in combined_text for word in ['liability', 'damages', 'loss']):
            return 'liability_terms'
        elif any(word in combined_text for word in ['performance', 'standard', 'quality']):
            return 'performance_terms'
        elif any(word in combined_text for word in ['time', 'days', 'months', 'deadline']):
            return 'temporal_terms'
        elif any(word in combined_text for word in ['payment', 'fee', 'cost', '$', 'dollar']):
            return 'financial_terms'
        elif any(word in combined_text for word in ['obligation', 'duty', 'responsibility']):
            return 'obligation_terms'
        else:
            return 'general_terms'

class FoodIndustrySpecExtractor:
    """Extract food industry specifications and requirements as unified entities"""
    
    def __init__(self):
        self.food_patterns = {
            # Nutritional specifications with complete context
            'nutritional_specification': r'(Protein|Fat|Carbohydrate|Fiber|Sugar|Sodium|Calories|Energy):\s*([0-9.,]+)\s*(g|mg|kcal|kJ|%)\s*(?:per\s+([0-9]+g|100g|serving))?',
            
            # Microbiological standards with limits - enhanced for separate detection
            'microbiological_specification': r'(Total plate count|Total viable count|E\.?\s*coli|Salmonella|Listeria|Yeast and molds?|Yeast|Mold):\s*([0-9.,<>]+|Absent)\s*(CFU/g|CFU/ml|per gram|per 25g)?\s*(?:(max|maximum))?',
            
            # Physical property specifications
            'physical_specification': r'(pH|Water activity|Moisture|Viscosity|Density|Brix):\s*([0-9.,\-]+)(?:\s*[-–to]\s*([0-9.,\-]+))?\s*(?:([%°C\w/]+))?',
            
            # Processing parameter specifications
            'processing_specification': r'(Processing temperature|Cooking time|Pasteurization|Sterilization):\s*([0-9.,\-]+)\s*([°C°F\s]*)\s*(?:for\s+([0-9.,]+)\s*(minutes?|hours?|seconds?))?',
            
            # Shelf life and storage specifications - enhanced pattern
            'storage_specification': r'(Shelf life|Best before|Expiry|Storage):\s*([^.\n\r]{5,200})',
            
            # Packaging specifications
            'packaging_specification': r'(Net weight|Volume|Package size|Fill weight):\s*([0-9.,]+)\s*(g|kg|ml|L|oz|lb|fl oz)',
            
            # Ingredient composition with percentages
            'composition_specification': r'(Ingredients?):\s*([^.]{20,200})\s*(?:\(([^)]+)\))?',
            
            # Allergen specifications
            'allergen_specification': r'(Contains?|May contain|Allergens?|Free from):\s*([^.]{10,150})',
            
            # Quality certification specifications
            'certification_specification': r'(Organic|Non-GMO|Gluten-free|Kosher|Halal|FDA approved|USDA certified|BRC|SQF|HACCP)\s*(?::\s*([^.]+))?',
            
            # Sensory specifications
            'sensory_specification': r'(Color|Taste|Texture|Appearance|Odor|Flavor):\s*([^.]{10,100})',
            
            # Chemical additive specifications
            'additive_specification': r'(Preservatives?|Additives?|E[0-9]+|BHA|BHT|Sorbic acid|Benzoic acid|Natural flavors?):\s*([^.]{5,100})',
        }
    
    def extract_food_specs(self, text: str) -> List[TechnicalSpecEntity]:
        """Extract food industry specifications as complete entities"""
        specs = []
        
        for spec_type, pattern in self.food_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                spec = self._create_food_spec(match, spec_type, text)
                if spec:
                    specs.append(spec)
        
        return specs
    
    def _create_food_spec(self, match: re.Match, spec_type: str, full_text: str) -> Optional[TechnicalSpecEntity]:
        """Create a food industry specification entity from match"""
        groups = match.groups()
        
        if len(groups) < 2:
            return None
        
        property_name = groups[0].strip() if groups[0] else ""
        value = groups[1].strip() if groups[1] else ""
        unit = ""
        qualifier = ""
        
        # Extract context
        context_start = max(0, match.start() - 100)
        context_end = min(len(full_text), match.end() + 100)
        full_context = full_text[context_start:context_end].strip()
        
        if spec_type == 'nutritional_specification':
            unit = groups[2] if len(groups) > 2 and groups[2] else ""
            qualifier = groups[3] if len(groups) > 3 and groups[3] else "per 100g"
            
        elif spec_type == 'microbiological_specification':
            unit = groups[2] if len(groups) > 2 and groups[2] else ""
            qualifier = groups[3] if len(groups) > 3 and groups[3] else ""
            
        elif spec_type == 'physical_specification':
            unit = groups[3] if len(groups) > 3 and groups[3] else ""
            range_end = groups[2] if len(groups) > 2 and groups[2] else None
            if range_end:
                value = f"{value}-{range_end}"
                
        elif spec_type == 'processing_specification':
            unit = groups[2] if len(groups) > 2 and groups[2] else "°C"
            if len(groups) > 4 and groups[3] and groups[4]:
                qualifier = f"for {groups[3]} {groups[4]}"
                
        elif spec_type == 'storage_specification':
            # Handle both shelf life and storage conditions
            if any(word in property_name.lower() for word in ['shelf', 'expiry', 'before']):
                # Extract time units from value
                time_match = re.search(r'(\d+)\s*(days?|months?|years?)', value)
                if time_match:
                    value = time_match.group(1)
                    unit = time_match.group(2)
                else:
                    unit = "specification"
            else:
                unit = "storage condition"
            qualifier = groups[2] if len(groups) > 2 and groups[2] else ""
            
        elif spec_type == 'packaging_specification':
            unit = groups[2] if len(groups) > 2 and groups[2] else ""
            
        elif spec_type == 'composition_specification':
            value = groups[1] if len(groups) > 1 else ""
            qualifier = groups[2] if len(groups) > 2 and groups[2] else ""
            unit = "ingredient list"
            
        elif spec_type in ['allergen_specification', 'certification_specification', 'sensory_specification', 'additive_specification']:
            value = groups[1] if len(groups) > 1 else ""
            qualifier = groups[2] if len(groups) > 2 and groups[2] else ""
            unit = "specification"
        
        confidence = self._calculate_food_confidence(property_name, value, unit, spec_type)
        
        return TechnicalSpecEntity(
            property=property_name,
            value=value,
            unit=unit,
            qualifier=qualifier,
            full_context=full_context,
            confidence=confidence,
            start_pos=match.start(),
            end_pos=match.end(),
            domain_specific_attributes={
                'specification_type': spec_type,
                'food_category': self._classify_food_spec(property_name, value, spec_type),
                'has_limit': any(word in qualifier.lower() for word in ['max', 'min', 'maximum', 'minimum']),
                'has_range': value and ('-' in value or 'to' in value.lower()),
                'regulatory_relevant': self._is_regulatory_relevant(spec_type, property_name)
            },
            specification_type=spec_type,
            related_properties=[],
            measurement_range=None
        )
    
    def _calculate_food_confidence(self, property_name: str, value: str, unit: str, spec_type: str) -> float:
        """Calculate confidence for food industry specification"""
        confidence = 0.7  # Base confidence for food specs
        
        # Nutritional and microbiological specs are usually well-defined
        if spec_type in ['nutritional_specification', 'microbiological_specification']:
            confidence += 0.2
        
        # Regulatory specifications have high confidence
        if spec_type in ['allergen_specification', 'certification_specification']:
            confidence += 0.15
        
        # Well-formatted numerical values increase confidence
        if value and re.match(r'^\d+(?:\.\d+)?$', value):
            confidence += 0.1
        
        # Unit presence increases confidence
        if unit and unit != "specification":
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _classify_food_spec(self, property_name: str, value: str, spec_type: str) -> str:
        """Classify food industry specification type"""
        if spec_type == 'nutritional_specification':
            return 'nutritional_information'
        elif spec_type == 'microbiological_specification':
            return 'food_safety'
        elif spec_type in ['physical_specification', 'processing_specification']:
            return 'quality_parameter'
        elif spec_type in ['storage_specification', 'packaging_specification']:
            return 'handling_requirement'
        elif spec_type in ['allergen_specification', 'certification_specification']:
            return 'regulatory_compliance'
        elif spec_type in ['composition_specification', 'additive_specification']:
            return 'ingredient_information'
        elif spec_type == 'sensory_specification':
            return 'quality_attribute'
        else:
            return 'general_food_spec'
    
    def _is_regulatory_relevant(self, spec_type: str, property_name: str) -> bool:
        """Check if specification is relevant for regulatory compliance"""
        regulatory_types = ['allergen_specification', 'certification_specification', 'microbiological_specification']
        regulatory_properties = ['Salmonella', 'E. coli', 'Listeria', 'allergen', 'FDA', 'USDA', 'organic']
        
        return (spec_type in regulatory_types or 
                any(prop.lower() in property_name.lower() for prop in regulatory_properties))

class TechnicalSpecificationExtractor:
    """
    Specialized extractor for technical specifications as complete units
    Prevents fragmentation of property-value-context relationships
    """
    
    def __init__(self):
        self.pharmaceutical_extractor = PharmaceuticalSpecExtractor()
        self.legal_extractor = LegalSpecExtractor()
        self.food_extractor = FoodIndustrySpecExtractor()
        
        # General technical patterns for other domains
        self.general_patterns = [
            # Standard property: value unit format
            r'([A-Za-z\s]+):\s*([0-9.,\-]+(?:\s*to\s*[0-9.,\-]+)?)\s*([%°C\w/]+)(?:\s+(max|min|typical))?',
            # Equals format
            r'([A-Za-z\s]+)\s*=\s*([0-9.,\-]+)\s*([%°C\w/]+)',
            # Range format
            r'([A-Za-z\s]+)\s*([0-9.,\-]+)\s*-\s*([0-9.,\-]+)\s*([%°C\w/]+)',
        ]
    
    def extract_technical_specs(self, text: str, domain: str = 'general') -> List[TechnicalSpecEntity]:
        """Extract complete technical specifications based on domain"""
        
        if domain == 'pharmaceutical':
            return self.pharmaceutical_extractor.extract_pharmaceutical_specs(text)
        elif domain == 'legal':
            return self.legal_extractor.extract_legal_specs(text)
        elif domain == 'food':
            return self.food_extractor.extract_food_specs(text)
        else:
            return self._extract_general_specs(text)
    
    def _extract_general_specs(self, text: str) -> List[TechnicalSpecEntity]:
        """Extract general technical specifications"""
        specs = []
        
        for i, pattern in enumerate(self.general_patterns):
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                spec = self._create_general_spec(match, f"general_pattern_{i}", text)
                if spec:
                    specs.append(spec)
        
        return specs
    
    def _create_general_spec(self, match: re.Match, pattern_type: str, full_text: str) -> Optional[TechnicalSpecEntity]:
        """Create a general technical specification entity"""
        groups = match.groups()
        
        if len(groups) < 3:
            return None
        
        property_name = groups[0].strip()
        value = groups[1].strip()
        unit = groups[2].strip()
        qualifier = groups[3] if len(groups) > 3 and groups[3] else ""
        
        # Extract context
        context_start = max(0, match.start() - 50)
        context_end = min(len(full_text), match.end() + 50)
        full_context = full_text[context_start:context_end].strip()
        
        confidence = self._calculate_general_confidence(property_name, value, unit)
        
        return TechnicalSpecEntity(
            property=property_name,
            value=value,
            unit=unit,
            qualifier=qualifier,
            full_context=full_context,
            confidence=confidence,
            start_pos=match.start(),
            end_pos=match.end(),
            domain_specific_attributes={
                'specification_type': pattern_type,
                'general_category': 'technical_measurement'
            },
            specification_type=pattern_type,
            related_properties=[],
            measurement_range=None
        )
    
    def _calculate_general_confidence(self, property_name: str, value: str, unit: str) -> float:
        """Calculate confidence for general specification"""
        confidence = 0.5
        
        if len(property_name) > 2:
            confidence += 0.2
        if re.match(r'^\d+(?:\.\d+)?$', value):
            confidence += 0.2
        if unit:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def create_compound_specifications(self, specs: List[TechnicalSpecEntity]) -> List[CompoundSpecification]:
        """Create compound specifications from related individual specs"""
        compound_specs = []
        
        # Group specifications by proximity and similarity
        grouped_specs = self._group_specifications_by_relationship(specs)
        
        for group in grouped_specs:
            if len(group) > 1:
                compound = CompoundSpecification(
                    main_property=self._determine_main_property(group),
                    specifications=group,
                    compound_context=self._build_compound_context(group),
                    confidence=self._calculate_compound_confidence(group),
                    specification_family=self._determine_specification_family(group),
                    preservation_score=self._calculate_preservation_score(group)
                )
                compound_specs.append(compound)
        
        return compound_specs
    
    def _group_specifications_by_relationship(self, specs: List[TechnicalSpecEntity]) -> List[List[TechnicalSpecEntity]]:
        """Group specifications that are related"""
        grouped = []
        used_indices = set()
        
        for i, spec in enumerate(specs):
            if i in used_indices:
                continue
                
            group = [spec]
            used_indices.add(i)
            
            # Find related specifications
            for j, other_spec in enumerate(specs):
                if j not in used_indices and self._are_specs_related(spec, other_spec):
                    group.append(other_spec)
                    used_indices.add(j)
            
            grouped.append(group)
        
        return grouped
    
    def _are_specs_related(self, spec1: TechnicalSpecEntity, spec2: TechnicalSpecEntity) -> bool:
        """Check if two specifications are related"""
        # Close proximity
        if abs(spec1.start_pos - spec2.start_pos) < 300:
            return True
        
        # Same category
        if (spec1.domain_specific_attributes.get('pharmaceutical_category') == 
            spec2.domain_specific_attributes.get('pharmaceutical_category')):
            return True
        
        # Related property names
        prop1_words = set(spec1.property.lower().split())
        prop2_words = set(spec2.property.lower().split())
        if prop1_words & prop2_words:
            return True
        
        return False
    
    def _determine_main_property(self, group: List[TechnicalSpecEntity]) -> str:
        """Determine the main property for a group of specifications"""
        # Use the specification with highest confidence
        main_spec = max(group, key=lambda s: s.confidence)
        return main_spec.property
    
    def _build_compound_context(self, group: List[TechnicalSpecEntity]) -> str:
        """Build compound context from group of specifications"""
        contexts = [spec.full_context for spec in group]
        return " | ".join(contexts)
    
    def _calculate_compound_confidence(self, group: List[TechnicalSpecEntity]) -> float:
        """Calculate confidence for compound specification"""
        if not group:
            return 0.0
        
        # Average confidence weighted by individual confidences
        total_confidence = sum(spec.confidence for spec in group)
        return total_confidence / len(group)
    
    def _determine_specification_family(self, group: List[TechnicalSpecEntity]) -> str:
        """Determine the family of specifications"""
        categories = [spec.domain_specific_attributes.get('pharmaceutical_category', 'general') 
                     for spec in group]
        
        # Most common category
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return max(category_counts.items(), key=lambda x: x[1])[0]
    
    def _calculate_preservation_score(self, group: List[TechnicalSpecEntity]) -> float:
        """Calculate how well the compound specification preserves relationships"""
        base_score = 0.8
        
        # Bonus for multiple related specifications
        if len(group) > 2:
            base_score += 0.1
        
        # Bonus for high individual confidences
        avg_confidence = sum(spec.confidence for spec in group) / len(group)
        base_score += avg_confidence * 0.1
        
        return min(1.0, base_score) 