"""
Enhanced Entity Normalization Module for Agricultural AI.

This module provides sophisticated entity normalization to eliminate duplicates
and improve semantic consistency across agricultural documents.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
import json
import os
from difflib import SequenceMatcher
from collections import defaultdict

logger = logging.getLogger(__name__)

class EnhancedEntityNormalizer:
    """Advanced entity normalization for agricultural and chemical documents."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with normalization rules and chemical databases."""
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "agro_entities.json"
        )
        self.load_normalization_config()
        self.entity_cache = {}  # Cache for normalized entities
        self.similarity_threshold = 0.85
        
    def load_normalization_config(self):
        """Load normalization rules and reference databases."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            self.normalization_rules = {
                'PRODUCT': [
                    self._standardize_product_name,
                    self._remove_trademark_symbols,
                    self._standardize_separators,
                    self._normalize_case_product
                ],
                'CHEMICAL_COMPOUND': [
                    self._standardize_chemical_name,
                    self._normalize_cas_numbers,
                    self._standardize_chemical_formula,
                    self._normalize_case_chemical
                ],
                'SPECIFICATION': [
                    self._normalize_units,
                    self._standardize_concentration_format,
                    self._normalize_ph_format
                ],
                'CROP': [
                    self._normalize_crop_names,
                    self._expand_scientific_names
                ],
                'NUTRIENT': [
                    self._standardize_nutrient_symbols,
                    self._normalize_nutrient_names
                ],
                'SAFETY_HAZARD': [
                    self._normalize_ghs_codes,
                    self._standardize_h_p_codes
                ]
            }
            
            # Load reference databases
            self.chemical_synonyms = self._build_chemical_synonym_db()
            self.crop_synonyms = self._build_crop_synonym_db()
            
        except Exception as e:
            logger.warning(f"Could not load normalization config: {e}")
            self.normalization_rules = {}
            self.chemical_synonyms = {}
            self.crop_synonyms = {}
    
    def normalize_entity(self, entity_text: str, entity_type: str, 
                        context: Optional[str] = None) -> Tuple[str, float]:
        """
        Normalize entity value with confidence scoring.
        
        Args:
            entity_text: Original entity text
            entity_type: Type of entity (PRODUCT, CHEMICAL_COMPOUND, etc.)
            context: Optional context for better normalization
            
        Returns:
            Tuple of (normalized_text, confidence_score)
        """
        # Check cache first
        cache_key = f"{entity_type}:{entity_text.lower()}"
        if cache_key in self.entity_cache:
            return self.entity_cache[cache_key]
        
        original_text = entity_text
        normalized = entity_text.strip()
        confidence = 1.0
        
        # Apply type-specific normalization rules
        rules = self.normalization_rules.get(entity_type, [])
        for rule in rules:
            try:
                result = rule(normalized, context)
                if isinstance(result, tuple):
                    normalized, rule_confidence = result
                    confidence = min(confidence, rule_confidence)
                else:
                    normalized = result
            except Exception as e:
                logger.warning(f"Normalization rule failed for {entity_type}: {e}")
                confidence *= 0.9
        
        # Apply general cleaning
        normalized = self._apply_general_cleaning(normalized)
        
        # Check for duplicates using similarity
        normalized, duplicate_confidence = self._check_for_duplicates(
            normalized, entity_type
        )
        confidence = min(confidence, duplicate_confidence)
        
        # Cache result
        result = (normalized, confidence)
        self.entity_cache[cache_key] = result
        
        return result
    
    def find_entity_clusters(self, entities: List[Dict[str, Any]], 
                           entity_type: str) -> List[List[Dict[str, Any]]]:
        """
        Find clusters of similar entities that should be merged.
        
        Args:
            entities: List of entity dictionaries
            entity_type: Type of entities
            
        Returns:
            List of entity clusters (groups of similar entities)
        """
        clusters = []
        processed = set()
        
        for i, entity in enumerate(entities):
            if i in processed:
                continue
                
            cluster = [entity]
            entity_value = entity.get('entity_value', '').lower()
            
            for j, other_entity in enumerate(entities[i+1:], i+1):
                if j in processed:
                    continue
                    
                other_value = other_entity.get('entity_value', '').lower()
                
                # Calculate similarity
                similarity = self._calculate_entity_similarity(
                    entity_value, other_value, entity_type
                )
                
                if similarity > self.similarity_threshold:
                    cluster.append(other_entity)
                    processed.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
            processed.add(i)
        
        return clusters
    
    def merge_entity_cluster(self, cluster: List[Dict[str, Any]], 
                           entity_type: str) -> Dict[str, Any]:
        """
        Merge a cluster of similar entities into a single canonical entity.
        
        Args:
            cluster: List of similar entities
            entity_type: Type of entities
            
        Returns:
            Merged canonical entity
        """
        if not cluster:
            return {}
        
        # Find the best representative (highest confidence)
        best_entity = max(cluster, key=lambda x: x.get('confidence_score', 0))
        
        # Normalize the best entity value
        normalized_value, norm_confidence = self.normalize_entity(
            best_entity['entity_value'], entity_type
        )
        
        # Aggregate metadata from all entities in cluster
        merged_metadata = {}
        all_mentions = []
        
        for entity in cluster:
            if entity.get('metadata'):
                for key, value in entity['metadata'].items():
                    if key not in merged_metadata:
                        merged_metadata[key] = []
                    if isinstance(value, list):
                        merged_metadata[key].extend(value)
                    else:
                        merged_metadata[key].append(value)
            
            all_mentions.append(entity['entity_value'])
        
        # Create merged entity
        merged_entity = {
            'entity_type': entity_type,
            'entity_value': normalized_value,
            'confidence_score': min(best_entity['confidence_score'], norm_confidence),
            'metadata': {
                **merged_metadata,
                'original_mentions': list(set(all_mentions)),
                'cluster_size': len(cluster),
                'normalization_applied': True
            }
        }
        
        return merged_entity
    
    # Normalization rule implementations
    def _standardize_product_name(self, text: str, context: Optional[str] = None) -> str:
        """Standardize product names."""
        # Remove extra spaces and normalize case
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Standardize common product name patterns
        text = re.sub(r'[-_]+', ' ', text)  # Convert dashes/underscores to spaces
        text = re.sub(r'\s*®\s*', '', text)  # Remove registered trademark
        text = re.sub(r'\s*™\s*', '', text)  # Remove trademark
        text = re.sub(r'\s*©\s*', '', text)  # Remove copyright
        
        # Standardize NPK format
        npk_pattern = r'\b(\d+)-(\d+)-(\d+)\b'
        text = re.sub(npk_pattern, r'\1-\2-\3', text)
        
        return text.upper().strip()
    
    def _remove_trademark_symbols(self, text: str, context: Optional[str] = None) -> str:
        """Remove trademark and copyright symbols."""
        symbols = ['®', '™', '©', '℠']
        for symbol in symbols:
            text = text.replace(symbol, '')
        return text.strip()
    
    def _standardize_separators(self, text: str, context: Optional[str] = None) -> str:
        """Standardize separators in product names."""
        # Convert various separators to consistent format
        text = re.sub(r'[-_/\\]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _normalize_case_product(self, text: str, context: Optional[str] = None) -> str:
        """Normalize case for product names."""
        # Keep all caps for product codes, title case for descriptive names
        if re.match(r'^[A-Z0-9\s\-]+$', text) and len(text.split()) <= 3:
            return text.upper()
        else:
            return text.title()
    
    def _standardize_chemical_name(self, text: str, context: Optional[str] = None) -> str:
        """Standardize chemical compound names."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Standardize common chemical name patterns
        text = re.sub(r'(\d+),(\d+)', r'\1,\2', text)  # Fix comma spacing in names
        
        # Check for known synonyms
        text_lower = text.lower()
        if text_lower in self.chemical_synonyms:
            return self.chemical_synonyms[text_lower]
        
        return text
    
    def _normalize_cas_numbers(self, text: str, context: Optional[str] = None) -> str:
        """Normalize CAS registry numbers."""
        # CAS number pattern: XXXXX-XX-X
        cas_pattern = r'(?:CAS\s*[:#]?\s*)?(\d{2,7})-(\d{2})-(\d)(?:\s*CAS)?'
        
        match = re.search(cas_pattern, text, re.IGNORECASE)
        if match:
            return f"CAS: {match.group(1)}-{match.group(2)}-{match.group(3)}"
        
        return text
    
    def _standardize_chemical_formula(self, text: str, context: Optional[str] = None) -> str:
        """Standardize chemical formulas."""
        # Convert subscripts to standard notation
        subscript_map = {'₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4', 
                        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'}
        
        for subscript, number in subscript_map.items():
            text = text.replace(subscript, number)
        
        return text
    
    def _normalize_case_chemical(self, text: str, context: Optional[str] = None) -> str:
        """Normalize case for chemical names."""
        # Chemical names should be lowercase except for proper nouns
        return text.lower()
    
    def _normalize_units(self, text: str, context: Optional[str] = None) -> str:
        """Normalize units and measurements."""
        # Standardize common unit formats
        unit_mappings = {
            r'\bpercent\b': '%',
            r'\bpct\b': '%',
            r'\bkg/hectare\b': 'kg/ha',
            r'\bkilogram per hectare\b': 'kg/ha',
            r'\bliters per hectare\b': 'L/ha',
            r'\bl/hectare\b': 'L/ha',
            r'\bmg/kg\b': 'ppm',
            r'\bmilligrams per kilogram\b': 'ppm'
        }
        
        for pattern, replacement in unit_mappings.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _standardize_concentration_format(self, text: str, context: Optional[str] = None) -> str:
        """Standardize concentration formats."""
        # Pattern: number followed by %
        conc_pattern = r'(\d+(?:\.\d+)?)\s*%'
        text = re.sub(conc_pattern, r'\1%', text)
        
        return text
    
    def _normalize_ph_format(self, text: str, context: Optional[str] = None) -> str:
        """Normalize pH value formats."""
        ph_pattern = r'pH\s*[:=]?\s*(\d+(?:\.\d+)?)'
        text = re.sub(ph_pattern, r'pH \1', text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_crop_names(self, text: str, context: Optional[str] = None) -> str:
        """Normalize crop names."""
        text_lower = text.lower().strip()
        
        # Check crop synonyms
        if text_lower in self.crop_synonyms:
            return self.crop_synonyms[text_lower]
        
        return text.lower()
    
    def _expand_scientific_names(self, text: str, context: Optional[str] = None) -> str:
        """Expand abbreviated scientific names."""
        # Common genus abbreviations
        genus_expansions = {
            'z. mays': 'zea mays',
            't. aestivum': 'triticum aestivum',
            'g. max': 'glycine max',
            's. lycopersicum': 'solanum lycopersicum'
        }
        
        text_lower = text.lower()
        for abbrev, full in genus_expansions.items():
            if abbrev in text_lower:
                text = text_lower.replace(abbrev, full)
        
        return text
    
    def _standardize_nutrient_symbols(self, text: str, context: Optional[str] = None) -> str:
        """Standardize nutrient symbols."""
        nutrient_mappings = {
            'nitrogen': 'N',
            'phosphorus': 'P',
            'phosphorous': 'P',  # Common misspelling
            'potassium': 'K',
            'calcium': 'Ca',
            'magnesium': 'Mg',
            'sulfur': 'S',
            'iron': 'Fe',
            'manganese': 'Mn',
            'zinc': 'Zn',
            'copper': 'Cu',
            'boron': 'B',
            'molybdenum': 'Mo'
        }
        
        text_lower = text.lower()
        for name, symbol in nutrient_mappings.items():
            if name in text_lower:
                return symbol
        
        return text
    
    def _normalize_nutrient_names(self, text: str, context: Optional[str] = None) -> str:
        """Normalize nutrient names."""
        return text.lower().strip()
    
    def _normalize_ghs_codes(self, text: str, context: Optional[str] = None) -> str:
        """Normalize GHS hazard codes."""
        # GHS pictogram codes: GHS01-GHS09
        ghs_pattern = r'GHS\s*0?(\d)'
        text = re.sub(ghs_pattern, r'GHS0\1', text, flags=re.IGNORECASE)
        
        return text.upper()
    
    def _standardize_h_p_codes(self, text: str, context: Optional[str] = None) -> str:
        """Standardize H and P codes."""
        # H-codes: H200-H999, P-codes: P100-P999
        hp_pattern = r'([HP])\s*(\d{3})'
        text = re.sub(hp_pattern, r'\1\2', text, flags=re.IGNORECASE)
        
        return text.upper()
    
    def _apply_general_cleaning(self, text: str) -> str:
        """Apply general text cleaning."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove leading/trailing punctuation (except for specific cases)
        text = text.strip('.,;:')
        
        return text
    
    def _check_for_duplicates(self, text: str, entity_type: str) -> Tuple[str, float]:
        """Check for existing similar entities and return canonical form."""
        # This would integrate with a database of existing entities
        # For now, return the text as-is with full confidence
        return text, 1.0
    
    def _calculate_entity_similarity(self, text1: str, text2: str, entity_type: str) -> float:
        """Calculate similarity between two entity values."""
        # Use different similarity metrics based on entity type
        if entity_type == 'CHEMICAL_COMPOUND':
            # For chemicals, exact match is more important
            if text1 == text2:
                return 1.0
            return SequenceMatcher(None, text1, text2).ratio() * 0.9
        
        elif entity_type == 'PRODUCT':
            # For products, handle variations in naming
            # Remove common variations
            clean1 = re.sub(r'[®™©\-\s]+', '', text1.lower())
            clean2 = re.sub(r'[®™©\-\s]+', '', text2.lower())
            
            if clean1 == clean2:
                return 0.95
            
            return SequenceMatcher(None, clean1, clean2).ratio()
        
        else:
            # Default similarity
            return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _build_chemical_synonym_db(self) -> Dict[str, str]:
        """Build database of chemical synonyms."""
        return {
            # Common chemical synonyms
            'glyphosate': 'N-(phosphonomethyl)glycine',
            '2,4-d': '2,4-dichlorophenoxyacetic acid',
            'urea': 'carbamide',
            'potash': 'potassium chloride',
            'muriate of potash': 'potassium chloride',
            # Add more as needed
        }
    
    def _build_crop_synonym_db(self) -> Dict[str, str]:
        """Build database of crop synonyms."""
        return {
            # Common crop synonyms
            'corn': 'maize',
            'zea mays': 'maize',
            'soy': 'soybean',
            'glycine max': 'soybean',
            'wheat': 'triticum aestivum',
            # Add more as needed
        } 