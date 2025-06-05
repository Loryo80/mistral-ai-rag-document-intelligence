"""
Entity Quality Filter for Enhanced Legal AI System
Implements confidence filtering, semantic validation, and compound preservation
to reduce extraction noise and improve entity quality.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Metrics for entity quality assessment"""
    confidence_score: float
    semantic_score: float
    length_score: float
    context_score: float
    language_score: float
    final_score: float
    passed_filter: bool
    rejection_reason: Optional[str] = None

class EntityQualityFilter:
    """Advanced quality filter for extracted entities"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize quality filter with configuration"""
        self.config = config or self._get_default_config()
        
        # Quality thresholds
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.min_semantic_score = self.config.get('min_semantic_score', 0.6)
        self.min_final_score = self.config.get('min_final_score', 0.65)
        
        # Initialize language patterns
        self._init_language_patterns()
        
        # Initialize semantic validation patterns
        self._init_semantic_patterns()
        
        # Initialize compound preservation patterns
        self._init_compound_patterns()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for quality filtering"""
        return {
            'min_confidence': 0.4,  # Increased from 0.3 to 0.4
            'min_semantic_score': 0.5,  # Increased from 0.4 to 0.5
            'min_final_score': 0.45,  # Increased from 0.35 to 0.45
            'max_entity_words': 6,
            'min_entity_length': 2,
            'preserve_compounds': True,
            'language_detection': True,
            'strict_mode': False,
            'chemical_validation': True,  # New: enable chemical compound validation
            'semantic_filtering': True,  # New: enable semantic filtering
            'domain_validation': True   # New: enable domain-specific validation
        }
    
    def _init_language_patterns(self):
        """Initialize enhanced language detection patterns with French support"""
        self.language_patterns = {
            'french': {
                'articles': ['le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'dans', 'aux', 'au'],
                'prepositions': ['sur', 'avec', 'pour', 'par', 'dans', 'sans', 'sous', 'vers', 'chez', 'entre'],
                'common_words': ['est', 'sont', 'cette', 'peut', 'doit', 'selon', 'entre', 'cas', 'tout', 'tous'],
                'patterns': [
                    r'\b(?:fiche|données|sécurité|rubrique|identification)\b',
                    r'\b(?:contact|avec|rincer|obtenir|médical)\b',
                    r'\b(?:classification|selon|danger|information)\b',
                    r'\b(?:mesures|protection|individuelle|collective)\b',
                    r'\b(?:produit|substance|composé|ingrédient)\b',
                    r'\b(?:concentration|limite|exposition|niveau)\b'
                ],
                'noise_phrases': [
                    r'\b(?:en cas de|dans la|de la|des produits|avec de|pour la|sur la)\b',
                    r'\b(?:apporter en|obtenir des|laver avec|contact avec|mesures de)\b',
                    r'\b(?:informations sur|classification selon|limite de|niveau important)\b',
                    r'\b(?:bottes en|division of|bibliographiques et|bioaccumulable et)\b'
                ],
                'document_artifacts': [
                    r'\bFDS\b',  # Fiche de Données de Sécurité
                    r'\bCLP\s+Annex\b',
                    r'\bAnnex\s+VIII\b'
                ]
            },
            'english': {
                'articles': ['the', 'a', 'an'],
                'prepositions': ['in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'into', 'onto'],
                'common_words': ['is', 'are', 'this', 'that', 'can', 'may', 'will', 'shall', 'such', 'each'],
                'patterns': [
                    r'\b(?:specification|property|characteristic|requirement)\b',
                    r'\b(?:content|level|maximum|minimum|typical)\b',
                    r'\b(?:application|method|procedure|process)\b',
                    r'\b(?:safety|data|sheet|material|chemical)\b',
                    r'\b(?:product|ingredient|component|substance)\b'
                ],
                'noise_phrases': [
                    r'\b(?:and such may|be used as|based on the|apply to them)\b',
                    r'\b(?:in case of|contact with|wash with|obtain medical)\b',
                    r'\b(?:information on|classification according|limit of|important level)\b'
                ]
            }
        }
    
    def _init_semantic_patterns(self):
        """Initialize semantic validation patterns for quality assessment"""
        
        # Enhanced meaningful content patterns (positive indicators)
        self.meaningful_patterns = {
            'chemical_names': [
                r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)*(?:\s+\d+)?(?:\s*[®™])?$',  # Proper chemical names
                r'\b\w+(?:\s+\w+){0,3}(?:\s+(?:acid|oxide|hydroxide|sulfate|chloride|nitrate|glycolate))$',
                r'\b(?:sodium|potassium|calcium|magnesium|aluminum)\s+\w+(?:\s+\w+)?$',
                r'\b\w*(?:amine|phenol|benzene|methyl|ethyl|propyl|starch)\w*$',
                # French chemical terms
                r'\b(?:acide|oxyde|hydroxyde|sulfate|chlorure|nitrate|glycolate)\s+\w+$',
                r'\b(?:sodium|potassium|calcium|magnésium|aluminium)\s+\w+(?:\s+\w+)?$'
            ],
            'product_names': [
                r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*(?:\s*[®™])?$',  # Brand names with trademarks
                r'\b\w+(?:\s+\w+){0,2}(?:\s+(?:powder|tablet|solution|gel|cream|poudre|comprimé|solution))$',
                r'\b[A-Z]{2,}(?:\s*\d+)?(?:\s*[®™])?$',  # Acronyms with optional numbers and trademarks
                # Enhanced brand detection
                r'\bLYCATAB(?:\s*®)?(?:\s+[A-Z]+)?$',  # LYCATAB brand variations
                r'\bPEARLITOL(?:\s*®)?(?:\s+[A-Z]+)?$',  # PEARLITOL brand variations
                r'\b[A-Z]{3,}(?:\s*[®™])?(?:\s+[A-Z0-9]+)?$',  # Generic brand pattern
                # Product codes and specifications
                r'\b[A-Z]+\s*\d+(?:[A-Z]+)?$',  # Product codes like PGS 200252
                r'\b\w+\s*-\s*\w+(?:\s*-\s*\w+)?$'  # Hyphenated product names
            ],
            'specifications': [
                r'\b\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?\s*(?:%|ppm|mg|kg|°C|pH|μm)$',
                r'\bpH\s*:?\s*\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?$',
                r'\b(?:max|min|typical|nominal|maximum|minimum)\s+\d+(?:\.\d+)?(?:\s*%)?$',
                # French specifications
                r'\b(?:maximum|minimum|typique|nominal)\s+\d+(?:\.\d+)?(?:\s*%)?$',
                r'\b\d+(?:\.\d+)?\s*(?:g/cm³|kg/m³|μm|nm)$'
            ],
            'applications': [
                r'\b\w+(?:\s+\w+){1,4}(?:\s+(?:application|use|purpose|utilisation|usage))$',
                r'\b(?:used|suitable|recommended|intended|utilisé|approprié|recommandé)\s+(?:for|as|in|pour|comme|dans)\s+\w+$'
            ]
        }
        
        # Enhanced noise patterns (negative indicators) with French support
        self.noise_patterns = {
            'sentence_fragments': [
                # English fragments
                r'^(?:and|or|but|the|of|in|on|at|to|for|with|by|from)\b',  # Starting with conjunctions/prepositions
                r'^(?:is|are|was|were|be|been|being|have|has|had|will|would|could|should)\b',  # Starting with verbs
                r'^(?:this|that|these|those|it|they|we|you|he|she)\b',  # Starting with pronouns
                r'\b(?:is|are|was|were)\s+a\s+\w+$',  # "is a something" fragments
                r'\b(?:can|may|must|shall|will|would)\s+be\b',  # Modal + be constructions
                r'\b(?:used|suitable|recommended)\s+for$',  # Incomplete phrases
                r'\b(?:and|or)\s+\w+(?:\s+\w+)?$',  # "and something" fragments
                r'^\w+(?:\s+\w+)?\s+(?:of|in|on|at|to|for|with|by|from)$',  # "something of/in/etc"
                # French fragments
                r'^(?:et|ou|mais|le|la|les|de|du|des|dans|sur|avec|pour|par)\b',  # French conjunctions/prepositions
                r'^(?:est|sont|était|étaient|être|avoir|avait|peut|doit|va|aller)\b',  # French verbs
                r'^(?:ce|cette|ces|ceux|il|elle|ils|elles|nous|vous)\b',  # French pronouns
                r'\b(?:est|sont)\s+(?:un|une)\s+\w+$',  # "est un/une something" fragments
                r'\b(?:peut|doit|va|aller)\s+être\b',  # French modal + être constructions
                r'\b(?:utilisé|approprié|recommandé)\s+pour$',  # French incomplete phrases
                r'\b(?:et|ou)\s+\w+(?:\s+\w+)?$',  # "et/ou something" fragments
                r'^\w+(?:\s+\w+)?\s+(?:de|du|des|dans|sur|avec|pour|par)$',  # French "something de/du/etc"
                # Specific French SDS fragments from LYCATAB analysis
                r'^(?:dans la|de la|des produits|avec de|pour la|sur la)\b',
                r'^(?:apporter en|obtenir des|laver avec|contact avec|mesures de)\b',
                r'^(?:informations sur|classification selon|limite de|niveau important)\b',
                r'^(?:bottes en|division of|bibliographiques et|bioaccumulable et)\b',
                r'^(?:en cas d|en cas de|en suspension dans)\b'
            ],
            'incomplete_specifications': [
                r'^\d+$',  # Just numbers
                r'^[A-Z]$',  # Single letters
                r'^\w{1,2}$',  # Very short tokens
                r'^(?:max|min|typical|nominal|maximum|minimum|typique)$',  # Qualifiers without values
                r'^(?:%|ppm|mg|kg|°C|pH|μm|g/cm³)$',  # Units without values
                r'^\s*[-–:(),.;]\s*$'  # Just punctuation
            ],
            'generic_terms': [
                # English generic terms
                r'^(?:product|ingredient|component|element|substance|material|compound)$',
                r'^(?:property|characteristic|feature|aspect|quality)$',
                r'^(?:information|data|details|description|specification)$',
                r'^(?:content|level|amount|quantity|concentration)$',
                r'^(?:value|number|figure|result|measurement)$',
                # French generic terms
                r'^(?:produit|ingrédient|composant|élément|substance|matériau|composé)$',
                r'^(?:propriété|caractéristique|aspect|qualité)$',
                r'^(?:information|données|détails|description|spécification)$',
                r'^(?:contenu|niveau|quantité|concentration)$',
                r'^(?:valeur|nombre|résultat|mesure)$'
            ],
            'document_artifacts': [
                # English document artifacts
                r'^(?:page|section|chapter|part|table|figure|appendix)\s*\d*$',
                r'^(?:see|refer|reference|note|footnote|remark)$',
                r'^(?:above|below|following|preceding|next|previous)$',
                r'^(?:continued|end|start|begin|conclusion)$',
                # French document artifacts
                r'^(?:page|section|chapitre|partie|tableau|figure|annexe)\s*\d*$',
                r'^(?:voir|référer|référence|note|remarque)$',
                r'^(?:ci-dessus|ci-dessous|suivant|précédent|prochain|précédente)$',
                r'^(?:continué|fin|début|commencer|conclusion)$',
                # SDS specific artifacts
                r'^(?:FDS|CLP|Annex|VIII)$',
                r'^(?:American Chemical Society|Chemical Abstracts Service)$'
            ]
        }
        
        # Enhanced chemical compound specific validation patterns with French support
        self.chemical_validation_patterns = {
            'valid_chemical_structures': [
                r'^[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*(?:\s+\d+)?(?:\s*[®™])?$',  # Proper chemical nomenclature with trademarks
                r'^\w+(?:-\w+)*(?:\s+\w+)*(?:\s*[®™])?$',  # Hyphenated chemical names with trademarks
                r'^(?:sodium|potassium|calcium|magnesium|aluminum|zinc|iron|copper)\s+\w+(?:\s+\w+)?$',  # Metal compounds
                r'^\w*(?:amine|acid|oxide|hydroxide|sulfate|chloride|nitrate|phosphate|glycolate|starch)\w*$',  # Chemical endings
                # French chemical structures
                r'^(?:sodium|potassium|calcium|magnésium|aluminium|zinc|fer|cuivre)\s+\w+(?:\s+\w+)?$',  # French metal compounds
                r'^\w*(?:amine|acide|oxyde|hydroxyde|sulfate|chlorure|nitrate|phosphate|glycolate|amidon)\w*$',  # French chemical endings
                # Specific product patterns
                r'^LYCATAB(?:\s*®)?(?:\s+[A-Z]+)?$',  # LYCATAB brand
                r'^PEARLITOL(?:\s*®)?(?:\s+[A-Z]+)?$',  # PEARLITOL brand
                r'^[A-Z]{3,}(?:\s*[®™])?(?:\s+[A-Z0-9]+)?$',  # Generic branded chemicals
                # Chemical nomenclature indicators
                r'^\w+\s+(?:starch|glycolate|mannitol|sorbitol|cellulose)(?:\s*[®™])?$',
                r'^(?:hypromellose|hydroxypropyl|methylcellulose)(?:\s+\w+)?(?:\s*[®™])?$'
            ],
            'invalid_chemical_fragments': [
                # English invalid fragments
                r'^\w+(?:\s+\w+)*\s+(?:is|are|was|were|and|or|but|the|of|in|on|at|to|for)$',
                r'^(?:high|low|medium|good|bad|excellent|poor)\s+\w+$',  # Quality descriptions
                r'^(?:used|suitable|recommended|intended)\s+for\s+\w+$',  # Usage descriptions
                r'^(?:can|may|must|shall|will|would)\s+be\s+\w+$',  # Modal constructions
                # French invalid fragments
                r'^\w+(?:\s+\w+)*\s+(?:est|sont|était|étaient|et|ou|mais|le|la|les|de|du|des|dans|sur)$',
                r'^(?:haut|bas|moyen|bon|mauvais|excellent|pauvre)\s+\w+$',  # French quality descriptions
                r'^(?:utilisé|approprié|recommandé|destiné)\s+pour\s+\w+$',  # French usage descriptions
                r'^(?:peut|doit|va|aller)\s+être\s+\w+$',  # French modal constructions
                # Specific SDS fragments to filter
                r'^(?:dans la FDS|de la substance|des produits de|eau et du)$',
                r'^(?:contact avec|mesures de|informations sur|classification selon)$',
                r'^(?:en cas de|niveau important de|limite de concentration)$'
            ],
            'trademark_indicators': [
                r'®',  # Registered trademark
                r'™',  # Trademark
                r'\bTM\b',  # Trademark abbreviation
                r'\bREG\b'  # Registered abbreviation
            ]
        }
    
    def _init_compound_patterns(self):
        """Initialize enhanced patterns for compound entity preservation with French support"""
        self.compound_patterns = [
            # Enhanced chemical compound patterns
            r'[A-Z][a-z]+®?\s+[A-Z][A-Z0-9]*(?:\s*-\s*[A-Z0-9]+)?',  # LYCATAB® PGS, etc.
            r'LYCATAB(?:\s*®)?\s*(?:PGS|CR|H)?(?:\s*-\s*[A-Z0-9]+)?',  # LYCATAB variations
            r'PEARLITOL(?:\s*®)?\s*(?:CR|H|SD)?(?:\s*-\s*[A-Z0-9]+)?',  # PEARLITOL variations
            r'NPK\s+\d+-\d+-\d+',  # NPK fertilizers
            r'pH\s+\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?',  # pH ranges
            r'\d+(?:\.\d+)?%\s*w/w',  # Weight percentages
            r'\d+(?:\.\d+)?\s*(?:kg|g|mg)/ha',  # Application rates
            
            # Enhanced technical specifications (English & French)
            r'Loss\s+on\s+drying:\s*\d+(?:\.\d+)?%?',
            r'Perte\s+(?:à|au)\s+séchage:\s*\d+(?:\.\d+)?%?',  # French: Loss on drying
            r'Bulk\s+density:\s*\d+(?:\.\d+)?\s*g/cm³',
            r'Densité\s+(?:apparente|en\s+vrac):\s*\d+(?:\.\d+)?\s*g/cm³',  # French: Bulk density
            r'Particle\s+size:\s*\d+(?:\.\d+)?\s*μm',
            r'Taille\s+(?:des\s+)?particules:\s*\d+(?:\.\d+)?\s*μm',  # French: Particle size
            r'Moisture\s+content:\s*\d+(?:\.\d+)?%?',
            r'Teneur\s+en\s+(?:eau|humidité):\s*\d+(?:\.\d+)?%?',  # French: Moisture content
            
            # Enhanced regulatory codes
            r'CAS\s+No\.?\s*\d+-\d+-\d+',
            r'N°\s*CAS\s*\d+-\d+-\d+',  # French CAS number
            r'E\s*\d{3,4}(?:\s*[a-z])?',  # E-numbers with optional letter
            r'INS\s*\d{3,4}',  # INS numbers
            r'EC\s+No\.?\s*\d+-\d+-\d+',  # EC numbers
            r'N°\s*CE\s*\d+-\d+-\d+',  # French EC number
            
            # Product codes and specifications
            r'[A-Z]+\s*\d{6,}(?:[A-Z]+)?',  # Product codes like PGS 000000200252
            r'EXP_\d{12}',  # Export codes
            r'SDS\s*[A-Z]{2}\s*[A-Z0-9_]+',  # SDS reference codes
            r'FDS\s*[A-Z]{2}\s*[A-Z0-9_]+',  # French SDS reference codes
            
            # Branded product patterns
            r'[A-Z]{3,}(?:\s*®|\s*™)(?:\s+[A-Z0-9]+)?',  # Branded products with trademarks
            r'sodium\s+starch\s+glycolate(?:\s*[®™])?',  # Specific compound
            r'glycolate\s+d\'amidon\s+sodique(?:\s*[®™])?',  # French: sodium starch glycolate
        ]
    
    def filter_entities(self, entities: List[Dict[str, Any]], 
                       document_type: str = "general",
                       language: str = "auto") -> Tuple[List[Dict[str, Any]], List[QualityMetrics]]:
        """
        Filter entities based on quality metrics
        
        Args:
            entities: List of extracted entities
            document_type: Type of document being processed
            language: Language of the document ('auto', 'english', 'french')
            
        Returns:
            Tuple of (filtered_entities, quality_metrics)
        """
        filtered_entities = []
        quality_metrics = []
        
        # Auto-detect language if needed
        if language == "auto":
            language = self._detect_document_language(entities)
        
        for entity in entities:
            metrics = self._calculate_quality_metrics(entity, document_type, language)
            quality_metrics.append(metrics)
            
            if metrics.passed_filter:
                # Apply compound preservation
                if self.config.get('preserve_compounds', True):
                    entity = self._preserve_compound_context(entity)
                
                filtered_entities.append(entity)
            else:
                logger.debug(f"Filtered out entity '{entity.get('value', '')}': {metrics.rejection_reason}")
        
        return filtered_entities, quality_metrics
    
    def _calculate_quality_metrics(self, entity: Dict[str, Any], 
                                 document_type: str, 
                                 language: str) -> QualityMetrics:
        """Calculate comprehensive quality metrics for an entity"""
        
        entity_value = entity.get('value', '') or entity.get('entity_value', '') or entity.get('text', '')
        entity_type = entity.get('type', '') or entity.get('entity_type', '')
        confidence = float(entity.get('confidence', 0) or entity.get('confidence_score', 0))
        context = entity.get('context', '')
        
        # 1. Confidence score (from extraction or estimated)
        if confidence == 0:
            # OpenAI doesn't provide confidence - estimate based on entity characteristics
            confidence = self._estimate_entity_confidence(entity_value, entity_type)
        confidence_score = min(confidence, 1.0)
        
        # 2. Semantic score (meaningfulness)
        semantic_score = self._calculate_semantic_score(entity_value, entity_type)
        
        # 3. Length score (appropriate length)
        length_score = self._calculate_length_score(entity_value)
        
        # 4. Context score (contextual relevance)
        context_score = self._calculate_context_score(entity_value, context, document_type)
        
        # 5. Language score (language consistency)
        language_score = self._calculate_language_score(entity_value, language)
        
        # 6. Calculate final weighted score
        weights = {
            'confidence': 0.3,
            'semantic': 0.25,
            'length': 0.15,
            'context': 0.2,
            'language': 0.1
        }
        
        final_score = (
            confidence_score * weights['confidence'] +
            semantic_score * weights['semantic'] +
            length_score * weights['length'] +
            context_score * weights['context'] +
            language_score * weights['language']
        )
        
        # Determine if entity passes filter
        passed_filter = (
            confidence_score >= self.min_confidence and
            semantic_score >= self.min_semantic_score and
            final_score >= self.min_final_score
        )
        
        # Determine rejection reason
        rejection_reason = None
        if not passed_filter:
            if confidence_score < self.min_confidence:
                rejection_reason = f"Low confidence: {confidence_score:.2f} < {self.min_confidence}"
            elif semantic_score < self.min_semantic_score:
                rejection_reason = f"Low semantic score: {semantic_score:.2f} < {self.min_semantic_score}"
            elif final_score < self.min_final_score:
                rejection_reason = f"Low final score: {final_score:.2f} < {self.min_final_score}"
        
        return QualityMetrics(
            confidence_score=confidence_score,
            semantic_score=semantic_score,
            length_score=length_score,
            context_score=context_score,
            language_score=language_score,
            final_score=final_score,
            passed_filter=passed_filter,
            rejection_reason=rejection_reason
        )
    
    def _calculate_semantic_score(self, entity_value: str, entity_type: str) -> float:
        """
        Calculate semantic quality score with enhanced filtering for irrelevant fragments
        """
        entity_lower = entity_value.lower().strip()
        
        # Start with base score
        semantic_score = 0.5
        
        # Check for meaningful patterns based on entity type
        meaningful_bonus = 0.0
        if entity_type.upper() == "CHEMICAL_COMPOUND":
            for pattern in self.meaningful_patterns['chemical_names']:
                if re.match(pattern, entity_value, re.IGNORECASE):
                    meaningful_bonus += 0.3
                    break
        elif entity_type.upper() == "PRODUCT":
            for pattern in self.meaningful_patterns['product_names']:
                if re.match(pattern, entity_value, re.IGNORECASE):
                    meaningful_bonus += 0.25
                    break
        elif entity_type.upper() == "SPECIFICATION":
            for pattern in self.meaningful_patterns['specifications']:
                if re.match(pattern, entity_value, re.IGNORECASE):
                    meaningful_bonus += 0.35
                    break
        elif entity_type.upper() == "APPLICATION":
            for pattern in self.meaningful_patterns['applications']:
                if re.match(pattern, entity_value, re.IGNORECASE):
                    meaningful_bonus += 0.25
                    break
        
        # Apply noise detection penalties - more aggressive filtering
        noise_penalty = 0.0
        
        # Check sentence fragments
        for pattern in self.noise_patterns['sentence_fragments']:
            if re.search(pattern, entity_lower):
                noise_penalty += 0.4  # Increased penalty
                break
        
        # Check incomplete specifications
        for pattern in self.noise_patterns['incomplete_specifications']:
            if re.match(pattern, entity_lower):
                noise_penalty += 0.5  # Heavy penalty
                break
        
        # Check generic terms
        for pattern in self.noise_patterns['generic_terms']:
            if re.match(pattern, entity_lower):
                noise_penalty += 0.3
                break
        
        # Check document artifacts
        for pattern in self.noise_patterns['document_artifacts']:
            if re.search(pattern, entity_lower):
                noise_penalty += 0.4
                break
        
        # Special chemical compound validation if enabled
        if (self.config.get('chemical_validation', True) and 
            entity_type.upper() == "CHEMICAL_COMPOUND"):
            chemical_score = self._validate_chemical_compound(entity_value)
            if chemical_score < 0.3:
                noise_penalty += 0.3  # Penalty for invalid chemical structures
        
        # Calculate final semantic score
        final_score = semantic_score + meaningful_bonus - noise_penalty
        
        # Additional penalties for specific problematic patterns
        if len(entity_value.split()) > 8:  # Too many words
            final_score -= 0.2
        
        if entity_value.count(' ') > 6:  # Too fragmented
            final_score -= 0.2
        
        # Check for incomplete phrases (ending with prepositions, conjunctions)
        # English incomplete phrases
        if re.search(r'\b(?:and|or|of|in|on|at|to|for|with|by|from|is|are|was|were)$', entity_lower):
            final_score -= 0.3
        
        # French incomplete phrases
        if re.search(r'\b(?:et|ou|de|du|des|dans|sur|avec|pour|par|est|sont|était|étaient)$', entity_lower):
            final_score -= 0.3
        
        return max(0.0, min(1.0, final_score))
    
    def _calculate_length_score(self, entity_value: str) -> float:
        """Calculate score based on entity length appropriateness"""
        if not entity_value:
            return 0.0
        
        word_count = len(entity_value.split())
        char_count = len(entity_value.strip())
        
        # Penalize very short or very long entities
        if char_count < self.config.get('min_entity_length', 2):
            return 0.1
        
        if word_count > self.config.get('max_entity_words', 6):
            return 0.3  # Very long entities are usually noise
        
        # Optimal length ranges
        if 3 <= char_count <= 30 and word_count <= 4:
            return 1.0
        elif char_count <= 50 and word_count <= 6:
            return 0.8
        else:
            return 0.6
    
    def _calculate_context_score(self, entity_value: str, context: str, document_type: str) -> float:
        """Calculate score based on contextual relevance"""
        if not context:
            return 0.6  # Neutral score for missing context
        
        context_lower = context.lower()
        entity_lower = entity_value.lower()
        
        # Document type specific context indicators
        type_indicators = {
            'sds': ['safety', 'hazard', 'chemical', 'composition', 'identification'],
            'technical': ['specification', 'property', 'parameter', 'value', 'range'],
            'agricultural': ['crop', 'yield', 'application', 'fertilizer', 'nutrient'],
            'food': ['ingredient', 'additive', 'nutrition', 'allergen', 'processing']
        }
        
        score = 0.5  # Base score
        
        # Check for relevant context indicators
        indicators = type_indicators.get(document_type.lower(), [])
        for indicator in indicators:
            if indicator in context_lower:
                score += 0.1
        
        # Check if entity appears naturally in context
        if entity_lower in context_lower:
            score += 0.2
        
        # Penalize if context seems unrelated
        if any(noise in context_lower for noise in ['page', 'header', 'footer', 'copyright']):
            score -= 0.3
        
        return max(0.0, min(score, 1.0))
    
    def _calculate_language_score(self, entity_value: str, expected_language: str) -> float:
        """Enhanced language consistency scoring with French support"""
        if not self.config.get('language_detection', True):
            return 1.0  # Skip language validation if disabled
        
        if expected_language not in ['english', 'french']:
            return 0.8  # Neutral score for unknown languages
        
        entity_lower = entity_value.lower()
        
        # Check for language-specific patterns
        lang_config = self.language_patterns.get(expected_language, {})
        
        # Enhanced French noise phrase detection
        if expected_language == 'french':
            # Check for French noise phrases that should be filtered
            for noise_phrase in lang_config.get('noise_phrases', []):
                if re.search(noise_phrase, entity_lower):
                    return 0.1  # Very low score for French noise phrases
            
            # Check for French document artifacts
            for artifact in lang_config.get('document_artifacts', []):
                if re.search(artifact, entity_lower):
                    return 0.2  # Low score for document artifacts
        
        # Penalize common words in wrong language
        wrong_lang = 'french' if expected_language == 'english' else 'english'
        wrong_config = self.language_patterns.get(wrong_lang, {})
        
        # Check if entity is a common word in the wrong language
        all_wrong_words = (
            wrong_config.get('articles', []) + 
            wrong_config.get('prepositions', []) + 
            wrong_config.get('common_words', [])
        )
        
        if entity_lower in all_wrong_words:
            return 0.2  # Low score for wrong language words
        
        # Check for wrong language noise phrases
        for noise_phrase in wrong_config.get('noise_phrases', []):
            if re.search(noise_phrase, entity_lower):
                return 0.1  # Very low score for wrong language noise
        
        # Check for language-specific patterns
        for pattern in wrong_config.get('patterns', []):
            if re.search(pattern, entity_lower):
                return 0.3
        
        # Boost score for trademark indicators (language-neutral)
        if any(re.search(pattern, entity_value) for pattern in self.chemical_validation_patterns.get('trademark_indicators', [])):
            return 1.0  # High score for trademarked products
        
        return 0.9  # Good score if no language conflicts
    
    def _detect_document_language(self, entities: List[Dict[str, Any]], 
                                sample_size: int = 50) -> str:
        """Enhanced auto-detection of document language from entity sample"""
        if not entities:
            return 'english'  # Default fallback
        
        # Sample entities for language detection
        sample_entities = entities[:sample_size]
        
        language_scores = {'english': 0, 'french': 0}
        
        for entity in sample_entities:
            entity_value = (entity.get('value', '') or 
                          entity.get('entity_value', '') or 
                          entity.get('text', '')).lower()
            
            for lang, config in self.language_patterns.items():
                # Check articles, prepositions, common words
                all_words = (
                    config.get('articles', []) + 
                    config.get('prepositions', []) + 
                    config.get('common_words', [])
                )
                
                if entity_value in all_words:
                    language_scores[lang] += 2  # Higher weight for exact matches
                
                # Check patterns
                for pattern in config.get('patterns', []):
                    if re.search(pattern, entity_value):
                        language_scores[lang] += 1
                
                # Check noise phrases (strong indicators)
                for noise_phrase in config.get('noise_phrases', []):
                    if re.search(noise_phrase, entity_value):
                        language_scores[lang] += 3  # Very strong indicator
                
                # Check document artifacts
                for artifact in config.get('document_artifacts', []):
                    if re.search(artifact, entity_value):
                        language_scores[lang] += 2
        
        # Additional French-specific detection
        french_indicators = [
            r'\bFDS\b',  # Fiche de Données de Sécurité
            r'\bdans\s+la\b',
            r'\bde\s+la\b',
            r'\bavec\s+de\b',
            r'\ben\s+cas\s+de\b',
            r'\bmesures\s+de\b',
            r'\bclassification\s+selon\b'
        ]
        
        for entity in sample_entities:
            entity_value = (entity.get('value', '') or 
                          entity.get('entity_value', '') or 
                          entity.get('text', ''))
            
            for indicator in french_indicators:
                if re.search(indicator, entity_value, re.IGNORECASE):
                    language_scores['french'] += 2
        
        # Return language with highest score, with minimum threshold
        max_score = max(language_scores.values())
        if max_score < 2:  # If very low scores, default to English
            detected_lang = 'english'
        else:
            detected_lang = max(language_scores.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Detected document language: {detected_lang} (scores: {language_scores})")
        
        return detected_lang
    
    def _estimate_entity_confidence(self, entity_value: str, entity_type: str) -> float:
        """Estimate confidence for entities without confidence scores (e.g., from OpenAI)"""
        if not entity_value:
            return 0.1
        
        base_confidence = 0.5  # Base confidence for OpenAI extractions
        
        # Boost confidence for high-quality indicators
        confidence_boosts = []
        
        # 1. Entity type quality
        high_quality_types = ['product', 'chemical_compound', 'specification', 'regulatory']
        if entity_type.lower() in high_quality_types:
            confidence_boosts.append(0.2)
        
        # 2. Length and structure quality
        if 3 <= len(entity_value.split()) <= 5:  # Good length
            confidence_boosts.append(0.1)
        
        # 3. Meaningful patterns
        meaningful_patterns = [
            r'[A-Z][a-z]+®',  # Registered trademarks
            r'\d+(?:\.\d+)?%',  # Percentages
            r'pH\s*\d+',  # pH values
            r'CAS\s+\d+-\d+-\d+',  # CAS numbers
            r'\d+(?:\.\d+)?\s*(?:kg|g|mg|L|mL|ha|m²)',  # With units
        ]
        
        for pattern in meaningful_patterns:
            if re.search(pattern, entity_value, re.IGNORECASE):
                confidence_boosts.append(0.15)
                break
        
        # 4. Avoid noise patterns
        noise_patterns = [
            r'^\d+$',  # Just numbers
            r'^[A-Z]{1,2}$',  # Single letters
            r'^(?:de|du|la|le|les|un|une|des|dans|avec|pour|par|sur)$'  # Common words
        ]
        
        for pattern in noise_patterns:
            if re.search(pattern, entity_value, re.IGNORECASE):
                confidence_boosts.append(-0.3)  # Penalty
                break
        
        # Calculate final confidence
        final_confidence = base_confidence + sum(confidence_boosts)
        return max(0.1, min(1.0, final_confidence))
    
    def _preserve_compound_context(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance entity with compound context preservation"""
        entity_value = entity.get('value', '') or entity.get('entity_value', '') or entity.get('text', '')
        
        # Check if entity matches compound patterns
        for pattern in self.compound_patterns:
            if re.search(pattern, entity_value):
                # Mark as compound for special handling
                if 'metadata' not in entity:
                    entity['metadata'] = {}
                entity['metadata']['is_compound'] = True
                entity['metadata']['compound_pattern'] = pattern
                break
        
        return entity
    
    def get_quality_statistics(self, quality_metrics: List[QualityMetrics]) -> Dict[str, Any]:
        """Generate quality statistics from metrics"""
        if not quality_metrics:
            return {}
        
        passed_count = sum(1 for m in quality_metrics if m.passed_filter)
        total_count = len(quality_metrics)
        
        avg_confidence = sum(m.confidence_score for m in quality_metrics) / total_count
        avg_semantic = sum(m.semantic_score for m in quality_metrics) / total_count
        avg_final = sum(m.final_score for m in quality_metrics) / total_count
        
        # Count rejection reasons
        rejection_counts = Counter(
            m.rejection_reason for m in quality_metrics 
            if not m.passed_filter and m.rejection_reason
        )
        
        return {
            'total_entities': total_count,
            'passed_entities': passed_count,
            'filtered_entities': total_count - passed_count,
            'pass_rate': passed_count / total_count if total_count > 0 else 0,
            'average_confidence': avg_confidence,
            'average_semantic_score': avg_semantic,
            'average_final_score': avg_final,
            'rejection_reasons': dict(rejection_counts),
            'quality_improvement': {
                'noise_reduction': (total_count - passed_count) / total_count if total_count > 0 else 0,
                'confidence_threshold_applied': self.min_confidence,
                'semantic_threshold_applied': self.min_semantic_score
            }
        }
    
    def _validate_chemical_compound(self, entity_value: str) -> float:
        """
        Domain-specific validation for chemical compounds
        Returns a score from 0.0 to 1.0 indicating chemical validity
        """
        if not self.config.get('chemical_validation', True):
            return 0.5  # Neutral score if validation disabled
        
        entity_lower = entity_value.lower().strip()
        score = 0.5  # Start with neutral
        
        # Check for valid chemical structure patterns
        valid_structure = False
        for pattern in self.chemical_validation_patterns['valid_chemical_structures']:
            if re.match(pattern, entity_value, re.IGNORECASE):
                valid_structure = True
                score += 0.3
                break
        
        # Check for invalid chemical fragments
        for pattern in self.chemical_validation_patterns['invalid_chemical_fragments']:
            if re.search(pattern, entity_lower):
                score -= 0.4
                break
        
        # Chemical name characteristics
        if len(entity_value.split()) <= 4:  # Reasonable length
            score += 0.1
        
        # Check for chemical nomenclature indicators
        chemical_indicators = [
            'acid', 'oxide', 'hydroxide', 'sulfate', 'chloride', 'nitrate', 'phosphate',
            'sodium', 'potassium', 'calcium', 'magnesium', 'aluminum', 'zinc', 'iron',
            'methyl', 'ethyl', 'propyl', 'butyl', 'phenyl', 'benzyl',
            'amine', 'alcohol', 'ester', 'ether', 'ketone', 'aldehyde'
        ]
        
        if any(indicator in entity_lower for indicator in chemical_indicators):
            score += 0.2
        
        # Check for trademark indicators (valid for branded chemicals)
        if '®' in entity_value or '™' in entity_value:
            score += 0.2
        
        # Check for proper capitalization (chemical names often have specific patterns)
        if re.search(r'^[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*', entity_value):
            score += 0.1
        
        # Penalty for obvious non-chemical fragments
        non_chemical_indicators = [
            'is used', 'can be', 'may be', 'will be', 'should be',
            'has been', 'was used', 'are used', 'were used',
            'suitable for', 'recommended for', 'intended for',
            'application', 'property', 'characteristic', 'feature'
        ]
        
        if any(indicator in entity_lower for indicator in non_chemical_indicators):
            score -= 0.3
        
        return max(0.0, min(1.0, score)) 