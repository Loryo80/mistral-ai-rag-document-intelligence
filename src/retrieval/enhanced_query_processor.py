"""
Enhanced Query Processing Module for Agricultural AI with Food Industry B2B Support.

This module provides sophisticated query enhancement that leverages extracted entities
to dramatically improve answer quality for entity-specific questions in both
agricultural and food industry domains.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import json
import os
from difflib import SequenceMatcher
from collections import defaultdict, Counter


logger = logging.getLogger(__name__)

class EnhancedQueryProcessor:
    """Advanced query processing with entity-aware semantic matching for agriculture and food industry."""
    
    def __init__(self, db_instance, config_path: Optional[str] = None):
        """Initialize with database connection and configuration."""
        self.db = db_instance
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "agro_entities.json"
        )
        # Load food industry configuration
        self.food_config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "food_industry_entities.json"
        )
        self.load_query_config()
        self.load_food_config()
        self.entity_cache = {}
        self.query_cache = {}
        
    def load_query_config(self):
        """Load query processing configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load query config: {e}")
            self.config = {
                "entity_types": {},
                "relationship_types": {},
                "intent_patterns": {},
                "expansion_rules": {}
            }
        
        # Initialize intent patterns for query classification (agricultural + food industry)
        self.intent_patterns = {
            # Agricultural intent patterns
            "product_information": [
                r"what is.*?(?:product|fertilizer|pesticide)",
                r"(?:tell me about|describe|information about).*?(?:product|brand)",
                r"(?:properties|characteristics|features) of",
                r"(?:composition|ingredients|active.*?ingredient)"
            ],
            "application_instructions": [
                r"how to (?:apply|use|mix)",
                r"application (?:rate|method|timing)",
                r"when to (?:apply|spray|use)",
                r"(?:dosage|dose|rate|amount) (?:for|of|per)"
            ],
            "efficacy_performance": [
                r"(?:effectiveness|efficacy|performance|results)",
                r"(?:yield|increase|improvement|benefit)",
                r"(?:trial|test|study|research) results?",
                r"(?:compared to|vs|versus|against)"
            ],
            "safety_regulatory": [
                r"(?:safety|hazard|risk|precaution)",
                r"(?:ghs|sds|msds|safety data)",
                r"(?:registration|approval|certified|compliant)",
                r"(?:ppe|protective equipment|warning)"
            ],
            "compatibility_mixing": [
                r"(?:compatible|mix|tank.*?mix|combine)",
                r"(?:can.*?be.*?mixed|mixing|blend)",
                r"(?:incompatible|do not mix|avoid)"
            ],
            "crop_specific": [
                r"(?:for|on|in) (?:corn|wheat|tomato|soybean|rice)",
                r"crop (?:specific|recommendation|application)",
                r"variety|cultivar|species"
            ],
            "technical_specs": [
                r"(?:specification|parameter|property)",
                r"(?:ph|density|solubility|concentration)",
                r"(?:analysis|analytical|test.*?method)",
                r"(?:storage|shelf.*?life|stability)"
            ],
            "comparison": [
                r"(?:compare|comparison|vs|versus|against)",
                r"(?:difference|similar|alternative|substitute)",
                r"(?:better|best|optimal|recommend)"
            ],
            
            # Food industry intent patterns
            "ingredient_information": [
                r"what is.*?(?:ingredient|additive|preservative|stabilizer)",
                r"(?:tell me about|describe|information about).*?(?:ingredient|food additive)",
                r"(?:properties|characteristics|features) of.*?(?:ingredient|additive)",
                r"(?:composition|components|formula) of.*?(?:ingredient|food product)"
            ],
            "nutritional_inquiry": [
                r"(?:nutritional|nutrition) (?:content|value|information)",
                r"(?:vitamin|mineral|protein|calories) (?:content|level|amount)",
                r"how (?:much|many) (?:calories|protein|vitamin|mineral)",
                r"(?:macro|micro)nutrients? in",
                r"daily value|dv|recommended daily"
            ],
            "allergen_safety": [
                r"(?:allergen|allergenic) (?:information|content|declaration)",
                r"(?:contains|has|includes) (?:any|which) (?:allergens?)",
                r"(?:allergen|dairy|gluten|nut|soy|egg|fish|shellfish).{0,5}free",
                r"(?:may contain|processed in|cross.{0,5}contamination)",
                r"allergen labeling|allergen statement"
            ],
            "food_regulatory": [
                r"(?:gras|fda|efsa) (?:status|approved|certified)",
                r"(?:e.{0,5}number|ins.{0,5}number|food additive number)",
                r"(?:regulatory|compliance) (?:status|approval|certification)",
                r"(?:kosher|halal|organic) certified",
                r"food grade|food quality"
            ],
            "food_application": [
                r"(?:used|suitable|applied) (?:in|for) (?:food|beverage|dairy|bakery)",
                r"(?:applications?|uses?) (?:in|for) (?:food industry|food products)",
                r"(?:bakery|dairy|beverage|meat|confectionery|snack) (?:applications?|products?)",
                r"can.*?be used in.*?(?:food|beverage|product)",
                r"suitable for.*?(?:food|beverage|dairy|bakery)"
            ],
            "food_processing": [
                r"(?:processing|manufacturing) (?:method|technique|process)",
                r"(?:spray dry|freeze dry|extract|ferment|encapsulat)",
                r"(?:stability|shelf life|storage) (?:conditions?|requirements?)",
                r"(?:solubility|dispersibility|flow) properties"
            ],
            "substitution_alternatives": [
                r"(?:alternative|substitute|replacement) (?:to|for)",
                r"what can (?:replace|substitute)",
                r"(?:alternatives|substitutes) (?:to|for)",
                r"(?:natural|synthetic) alternative",
                r"clean label (?:alternative|substitute)"
            ],
            
            # Legal document intent patterns
            "contract_analysis": [
                r"(?:contract|agreement) (?:terms|conditions|provisions)",
                r"(?:analyze|review|examine) (?:contract|agreement)",
                r"(?:obligations|responsibilities) (?:under|in) (?:contract|agreement)",
                r"contractual (?:obligations|terms|provisions)"
            ],
            "compliance_inquiry": [
                r"(?:compliance|regulatory) (?:requirements|obligations)",
                r"(?:must|required to|obligated to|shall)",
                r"(?:regulation|law|standard) (?:requires|mandates|specifies)",
                r"(?:compliance with|adherence to|conform to)"
            ],
            "liability_warranty": [
                r"(?:liability|responsibility|fault) (?:for|of|under)",
                r"(?:warranty|guarantee|assurance) (?:of|for|that)",
                r"(?:indemnification|indemnity|hold harmless)",
                r"(?:damages|loss|injury|harm) (?:arising|resulting)"
            ],
            "intellectual_property": [
                r"(?:intellectual property|ip|patent|trademark|copyright)",
                r"(?:proprietary|confidential) (?:information|data|technology)",
                r"(?:license|licensing) (?:rights|agreement|terms)",
                r"(?:non-disclosure|confidentiality|trade secret)"
            ],
            "termination_dispute": [
                r"(?:termination|end|cancel|dissolution) (?:of|clause|provision)",
                r"(?:breach|violation|default) (?:of|under)",
                r"(?:dispute|conflict|disagreement) (?:resolution|procedure)",
                r"(?:arbitration|mediation|litigation|court)"
            ]
        }
        
        # Compile regex patterns for performance
        self.compiled_patterns = {}
        for intent, patterns in self.intent_patterns.items():
            self.compiled_patterns[intent] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        
        # Entity relevance scoring weights
        self.entity_relevance_weights = {
            "exact_match": 1.0,
            "partial_match": 0.8,
            "synonym_match": 0.9,
            "related_match": 0.6,
            "context_match": 0.7
        }
        
        # Intent-entity relevance mapping (agricultural + food industry)
        self.intent_entity_relevance = {
            # Agricultural intents
            "product_information": {
                "PRODUCT": 1.0, "CHEMICAL_COMPOUND": 0.9, "SPECIFICATION": 0.8,
                "COMPANY_INFO": 0.7, "REGULATORY": 0.6
            },
            "application_instructions": {
                "APPLICATION": 1.0, "METHOD": 1.0, "TIMING": 0.9,
                "CROP": 0.8, "PRODUCT": 0.8, "METRIC": 0.7
            },
            "efficacy_performance": {
                "METRIC": 1.0, "APPLICATION": 0.9, "CROP": 0.8,
                "PRODUCT": 0.8, "SPECIFICATION": 0.6
            },
            "safety_regulatory": {
                "SAFETY_HAZARD": 1.0, "REGULATORY": 1.0, "CHEMICAL_COMPOUND": 0.8,
                "PRODUCT": 0.7, "STORAGE_CONDITION": 0.9
            },
            "compatibility_mixing": {
                "PRODUCT": 1.0, "CHEMICAL_COMPOUND": 0.9, "APPLICATION": 0.8,
                "METHOD": 0.7
            },
            "crop_specific": {
                "CROP": 1.0, "APPLICATION": 0.9, "PRODUCT": 0.8,
                "TIMING": 0.8, "METHOD": 0.7
            },
            "technical_specs": {
                "SPECIFICATION": 1.0, "CHEMICAL_COMPOUND": 0.9, "PRODUCT": 0.8,
                "STORAGE_CONDITION": 0.8, "REGULATORY": 0.6
            },
            "comparison": {
                "PRODUCT": 1.0, "METRIC": 0.9, "SPECIFICATION": 0.8,
                "APPLICATION": 0.7, "CROP": 0.7
            },
            
            # Food industry intents
            "ingredient_information": {
                "FOOD_INGREDIENT": 1.0, "NUTRITIONAL_COMPONENT": 0.9, "FOOD_ADDITIVE_TYPE": 0.8,
                "FOOD_SAFETY_STANDARD": 0.7, "FOOD_PROCESSING_METHOD": 0.6
            },
            "nutritional_inquiry": {
                "NUTRITIONAL_COMPONENT": 1.0, "FOOD_INGREDIENT": 0.9, "FOOD_APPLICATION": 0.7,
                "ALLERGEN_INFO": 0.6
            },
            "allergen_safety": {
                "ALLERGEN_INFO": 1.0, "FOOD_INGREDIENT": 0.9, "FOOD_SAFETY_STANDARD": 0.8,
                "FOOD_APPLICATION": 0.7
            },
            "food_regulatory": {
                "FOOD_SAFETY_STANDARD": 1.0, "FOOD_INGREDIENT": 0.9, "NUTRITIONAL_COMPONENT": 0.7,
                "FOOD_ADDITIVE_TYPE": 0.8
            },
            "food_application": {
                "FOOD_APPLICATION": 1.0, "FOOD_INGREDIENT": 0.9, "FOOD_ADDITIVE_TYPE": 0.8,
                "FOOD_PROCESSING_METHOD": 0.7
            },
            "food_processing": {
                "FOOD_PROCESSING_METHOD": 1.0, "FOOD_STORAGE_CONDITION": 0.9, 
                "FOOD_INGREDIENT": 0.8, "FOOD_ADDITIVE_TYPE": 0.7
            },
            "substitution_alternatives": {
                "FOOD_INGREDIENT": 1.0, "FOOD_ADDITIVE_TYPE": 0.9, "NUTRITIONAL_COMPONENT": 0.8,
                "FOOD_APPLICATION": 0.7
            },
            
            # Legal document intents
            "contract_analysis": {
                "LEGAL_PROVISION": 1.0, "LEGAL_OBLIGATION": 0.9, "LEGAL_CLAUSE": 0.8,
                "LEGAL_TERM": 0.7, "PARTY": 0.8, "AGREEMENT_TYPE": 0.9
            },
            "compliance_inquiry": {
                "REGULATORY_REQUIREMENT": 1.0, "LEGAL_OBLIGATION": 0.9, "COMPLIANCE_STANDARD": 0.8,
                "LEGAL_PROVISION": 0.7, "REGULATORY_BODY": 0.8
            },
            "liability_warranty": {
                "LIABILITY_CLAUSE": 1.0, "WARRANTY_PROVISION": 1.0, "INDEMNITY_CLAUSE": 0.9,
                "DAMAGE_TYPE": 0.8, "LEGAL_REMEDY": 0.7
            },
            "intellectual_property": {
                "IP_RIGHT": 1.0, "CONFIDENTIAL_INFO": 0.9, "LICENSE_TERM": 0.8,
                "PROPRIETARY_TECH": 0.8, "TRADE_SECRET": 0.9
            },
            "termination_dispute": {
                "TERMINATION_CLAUSE": 1.0, "BREACH_TYPE": 0.9, "DISPUTE_RESOLUTION": 0.8,
                "ARBITRATION_CLAUSE": 0.8, "GOVERNING_LAW": 0.7
            }
        }

    def load_food_config(self):
        """Load food industry entity configuration."""
        try:
            with open(self.food_config_path, 'r', encoding='utf-8') as f:
                self.food_config = json.load(f)
                # Merge food entity types with existing config
                if 'entity_types' not in self.config:
                    self.config['entity_types'] = {}
                self.config['entity_types'].update(self.food_config.get('entity_types', {}))
                
                # Merge food relationship types
                if 'relationship_types' not in self.config:
                    self.config['relationship_types'] = {}
                self.config['relationship_types'].update(self.food_config.get('relationship_types', {}))
                
        except Exception as e:
            logger.warning(f"Could not load food industry config: {e}")
            self.food_config = {"entity_types": {}, "relationship_types": {}}

    def classify_query_domain(self, query: str) -> str:
        """
        Classify query domain as agricultural, food_industry, or mixed.
        
        Args:
            query: User query string
            
        Returns:
            Domain classification: 'agricultural', 'food_industry', or 'mixed'
        """
        query_lower = query.lower()
        
        # Agricultural keywords
        agricultural_keywords = [
            'fertilizer', 'pesticide', 'herbicide', 'crop', 'field', 'soil', 
            'agriculture', 'farming', 'plant', 'seed', 'cultivation', 'harvest',
            'fungicide', 'insecticide', 'nitrogen', 'phosphorus', 'potassium'
        ]
        
        # Food industry keywords
        food_keywords = [
            'ingredient', 'additive', 'preservative', 'flavor', 'nutrition', 
            'food', 'beverage', 'dietary', 'allergen', 'vitamin', 'mineral',
            'protein', 'supplement', 'gras', 'fda', 'e-number', 'kosher', 'halal',
            'bakery', 'dairy', 'confectionery', 'snack', 'emulsifier', 'stabilizer',
            'sweetener', 'colorant', 'thickener', 'antioxidant', 'pectin', 'gelatin',
            'lecithin', 'carrageenan', 'xanthan', 'agar', 'maltodextrin', 'dextrose'
        ]
        
        # Legal document keywords
        legal_keywords = [
            'contract', 'agreement', 'clause', 'provision', 'obligation', 'liability',
            'warranty', 'indemnity', 'breach', 'compliance', 'regulatory', 'license',
            'intellectual property', 'confidentiality', 'non-disclosure', 'terms',
            'conditions', 'governing law', 'jurisdiction', 'arbitration', 'dispute',
            'termination', 'force majeure', 'amendment', 'assignment', 'sublicense'
        ]
        
        # Mixed domain keywords
        mixed_keywords = [
            'organic', 'natural', 'extract', 'processing', 'manufacturing',
            'quality', 'safety', 'regulatory', 'certification'
        ]
        
        # Count keyword occurrences
        agricultural_score = sum(1 for keyword in agricultural_keywords if keyword in query_lower)
        food_score = sum(1 for keyword in food_keywords if keyword in query_lower)
        legal_score = sum(1 for keyword in legal_keywords if keyword in query_lower)
        mixed_score = sum(1 for keyword in mixed_keywords if keyword in query_lower)
        
        # Determine domain
        if legal_score > max(food_score, agricultural_score) and legal_score > 0:
            return 'legal'
        elif food_score > agricultural_score and food_score > 0:
            return 'food_industry'
        elif agricultural_score > food_score and agricultural_score > 0:
            return 'agricultural'
        elif mixed_score > 0 or (food_score > 0 and agricultural_score > 0) or (legal_score > 0 and (food_score > 0 or agricultural_score > 0)):
            return 'mixed'
        else:
            return 'agricultural'  # Default

    def enhance_query_with_entities(self, query: str, project_id: Optional[str] = None,
                                  document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhance a user query by identifying relevant entities and building context.
        
        Args:
            query: The original user query
            project_id: Optional project ID to scope entity search
            document_id: Optional document ID to focus entity search
            
        Returns:
            Dictionary with enhanced query and metadata
        """
        try:
            # Use cache if available
            cache_key = f"{query}:{project_id}:{document_id}"
            if cache_key in self.query_cache:
                logger.debug("Using cached query enhancement")
                return self.query_cache[cache_key]
            
            # Step 0: Classify query domain
            query_domain = self.classify_query_domain(query)
            logger.info(f"Query domain classified as: {query_domain}")
            
            # Step 1: Classify query intent
            query_intent = self._classify_query_intent(query)
            logger.info(f"Query intent classified as: {query_intent['primary_intent']} (confidence: {query_intent['confidence']:.2f})")
            
            # Step 2: Extract entity mentions from the query
            entity_mentions = self._extract_entity_mentions_from_query(query, query_domain)
            logger.info(f"Found {len(entity_mentions)} entity mentions in query")
            
            # Step 3: Get relevant entities from database
            relevant_entities = self._get_relevant_entities(query, entity_mentions, project_id, document_id, query_domain)
            logger.info(f"Retrieved {len(relevant_entities)} relevant entities from database")
            
            # Step 4: Find semantic matches and score relevance
            entity_matches = self._find_semantic_entity_matches(query, relevant_entities, query_intent)
            logger.info(f"Found {len(entity_matches)} semantic entity matches")
            
            # Step 5: Build enhanced query
            enhanced_query = self._build_enhanced_query(query, entity_matches, query_intent)
            
            # Step 6: Generate expansion terms for better retrieval
            expansion_terms = self._generate_expansion_terms(entity_matches, query_intent, query_domain)
            
            # Step 7: Suggest query filters for improved precision
            suggested_filters = self._suggest_query_filters(entity_matches, query_domain)
            
            # Step 8: Calculate enhancement confidence
            enhancement_confidence = self._calculate_enhancement_confidence(entity_matches)
            
            # Build result
            result = {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "expansion_terms": expansion_terms,
                "suggested_filters": suggested_filters,
                "query_intent": query_intent,
                "query_domain": query_domain,
                "entity_matches": entity_matches[:10],  # Limit to top 10 for performance
                "enhancement_confidence": enhancement_confidence,
                "processing_metadata": {
                    "total_entities_considered": len(relevant_entities),
                    "semantic_matches_found": len(entity_matches),
                    "expansion_terms_count": len(expansion_terms),
                    "enhancement_applied": enhancement_confidence > 0.3
                }
            }
            
            # Cache result
            self.query_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return {
                "original_query": query,
                "enhanced_query": query,  # Fallback to original
                "expansion_terms": [],
                "suggested_filters": {},
                "query_intent": {"primary_intent": "general", "confidence": 0.0},
                "query_domain": "agricultural",
                "entity_matches": [],
                "enhancement_confidence": 0.0,
                "processing_metadata": {"error": str(e)}
            }

    def _classify_query_intent(self, query: str) -> Dict[str, Any]:
        """Classify the intent of the user query."""
        intent_scores = {}
        
        for intent, patterns in self.compiled_patterns.items():
            score = 0.0
            matches = []
            
            for pattern in patterns:
                match = pattern.search(query)
                if match:
                    # Weight by match strength and position
                    match_strength = len(match.group(0)) / len(query)
                    position_weight = 1.0 - (match.start() / len(query)) * 0.3  # Earlier matches weighted higher
                    score += match_strength * position_weight
                    matches.append(match.group(0))
            
            if score > 0:
                intent_scores[intent] = {
                    "score": min(score, 1.0),  # Cap at 1.0
                    "matches": matches
                }
        
        # Determine primary intent
        if intent_scores:
            primary_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k]["score"])
            confidence = intent_scores[primary_intent]["score"]
        else:
            primary_intent = "general"
            confidence = 0.0
        
        return {
            "primary_intent": primary_intent,
            "confidence": confidence,
            "all_intents": intent_scores,
            "is_multi_intent": len([k for k, v in intent_scores.items() if v["score"] > 0.3]) > 1
        }

    def _extract_entity_mentions_from_query(self, query: str, query_domain: str) -> List[Dict[str, Any]]:
        """Extract potential entity mentions from the user query."""
        mentions = []
        query_words = query.lower().split()
        
        # Enhanced entity patterns for both agricultural and food industry
        entity_patterns = {
            'product_name': r'\b(?:product|brand|formulation|ingredient)\s+([A-Za-z0-9\-\s]+)',
            'chemical_compound': r'\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\s+(?:compound|chemical|acid|salt)',
            'crop_name': r'\b(?:crop|plant|variety|cultivar)\s+([A-Za-z\s]+)',
            'application_method': r'\b(?:spray|apply|use|mix)\s+([A-Za-z\s]+)',
            'measurement': r'(\d+(?:\.\d+)?)\s*(kg|g|ml|l|ppm|%|percent)',
            
            # Food industry specific patterns
            'food_ingredient': r'\b(?:ingredient|additive|preservative|stabilizer|emulsifier)\s+([A-Za-z0-9\-\s]+)',
            'vitamin_mineral': r'\b(vitamin\s+[A-K]\d*|calcium|iron|zinc|magnesium|potassium)',
            'allergen': r'\b(dairy|gluten|nuts?|soy|eggs?|fish|shellfish|wheat|sesame)',
            'food_category': r'\b(bakery|dairy|beverage|meat|confectionery|snack|sauce)\s+(?:product|application)',
            'certification': r'\b(organic|kosher|halal|non-gmo|gras|fda\s+approved)',
            'e_number': r'\bE\s*(\d{3,4})\b',
            'processing_method': r'\b(spray\s+dry|freeze\s+dry|extract|ferment|encapsulat)',
        }
        
        # Extract patterns based on domain
        relevant_patterns = entity_patterns
        if query_domain == 'agricultural':
            # Focus on agricultural patterns
            relevant_patterns = {k: v for k, v in entity_patterns.items() 
                               if k in ['product_name', 'chemical_compound', 'crop_name', 'application_method', 'measurement']}
        elif query_domain == 'food_industry':
            # Focus on food industry patterns
            relevant_patterns = {k: v for k, v in entity_patterns.items() 
                               if k in ['food_ingredient', 'vitamin_mineral', 'allergen', 'food_category', 'certification', 'e_number', 'processing_method', 'measurement']}
        
        for pattern_name, pattern in relevant_patterns.items():
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                mention_text = match.group(1) if match.groups() else match.group(0)
                mentions.append({
                    'text': mention_text.strip(),
                    'type': pattern_name,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        # Extract entities by looking for known entity examples from config
        entity_types = self.config.get('entity_types', {})
        for entity_type, entity_config in entity_types.items():
            examples = entity_config.get('examples', [])
            for example in examples:
                if example.lower() in query.lower():
                    # Find position
                    pos = query.lower().find(example.lower())
                    mentions.append({
                        'text': example,
                        'type': 'known_entity',
                        'entity_type': entity_type,
                        'start': pos,
                        'end': pos + len(example),
                        'confidence': 0.9
                    })
        
        # Remove overlapping mentions
        mentions = self._remove_overlapping_mentions(mentions)
        
        return sorted(mentions, key=lambda x: x['confidence'], reverse=True)

    def _get_relevant_entities(self, query: str, mentions: List[Dict[str, Any]],
                              project_id: Optional[str], document_id: Optional[str], query_domain: str) -> List[Dict[str, Any]]:
        """Retrieve relevant entities from the database."""
        entities = []
        
        try:
            # Build search terms from mentions and query
            search_terms = [mention['text'] for mention in mentions]
            search_terms.extend(query.split())
            search_terms = [term for term in search_terms if len(term) > 2]
            
            # Search in entities table
            for term in search_terms[:5]:  # Limit to avoid too many queries
                entity_query = """
                SELECT DISTINCT e.*, d.filename 
                FROM entities e 
                JOIN documents d ON e.document_id = d.id 
                WHERE (e.entity_value ILIKE %s OR e.normalized_value ILIKE %s)
                """
                
                params = [f"%{term}%", f"%{term}%"]
                
                # Add project filter if specified
                if project_id:
                    entity_query += " AND d.project_id = %s"
                    params.append(project_id)
                
                # Add document filter if specified
                if document_id:
                    entity_query += " AND e.document_id = %s"
                    params.append(document_id)
                
                # Add domain-specific filtering
                if query_domain == 'food_industry':
                    entity_query += " AND (e.entity_type LIKE 'FOOD_%' OR e.entity_type IN ('NUTRITIONAL_COMPONENT', 'ALLERGEN_INFO'))"
                elif query_domain == 'agricultural':
                    entity_query += " AND (e.entity_type NOT LIKE 'FOOD_%' OR e.entity_type IS NULL)"
                
                entity_query += " ORDER BY e.confidence_score DESC LIMIT 20"
                
                try:
                    results = self.db.execute_query(entity_query, params)
                    for result in results:
                        entities.append(dict(result))
                except Exception as e:
                    logger.warning(f"Database query failed for term '{term}': {e}")
                    continue
            
            # For food industry queries, also search food-specific tables
            if query_domain in ['food_industry', 'mixed'] and search_terms:
                food_entity_query = """
                SELECT e.*, fie.food_type, fie.ingredient_category, fie.allergen_info, fie.regulatory_status, d.filename
                FROM entities e
                JOIN food_industry_entities fie ON e.id = fie.entity_id
                JOIN documents d ON e.document_id = d.id
                WHERE e.entity_value ILIKE %s OR fie.functional_class ILIKE %s
                """
                
                params = [f"%{search_terms[0]}%", f"%{search_terms[0]}%"]
                
                if project_id:
                    food_entity_query += " AND d.project_id = %s"
                    params.append(project_id)
                
                food_entity_query += " ORDER BY e.confidence_score DESC LIMIT 10"
                
                try:
                    food_results = self.db.execute_query(food_entity_query, params)
                    for result in food_results:
                        food_entity = dict(result)
                        food_entity['is_food_entity'] = True
                        entities.append(food_entity)
                except Exception as e:
                    logger.warning(f"Food entity query failed: {e}")
            
            # Use frequency-based entity search for common entities
            if project_id:
                frequent_entities = self._get_frequent_entities(project_id)
                for entity_value in frequent_entities[:10]:
                    if any(term.lower() in entity_value.lower() for term in search_terms):
                        # Add frequent entity info
                        entities.append({
                            'entity_value': entity_value,
                            'entity_type': 'FREQUENT',
                            'confidence_score': 0.7,
                            'source': 'frequency_based'
                        })
                        
        except Exception as e:
            logger.error(f"Error retrieving entities: {e}")
        
        # Remove duplicates based on entity_value
        seen = set()
        unique_entities = []
        for entity in entities:
            entity_key = entity.get('entity_value', '').lower()
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities[:50]  # Limit results

    def _find_semantic_entity_matches(self, query: str, entities: List[Dict[str, Any]],
                                    query_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find and score semantic matches between query and entities."""
        matches = []
        primary_intent = query_intent.get("primary_intent", "general")
        
        for entity in entities:
            entity_type = entity.get("type", "")
            entity_value = entity.get("value", "")
            entity_context = entity.get("context", "")
            
            # Calculate multiple relevance scores
            match_scores = {}
            
            # 1. Intent relevance
            match_scores["intent_relevance"] = self._calculate_intent_relevance(entity_type, primary_intent)
            
            # 2. Text similarity
            query_lower = query.lower()
            value_lower = entity_value.lower()
            
            if value_lower in query_lower:
                match_scores["text_similarity"] = 1.0
            elif any(word in query_lower for word in value_lower.split()):
                match_scores["text_similarity"] = 0.7
            else:
                # Use sequence matcher for partial similarity
                similarity = SequenceMatcher(None, query_lower, value_lower).ratio()
                match_scores["text_similarity"] = similarity if similarity > 0.3 else 0.0
            
            # 3. Pattern relevance
            match_scores["pattern_relevance"] = self._calculate_pattern_relevance(entity_value, query)
            
            # 4. Context relevance
            if entity_context:
                context_words = set(entity_context.lower().split())
                query_words = set(query_lower.split())
                common_words = context_words.intersection(query_words)
                match_scores["context_relevance"] = len(common_words) / max(len(query_words), 1) if query_words else 0.0
            else:
                match_scores["context_relevance"] = 0.0
            
            # 5. Frequency relevance (more frequent entities are often more important)
            entity_frequency = entity.get("frequency", 1)
            max_frequency = max([e.get("frequency", 1) for e in entities]) if entities else 1
            match_scores["frequency_relevance"] = entity_frequency / max_frequency if max_frequency > 0 else 0.0
            
            # Calculate weighted overall score
            weights = {
                "intent_relevance": 0.3,
                "text_similarity": 0.4,
                "pattern_relevance": 0.15,
                "context_relevance": 0.1,
                "frequency_relevance": 0.05
            }
            
            overall_score = sum(
                match_scores.get(score_type, 0.0) * weight
                for score_type, weight in weights.items()
            )
            
            # Only include entities with reasonable relevance
            if overall_score > 0.2:
                matches.append({
                    "entity": entity,
                    "overall_score": overall_score,
                    "match_scores": match_scores,
                    "relevance_explanation": self._explain_relevance(match_scores)
                })
        
        # Sort by overall score (descending)
        matches.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return matches

    def _build_enhanced_query(self, original_query: str, matches: List[Dict[str, Any]],
                            query_intent: Dict[str, Any]) -> str:
        """Build an enhanced query with entity context."""
        if not matches:
            return original_query
        
        # Start with original query
        enhanced_parts = [original_query]
        
        # Add top entity contexts for better retrieval
        top_matches = matches[:3]  # Top 3 entities
        
        for match in top_matches:
            entity = match["entity"]
            entity_type = entity.get("type", "")
            entity_value = entity.get("value", "")
            
            # Add entity-specific context based on type
            if entity_type == "PRODUCT" and match["overall_score"] > 0.7:
                enhanced_parts.append(f"product information for {entity_value}")
            elif entity_type == "CROP" and match["overall_score"] > 0.6:
                enhanced_parts.append(f"crop-specific information for {entity_value}")
            elif entity_type == "APPLICATION" and match["overall_score"] > 0.6:
                enhanced_parts.append(f"application methods and {entity_value}")
            elif entity_type == "CHEMICAL_COMPOUND" and match["overall_score"] > 0.7:
                enhanced_parts.append(f"chemical information about {entity_value}")
        
        # Join with improved weighting for search
        enhanced_query = " | ".join(enhanced_parts)
        
        return enhanced_query

    def _generate_expansion_terms(self, matches: List[Dict[str, Any]],
                                 query_intent: Dict[str, Any], query_domain: str) -> List[str]:
        """Generate expansion terms to improve retrieval coverage."""
        expansion_terms = []
        
        # Intent-based expansion terms
        intent_expansions = {
            'agricultural': {
                'product_information': ['specification', 'datasheet', 'technical', 'properties'],
                'application_instructions': ['rate', 'method', 'timing', 'instructions', 'guidelines'],
                'efficacy_performance': ['yield', 'effectiveness', 'results', 'performance', 'trial'],
                'safety_regulatory': ['safety', 'ghs', 'hazard', 'regulation', 'approval'],
                'compatibility_mixing': ['compatible', 'tank mix', 'combination', 'mixing'],
                'crop_specific': ['crop', 'variety', 'cultivar', 'species'],
                'technical_specs': ['analysis', 'parameter', 'specification', 'property'],
                'comparison': ['compare', 'alternative', 'versus', 'similar']
            },
            'food_industry': {
                'ingredient_information': ['datasheet', 'specification', 'properties', 'composition'],
                'nutritional_inquiry': ['nutrition', 'vitamin', 'mineral', 'content', 'value'],
                'allergen_safety': ['allergen', 'safe', 'free', 'declaration', 'labeling'],
                'food_regulatory': ['gras', 'approved', 'certified', 'compliant', 'regulation'],
                'food_application': ['application', 'use', 'suitable', 'bakery', 'dairy'],
                'food_processing': ['processing', 'stability', 'storage', 'shelf life'],
                'substitution_alternatives': ['alternative', 'substitute', 'replacement', 'similar']
            },
            'legal': {
                'contract_analysis': ['clause', 'provision', 'term', 'obligation', 'condition'],
                'compliance_inquiry': ['requirement', 'mandate', 'regulation', 'standard', 'compliance'],
                'liability_warranty': ['liability', 'warranty', 'indemnity', 'damages', 'responsibility'],
                'intellectual_property': ['patent', 'trademark', 'copyright', 'confidential', 'proprietary'],
                'termination_dispute': ['termination', 'breach', 'dispute', 'arbitration', 'resolution']
            }
        }
        
        # Get domain-specific expansions
        domain_expansions = intent_expansions.get(query_domain, intent_expansions['agricultural'])
        primary_intent = query_intent.get('primary_intent', 'general')
        
        if primary_intent in domain_expansions:
            expansion_terms.extend(domain_expansions[primary_intent])
        
        # Entity-based expansion terms
        for match in matches[:5]:  # Limit to top 5 matches
            entity_type = match.get('entity_type', '')
            entity_value = match.get('entity_value', '')
            
            # Agricultural entity expansions
            if entity_type == 'PRODUCT':
                expansion_terms.extend(['application', 'rate', 'instructions', 'specification'])
            elif entity_type == 'CROP':
                expansion_terms.extend(['variety', 'cultivation', 'growth', 'harvest'])
            elif entity_type == 'CHEMICAL_COMPOUND':
                expansion_terms.extend(['formula', 'properties', 'safety', 'analysis'])
            
            # Food industry entity expansions
            elif entity_type == 'FOOD_INGREDIENT':
                expansion_terms.extend(['specification', 'application', 'usage', 'properties'])
            elif entity_type == 'NUTRITIONAL_COMPONENT':
                expansion_terms.extend(['content', 'value', 'bioavailability', 'daily value'])
            elif entity_type == 'ALLERGEN_INFO':
                expansion_terms.extend(['declaration', 'labeling', 'cross contamination', 'free'])
            elif entity_type == 'FOOD_SAFETY_STANDARD':
                expansion_terms.extend(['approved', 'certified', 'compliant', 'regulation'])
            elif entity_type == 'FOOD_APPLICATION':
                expansion_terms.extend(['suitable', 'use', 'processing', 'manufacturing'])
            
            # Add synonyms and related terms from entity value
            if len(entity_value) > 3:
                # Simple synonym expansion (could be enhanced with a proper thesaurus)
                synonyms = self._get_entity_synonyms(entity_value, entity_type)
                expansion_terms.extend(synonyms)
        
        # Domain-specific general expansion terms
        if query_domain == 'food_industry':
            expansion_terms.extend(['food grade', 'kosher', 'halal', 'organic', 'non-gmo'])
        elif query_domain == 'agricultural':
            expansion_terms.extend(['field', 'farm', 'crop protection', 'yield'])
        
        # Remove duplicates and empty terms
        expansion_terms = [term for term in expansion_terms if term and len(term) > 1]
        return list(set(expansion_terms))[:10]

    def _suggest_query_filters(self, matches: List[Dict[str, Any]], query_domain: str) -> Dict[str, List[str]]:
        """Suggest filters to improve query precision."""
        filters = {
            'entity_types': [],
            'document_types': [],
            'confidence_threshold': 0.5,
            'domain_specific': []
        }
        
        # Extract entity types from matches
        entity_types = list(set([match.get('entity_type', '') for match in matches if match.get('entity_type')]))
        filters['entity_types'] = entity_types
        
        # Suggest document types based on entity types and domain
        if query_domain == 'food_industry':
            food_doc_types = ['food_sds', 'nutritional_info', 'ingredient_spec', 'coa', 'allergen_declaration', 'regulatory_compliance']
            filters['document_types'].extend(food_doc_types)
            
            # Food industry specific filters
            if any('ALLERGEN' in et for et in entity_types):
                filters['domain_specific'].append('allergen_free')
            if any('NUTRITIONAL' in et for et in entity_types):
                filters['domain_specific'].append('nutritional_data')
            if any('FOOD_SAFETY' in et for et in entity_types):
                filters['domain_specific'].append('regulatory_approved')
        
        elif query_domain == 'legal':
            legal_doc_types = ['contract', 'agreement', 'legal_document', 'compliance_doc', 'regulatory_filing']
            filters['document_types'].extend(legal_doc_types)
            
            # Legal specific filters
            if any('LEGAL_' in et for et in entity_types):
                filters['domain_specific'].append('legal_analysis')
            if any('CONTRACT' in et or 'AGREEMENT' in et for et in entity_types):
                filters['domain_specific'].append('contract_terms')
            if any('COMPLIANCE' in et or 'REGULATORY' in et for et in entity_types):
                filters['domain_specific'].append('regulatory_compliance')
        
        elif query_domain == 'agricultural':
            agri_doc_types = ['agricultural', 'general']
            filters['document_types'].extend(agri_doc_types)
            
            # Agricultural specific filters
            if any(et in ['PRODUCT', 'CHEMICAL_COMPOUND'] for et in entity_types):
                filters['domain_specific'].append('product_specs')
            if 'CROP' in entity_types:
                filters['domain_specific'].append('crop_specific')
        
        else:  # mixed domain
            filters['document_types'].extend(['agricultural', 'food_sds', 'legal_document', 'general'])
        
        # Adjust confidence threshold based on match quality
        if matches:
            avg_confidence = sum(match.get('confidence_score', 0.5) for match in matches) / len(matches)
            filters['confidence_threshold'] = max(0.3, avg_confidence - 0.2)
        
        return filters

    def _calculate_enhancement_confidence(self, matches: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the query enhancement."""
        if not matches:
            return 0.0
        
        # Consider top 3 matches
        top_matches = matches[:3]
        scores = [match["overall_score"] for match in top_matches]
        
        # Weighted average with diminishing returns
        weights = [0.5, 0.3, 0.2]
        weighted_score = sum(
            score * weight for score, weight in zip(scores, weights[:len(scores)])
        )
        
        # Bonus for multiple good matches
        good_matches = len([score for score in scores if score > 0.5])
        bonus = min(good_matches * 0.1, 0.3)
        
        return min(weighted_score + bonus, 1.0)

    def _calculate_intent_relevance(self, entity_type: str, primary_intent: str) -> float:
        """Calculate how relevant an entity type is to the query intent."""
        relevance_map = self.intent_entity_relevance.get(primary_intent, {})
        return relevance_map.get(entity_type, 0.3)  # Default low relevance

    def _calculate_pattern_relevance(self, entity_value: str, query: str) -> float:
        """Calculate relevance based on pattern matching."""
        entity_words = set(entity_value.lower().split())
        query_words = set(query.lower().split())
        
        # Jaccard similarity
        intersection = entity_words.intersection(query_words)
        union = entity_words.union(query_words)
        
        if not union:
            return 0.0
        
        jaccard = len(intersection) / len(union)
        
        # Boost for exact phrase matches
        if entity_value.lower() in query.lower():
            jaccard += 0.5
        
        return min(jaccard, 1.0)

    def _explain_relevance(self, match_scores: Dict[str, float]) -> str:
        """Generate human-readable explanation of relevance scoring."""
        explanations = []
        
        if match_scores.get("text_similarity", 0) > 0.8:
            explanations.append("exact text match")
        elif match_scores.get("text_similarity", 0) > 0.5:
            explanations.append("partial text match")
        
        if match_scores.get("intent_relevance", 0) > 0.8:
            explanations.append("highly relevant to query intent")
        elif match_scores.get("intent_relevance", 0) > 0.5:
            explanations.append("moderately relevant to query intent")
        
        if match_scores.get("context_relevance", 0) > 0.5:
            explanations.append("relevant context")
        
        if match_scores.get("frequency_relevance", 0) > 0.7:
            explanations.append("frequently mentioned entity")
        
        return "; ".join(explanations) if explanations else "general relevance"

    def _extract_intent_keywords(self, query: str, intent_type: str) -> List[str]:
        """Extract keywords relevant to the classified intent."""
        keywords = []
        
        intent_keyword_map = {
            "product_information": ["product", "ingredient", "composition", "specification", "brand"],
            "application_instructions": ["apply", "use", "rate", "timing", "method", "how"],
            "efficacy_performance": ["effective", "yield", "result", "performance", "trial"],
            "safety_regulatory": ["safety", "hazard", "ghs", "regulation", "approved"],
            "compatibility_mixing": ["compatible", "mix", "tank", "combine"],
            "crop_specific": ["crop", "variety", "plant", "species"],
            "technical_specs": ["specification", "parameter", "analysis", "test"],
            "comparison": ["compare", "versus", "better", "difference"]
        }
        
        intent_keywords = intent_keyword_map.get(intent_type, [])
        query_lower = query.lower()
        
        for keyword in intent_keywords:
            if keyword in query_lower:
                keywords.append(keyword)
        
        return keywords

    def _remove_overlapping_mentions(self, mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping entity mentions, keeping the highest confidence ones."""
        if not mentions:
            return mentions
        
        # Sort by confidence (descending)
        mentions.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        filtered_mentions = []
        
        for mention in mentions:
            start, end = mention["start"], mention["end"]
            
            # Check if this mention overlaps with any already accepted mention
            overlaps = False
            for accepted in filtered_mentions:
                acc_start, acc_end = accepted["start"], accepted["end"]
                
                # Check for overlap
                if not (end <= acc_start or start >= acc_end):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_mentions.append(mention)
        
        return filtered_mentions

    def _get_frequent_entities(self, project_id: str, limit: int = 20) -> List[str]:
        """Get frequently occurring entities in the project."""
        try:
            # Use the corrected method name
            entities = self.db.get_frequent_entities(project_id=project_id, limit=limit)
            
            # Extract entity values
            frequent_entities = []
            for entity in entities:
                entity_value = entity.get('entity_value', '')
                if entity_value and entity_value not in frequent_entities:
                    frequent_entities.append(entity_value)
            
            return frequent_entities[:limit]
            
        except Exception as e:
            logger.warning(f"Could not retrieve frequent entities: {e}")
            return []
    
    def _search_similar_entities(self, entity_text: str, project_id: str, limit: int = 10) -> List[str]:
        """Search for entities similar to the given text."""
        try:
            # Use the corrected method name  
            entities = self.db.search_entities_by_similarity(entity_text, project_id=project_id, limit=limit)
            
            # Extract entity values
            similar_entities = []
            for entity in entities:
                entity_value = entity.get('entity_value', '')
                if entity_value and entity_value not in similar_entities:
                    similar_entities.append(entity_value)
            
            return similar_entities[:limit]
            
        except Exception as e:
            logger.warning(f"Could not search similar entities: {e}")
            return []

    def _get_entity_synonyms(self, entity_value: str, entity_type: str) -> List[str]:
        """Get synonyms and related terms for an entity."""
        synonyms = []
        
        # Simple synonym mapping for common terms
        synonym_map = {
            # Agricultural synonyms
            'nitrogen': ['n', 'nitrate', 'ammonium'],
            'phosphorus': ['p', 'phosphate', 'p2o5'],
            'potassium': ['k', 'potash', 'k2o'],
            'fertilizer': ['nutrient', 'plant food', 'amendment'],
            'pesticide': ['crop protection', 'pest control', 'chemical'],
            
            # Food industry synonyms
            'vitamin c': ['ascorbic acid', 'l-ascorbic acid', 'ascorbate'],
            'vitamin e': ['tocopherol', 'alpha-tocopherol', 'd-alpha-tocopherol'],
            'preservative': ['antimicrobial', 'antioxidant', 'stabilizer'],
            'emulsifier': ['surfactant', 'lecithin', 'stabilizer'],
            'sweetener': ['sugar substitute', 'artificial sweetener', 'low calorie sweetener'],
            'thickener': ['hydrocolloid', 'gelling agent', 'viscosity modifier'],
            'colorant': ['food coloring', 'food dye', 'natural color'],
            'flavoring': ['flavor enhancer', 'natural flavor', 'artificial flavor']
        }
        
        entity_lower = entity_value.lower()
        for key, values in synonym_map.items():
            if key in entity_lower or entity_lower in key:
                synonyms.extend(values)
        
        # Add partial matches for compound names
        words = entity_value.split()
        if len(words) > 1:
            synonyms.extend(words)
        
        return synonyms[:3]  # Limit to 3 synonyms

    def search_food_ingredients(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Specialized search for food ingredients with B2B focus.
        
        Args:
            query: Search query for food ingredients
            filters: Optional filters (allergen_free, organic, food_grade, etc.)
            
        Returns:
            Comprehensive ingredient search results
        """
        if not filters:
            filters = {}
        
        logger.info(f"Searching food ingredients for: {query}")
        
        # Enhance query with food industry focus
        enhanced_query_result = self.enhance_query_with_entities(query)
        
        # Extract food-specific information
        results = {
            'query': query,
            'domain': 'food_industry',
            'ingredients': [],
            'nutritional_data': [],
            'allergen_information': [],
            'regulatory_compliance': [],
            'applications': [],
            'total_results': 0,
            'filters_applied': filters,
            'search_metadata': enhanced_query_result
        }
        
        try:
            # Search food industry entities
            food_search_query = """
            SELECT e.*, fie.*, d.filename, d.content_type
            FROM entities e
            JOIN food_industry_entities fie ON e.id = fie.entity_id
            JOIN documents d ON e.document_id = d.id
            WHERE (e.entity_value ILIKE %s OR fie.functional_class ILIKE %s)
            """
            
            params = [f"%{query}%", f"%{query}%"]
            
            # Apply filters
            if filters.get('food_grade'):
                food_search_query += " AND fie.food_grade = true"
            
            if filters.get('organic'):
                food_search_query += " AND fie.organic_certified = true"
            
            if filters.get('allergen_free'):
                allergens = filters['allergen_free']
                if isinstance(allergens, list):
                    # Exclude ingredients containing specified allergens
                    food_search_query += " AND NOT EXISTS (SELECT 1 FROM allergen_information ai WHERE ai.entity_id = e.id AND ai.allergen_type = ANY(%s) AND ai.declaration_type = 'contains')"
                    params.append(allergens)
            
            if filters.get('regulatory_status'):
                food_search_query += " AND fie.regulatory_status ILIKE %s"
                params.append(f"%{filters['regulatory_status']}%")
            
            food_search_query += " ORDER BY e.confidence_score DESC LIMIT 50"
            
            search_results = self.db.execute_query(food_search_query, params)
            
            for result in search_results:
                ingredient_data = dict(result)
                results['ingredients'].append(ingredient_data)
            
            results['total_results'] = len(results['ingredients'])
            
            # Get nutritional information for found ingredients
            if results['ingredients']:
                entity_ids = [ing['id'] for ing in results['ingredients'] if 'id' in ing]
                if entity_ids:
                    nutrition_query = """
                    SELECT ni.*, e.entity_value
                    FROM nutritional_information ni
                    JOIN entities e ON ni.entity_id = e.id
                    WHERE ni.entity_id = ANY(%s)
                    ORDER BY ni.content_value DESC
                    """
                    
                    nutrition_results = self.db.execute_query(nutrition_query, [entity_ids])
                    results['nutritional_data'] = [dict(r) for r in nutrition_results]
                    
                    # Get allergen information
                    allergen_query = """
                    SELECT ai.*, e.entity_value
                    FROM allergen_information ai
                    JOIN entities e ON ai.entity_id = e.id
                    WHERE ai.entity_id = ANY(%s)
                    """
                    
                    allergen_results = self.db.execute_query(allergen_query, [entity_ids])
                    results['allergen_information'] = [dict(r) for r in allergen_results]
            
            # Add search insights
            results['insights'] = {
                'search_complexity': len(enhanced_query_result.get('entity_matches', [])),
                'domain_relevance': 'high' if 'food' in query.lower() else 'medium',
                'regulatory_focus': any(term in query.lower() for term in ['gras', 'fda', 'approved', 'certified']),
                'allergen_focus': any(term in query.lower() for term in ['allergen', 'dairy', 'gluten', 'nut']),
                'nutrition_focus': any(term in query.lower() for term in ['vitamin', 'mineral', 'protein', 'nutrition'])
            }
            
            # Generate recommendations
            recommendations = []
            if results['allergen_information']:
                recommendations.append("Review allergen declarations for labeling compliance")
            if any('gras' in ing.get('regulatory_status', '').lower() for ing in results['ingredients']):
                recommendations.append("GRAS-approved ingredients found - suitable for US market")
            if len(results['ingredients']) > 10:
                recommendations.append("Consider refining search criteria for more specific results")
            
            results['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Error in food ingredient search: {e}")
            results['error'] = str(e)
        
        return results 