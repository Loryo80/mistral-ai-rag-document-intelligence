# retriever.py (Enhanced Agricultural Retrieval Module - Fixed Version)
import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgriculturalRetriever:
    """Enhanced retrieval system optimized for agricultural/product fiche documents."""
    
    def __init__(self, db_client_instance, embedding_generator_func, working_dir: str = "rag_storage"):
        """
        Initialize the agricultural retriever with optional LightRAG integration.

        Args:
            db_client_instance: Database client for traditional retrieval
            embedding_generator_func: Function to generate embeddings
            working_dir: Directory for LightRAG storage
        """
        self.db = db_client_instance
        self.embedding_generator = embedding_generator_func
        self.working_dir = working_dir
        self.lightrag = None
        self.lightrag_available = False
        
        # Create working directory
        os.makedirs(working_dir, exist_ok=True)
        
        # Try to initialize LightRAG with better error handling
        self._initialize_lightrag()
        
        # Domain-specific keywords for query enhancement
        self.domain_keywords = {
            'agricultural': {
                'crops': ['wheat', 'corn', 'rice', 'soybean', 'barley', 'oats', 'canola', 'sunflower'],
                'chemicals': ['herbicide', 'pesticide', 'fungicide', 'insecticide', 'fertilizer', 'nutrient'],
                'processes': ['application', 'spraying', 'treatment', 'cultivation', 'harvest'],
                'measurements': ['concentration', 'dosage', 'rate', 'ph', 'temperature', 'humidity'],
                'safety': ['toxicity', 'hazard', 'precaution', 'exposure', 'protective equipment']
            },
            'food_industry': {
                'ingredients': ['additive', 'preservative', 'stabilizer', 'emulsifier', 'thickener', 'sweetener'],
                'nutrition': ['vitamin', 'mineral', 'protein', 'carbohydrate', 'fat', 'fiber', 'calorie'],
                'allergens': ['dairy', 'gluten', 'nuts', 'soy', 'eggs', 'fish', 'shellfish', 'sesame'],
                'processing': ['pasteurization', 'sterilization', 'fermentation', 'extraction', 'encapsulation'],
                'regulatory': ['gras', 'fda', 'efsa', 'kosher', 'halal', 'organic', 'non-gmo'],
                'applications': ['bakery', 'dairy', 'beverage', 'confectionery', 'meat', 'snack']
            },
            'legal': {
                'contract_terms': ['obligation', 'liability', 'warranty', 'indemnity', 'breach', 'termination'],
                'ip_rights': ['patent', 'trademark', 'copyright', 'trade secret', 'confidential', 'proprietary'],
                'compliance': ['regulatory', 'standard', 'requirement', 'mandate', 'certification', 'approval'],
                'dispute_resolution': ['arbitration', 'mediation', 'litigation', 'jurisdiction', 'governing law'],
                'agreement_types': ['license', 'supply', 'distribution', 'service', 'employment', 'joint venture']
            }
        }
    
    def _initialize_lightrag(self):
        """Initialize LightRAG with graceful fallback if dependencies are missing."""
        try:
            # Check if LightRAG is available - use basic import only
            try:
                from lightrag import LightRAG, QueryParam
                logger.info("LightRAG basic imports found, attempting initialization...")
            except ImportError as import_error:
                logger.info(f"LightRAG not available: {import_error}")
                self.lightrag_available = False
                return
            
            # Try to initialize with minimal configuration and timeout
            try:
                logger.info("Initializing LightRAG with minimal configuration...")
                
                # Use minimal initialization without external LLM/embedding functions
                # LightRAG will use its internal defaults
                self.lightrag = LightRAG(working_dir=self.working_dir)
                
                self.lightrag_available = True
                logger.info("LightRAG initialized successfully")
                
            except Exception as init_error:
                logger.warning(f"LightRAG initialization failed: {init_error}")
                self.lightrag_available = False
                self.lightrag = None
                
        except Exception as e:
            logger.error(f"Error during LightRAG setup: {e}")
            self.lightrag_available = False
            self.lightrag = None
    
    def enhance_domain_query(self, query: str, domain: str = 'auto') -> str:
        """Enhance query with domain-specific context."""
        query_lower = query.lower()
        
        # Auto-detect domain if not specified
        if domain == 'auto':
            domain = self._detect_query_domain(query_lower)
        
        # Add context for domain-specific terms
        enhancements = []
        
        if domain in self.domain_keywords:
            for category, keywords in self.domain_keywords[domain].items():
                for keyword in keywords:
                    if keyword in query_lower:
                        if domain == 'agricultural':
                            if category == 'crops':
                                enhancements.append(f"agricultural crop {keyword}")
                            elif category == 'chemicals':
                                enhancements.append(f"agricultural chemical {keyword}")
                            elif category == 'processes':
                                enhancements.append(f"farming process {keyword}")
                        elif domain == 'food_industry':
                            if category == 'ingredients':
                                enhancements.append(f"food ingredient {keyword}")
                            elif category == 'nutrition':
                                enhancements.append(f"nutritional component {keyword}")
                            elif category == 'regulatory':
                                enhancements.append(f"food regulation {keyword}")
                        elif domain == 'legal':
                            if category == 'contract_terms':
                                enhancements.append(f"legal provision {keyword}")
                            elif category == 'ip_rights':
                                enhancements.append(f"intellectual property {keyword}")
                            elif category == 'compliance':
                                enhancements.append(f"regulatory compliance {keyword}")
        
        if enhancements:
            enhanced_query = f"{query} Context: {', '.join(enhancements[:3])}"
            logger.info(f"Enhanced query with {domain} context")
            return enhanced_query
        
        return query
    
    def _detect_query_domain(self, query_lower: str) -> str:
        """Detect the most likely domain for the query."""
        domain_scores = {'agricultural': 0, 'food_industry': 0, 'legal': 0}
        
        for domain, categories in self.domain_keywords.items():
            for category, keywords in categories.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        domain_scores[domain] += 1
        
        # Return domain with highest score, default to agricultural
        return max(domain_scores, key=domain_scores.get) if max(domain_scores.values()) > 0 else 'agricultural'
    
    def query(self, query: str, mode: str = "hybrid", document_id: Optional[str] = None, 
              project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Query documents using multiple retrieval strategies.

        Args:
            query: User's query
            mode: Retrieval mode ("local", "global", "hybrid")
            document_id: Optional specific document
            project_id: Optional project filter
            
        Returns:
            Dictionary with retrieved context and metadata
        """
        try:
            # Enhance query with domain-specific context
            enhanced_query = self.enhance_domain_query(query)
            
            # Try LightRAG first if available
            if self.lightrag_available and self.lightrag:
                try:
                    return self._query_lightrag(enhanced_query, mode)
                except Exception as e:
                    logger.warning(f"LightRAG query failed: {e}, falling back to traditional retrieval")
            
            # Fallback to traditional database retrieval
            return self._query_traditional(enhanced_query, document_id, project_id)
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "context": [f"I encountered an error processing your query. Please try rephrasing your question."],
                "entities": [],
                "metadata": {"error": str(e), "fallback_used": True}
            }
    
    def _query_lightrag(self, query: str, mode: str) -> Dict[str, Any]:
        """Query using LightRAG with different modes."""
        try:
            from lightrag import QueryParam
            
            # Convert mode to LightRAG query param
            if mode == "local":
                query_param = QueryParam(mode="local")
            elif mode == "global":
                query_param = QueryParam(mode="global")
            else:
                query_param = QueryParam(mode="hybrid")
            
            # Execute query
            response = self.lightrag.query(query, param=query_param)
            
            return {
                "answer": response,
                "context": [response],
                "entities": [],
                "metadata": {"source": "lightrag", "mode": mode}
            }
            
        except Exception as e:
            logger.error(f"LightRAG query error: {e}")
            raise
    
    def _query_traditional(self, query: str, document_id: Optional[str], 
                          project_id: Optional[str]) -> Dict[str, Any]:
        """Traditional vector similarity search fallback."""
        try:
            # Generate query embedding
            query_embeddings = self.embedding_generator([query], "query")
            if not query_embeddings:
                raise Exception("Failed to generate query embedding")
            
            query_embedding = query_embeddings[0]
            
            # Fetch similar embeddings with project filter
            similar_embeddings = self.db.fetch_similar_embeddings(
                query_embedding=query_embedding,
                limit=5,
                match_threshold=0.7,
                project_id=project_id
            )
            
            if not similar_embeddings:
                return {
                    "context": ["No relevant documents found for your query. Please try rephrasing your question or check if documents have been uploaded to the selected project."],
                    "entities": [],
                    "metadata": {"source": "traditional", "matches": 0}
                }
            
            # Get chunks for the similar embeddings
            chunk_ids = [emb.get("chunk_id") for emb in similar_embeddings if emb.get("chunk_id")]
            chunks = self.db.fetch_chunks_by_ids(chunk_ids)
            
            # Prepare context
            context_chunks = []
            for chunk in chunks:
                context_chunks.append({
                    "text": chunk.get("chunk_text", ""),
                    "document_id": chunk.get("document_id"),
                    "page_number": chunk.get("page_number", 1),
                    "chunk_id": chunk.get("id")
                })
            
            # Get related entities from agricultural entities table
            entities = []
            try:
                if document_id:
                    entities = self.db.query_agricultural_entities(document_id=document_id, limit=10)
                elif project_id:
                    entities = self.db.query_agricultural_entities(project_id=project_id, limit=10)
            except Exception as e:
                logger.warning(f"Could not fetch agricultural entities: {e}")
            
            return {
                "context": context_chunks,
                "entities": entities,
                "metadata": {
                    "source": "traditional", 
                    "matches": len(similar_embeddings),
                    "chunks": len(context_chunks),
                    "entities": len(entities)
                }
            }
            
        except Exception as e:
            logger.error(f"Traditional query error: {e}")
            raise
    
    def insert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Insert documents into the retrieval system.
        
        Args:
            documents: List of document dictionaries with content and metadata
            
        Returns:
            Boolean indicating success
        """
        try:
            if self.lightrag_available and self.lightrag:
                return self._insert_lightrag(documents)
            else:
                logger.info("LightRAG not available, documents stored in traditional database only")
                return True
                
        except Exception as e:
            logger.error(f"Document insertion failed: {e}")
            return False
    
    def _insert_lightrag(self, documents: List[Dict[str, Any]]) -> bool:
        """Insert documents into LightRAG."""
        try:
            for doc in documents:
                content = doc.get("content", "")
                if content.strip():
                    # Insert document into LightRAG
                    self.lightrag.insert(content)
                    logger.info(f"Inserted document into LightRAG: {doc.get('metadata', {}).get('filename', 'Unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"LightRAG insertion error: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of the retrieval system."""
        total_keywords = sum(len(keywords) for domain in self.domain_keywords.values() for keywords in domain.values())
        return {
            "lightrag_available": self.lightrag_available,
            "working_dir": self.working_dir,
            "domain_keywords_loaded": total_keywords,
            "supported_domains": list(self.domain_keywords.keys()),
            "database_connected": self.db is not None,
            "embedding_generator_available": self.embedding_generator is not None
        }
