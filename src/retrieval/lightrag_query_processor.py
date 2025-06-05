"""
LightRAG-Powered Query Processor
Combines LightRAG's knowledge graph intelligence with your existing domain expertise
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

# Use absolute imports to avoid relative import issues
try:
    from src.retrieval.simple_query_processor import SimpleQueryProcessor
    from src.retrieval.enhanced_query_processor import EnhancedQueryProcessor
    from src.storage.lightrag_integration import EnhancedLightRAG
    from src.storage.db import Database
except ImportError:
    # Fallback for direct imports
    from .simple_query_processor import SimpleQueryProcessor
    from .enhanced_query_processor import EnhancedQueryProcessor
    from ..storage.lightrag_integration import EnhancedLightRAG
    from ..storage.db import Database

logger = logging.getLogger(__name__)

@dataclass
class LightRAGQueryResult:
    """Result structure for LightRAG queries"""
    query: str
    mode: str
    lightrag_response: str
    graph_entities: List[Dict[str, Any]]
    graph_relationships: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    fallback_used: bool
    domain_specific_insights: Optional[Dict[str, Any]] = None

class LightRAGQueryProcessor:
    """
    Advanced query processor that combines:
    1. LightRAG's global knowledge graph reasoning
    2. Your existing domain-specific entity extraction
    3. Hybrid retrieval with fallback mechanisms
    """
    
    def __init__(self, supabase_manager: Database):
        self.supabase = supabase_manager
        
        # Initialize existing processors for fallback and domain expertise
        self.simple_processor = SimpleQueryProcessor(supabase_manager)
        self.enhanced_processor = EnhancedQueryProcessor(supabase_manager)
        
        # Initialize LightRAG integration
        self.lightrag = EnhancedLightRAG(supabase_manager)
        self._lightrag_initialized = False
        
        # Query routing configuration
        self.query_routing = {
            'simple_keywords': ['what', 'define', 'explain', 'basic'],
            'enhanced_keywords': ['analyze', 'compare', 'relationship', 'impact', 'optimization'],
            'lightrag_keywords': ['complex', 'multi-step', 'reasoning', 'comprehensive', 'synthesis']
        }
        
    async def initialize(self):
        """Initialize LightRAG with existing knowledge base"""
        if not self._lightrag_initialized:
            logger.info("Initializing LightRAG query processor...")
            await self.lightrag.initialize_from_existing_data()
            self._lightrag_initialized = True
            logger.info("LightRAG query processor ready")
            
    async def process_query(
        self, 
        query: str, 
        project_id: str,
        mode: str = "auto",
        fallback_enabled: bool = True,
        include_domain_insights: bool = True
    ) -> LightRAGQueryResult:
        """
        Process query using optimal combination of LightRAG and domain expertise
        
        Args:
            query: User query
            project_id: Project context
            mode: "auto", "lightrag", "enhanced", "simple", or "hybrid"
            fallback_enabled: Enable fallback to simpler processors
            include_domain_insights: Include domain-specific analysis
        """
        start_time = datetime.now()
        fallback_used = False
        
        try:
            # Ensure LightRAG is initialized
            await self.initialize()
            
            # Determine optimal processing mode
            if mode == "auto":
                mode = self._determine_optimal_mode(query)
                
            logger.info(f"Processing query with mode: {mode}")
            
            # Route to appropriate processor
            if mode == "lightrag":
                result = await self._process_with_lightrag(query, project_id)
            elif mode == "hybrid":
                result = await self._process_hybrid(query, project_id, include_domain_insights)
            elif mode == "enhanced":
                result = await self._process_with_enhanced(query, project_id)
            elif mode == "simple":
                result = await self._process_with_simple(query, project_id)
            else:
                # Default to hybrid
                result = await self._process_hybrid(query, project_id, include_domain_insights)
                
        except Exception as e:
            logger.error(f"Error in primary processing: {e}")
            if fallback_enabled:
                logger.info("Falling back to enhanced processor")
                result = await self._fallback_processing(query, project_id)
                fallback_used = True
            else:
                raise
                
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create result object
        lightrag_result = LightRAGQueryResult(
            query=query,
            mode=mode,
            lightrag_response=result.get('lightrag_response', ''),
            graph_entities=result.get('graph_entities', []),
            graph_relationships=result.get('graph_relationships', []),
            confidence_score=result.get('confidence_score', 0.5),
            processing_time=processing_time,
            fallback_used=fallback_used,
            domain_specific_insights=result.get('domain_insights')
        )
        
        # Log query for analytics
        await self._log_query(lightrag_result, project_id)
        
        return lightrag_result
        
    def process_query_sync(
        self, 
        query: str, 
        project_ids: List[str],
        mode: str = "auto"
    ) -> str:
        """
        Simplified synchronous query processing method for Streamlit integration
        Falls back to enhanced processor with mode indication
        """
        try:
            logger.info(f"Processing query with LightRAG mode: {mode}")
            
            # Use existing processors as fallback for now
            # Future versions will integrate with actual LightRAG
            project_id = project_ids[0] if project_ids else None
            
            if mode in ["enhanced", "auto"] and hasattr(self, 'enhanced_processor'):
                try:
                    # Use enhanced processor but indicate LightRAG mode
                    result = self.enhanced_processor.process_query(query, project_id)
                    return f"**🧠 LightRAG Mode: {mode}** (Enhanced Fallback)\n\n{result}"
                except Exception as e:
                    logger.warning(f"Enhanced processor failed: {e}, falling back to simple")
            
            # Fallback to simple processor
            try:
                result = self.simple_processor.simple_query(query, project_id)
                answer = result.get('answer', 'No answer available')
                return f"**🧠 LightRAG Mode: {mode}** (Simple Fallback)\n\n{answer}"
            except Exception as e:
                return f"**❌ LightRAG Processing Error:** {str(e)}\n\nPlease try using Simple mode directly."
                
        except Exception as e:
            logger.error(f"LightRAG query processing failed: {e}")
            return f"**❌ LightRAG Error:** {str(e)}\n\nPlease try using Simple or Enhanced mode instead."
        
    def _determine_optimal_mode(self, query: str) -> str:
        """Determine optimal processing mode based on query characteristics"""
        query_lower = query.lower()
        
        # Count keyword matches for each mode
        simple_score = sum(1 for keyword in self.query_routing['simple_keywords'] if keyword in query_lower)
        enhanced_score = sum(1 for keyword in self.query_routing['enhanced_keywords'] if keyword in query_lower)
        lightrag_score = sum(1 for keyword in self.query_routing['lightrag_keywords'] if keyword in query_lower)
        
        # Consider query complexity (length, question words, conjunctions)
        complexity_score = 0
        if len(query.split()) > 10:
            complexity_score += 1
        if any(word in query_lower for word in ['and', 'or', 'but', 'however', 'because']):
            complexity_score += 1
        if query_lower.count('?') > 1:
            complexity_score += 1
            
        # Determine mode based on scores
        if lightrag_score > 0 or complexity_score >= 2:
            return "hybrid"  # Use both LightRAG and domain expertise
        elif enhanced_score > simple_score:
            return "enhanced"
        elif simple_score > 0:
            return "simple"
        else:
            return "hybrid"  # Default to most comprehensive
            
    async def _process_with_lightrag(self, query: str, project_id: str) -> Dict[str, Any]:
        """Process query primarily with LightRAG"""
        lightrag_result = await self.lightrag.enhanced_query(query, mode="lightrag")
        
        return {
            'lightrag_response': lightrag_result.get('lightrag', ''),
            'graph_entities': [],
            'graph_relationships': [],
            'confidence_score': 0.8,
            'processing_method': 'lightrag_only'
        }
        
    async def _process_hybrid(self, query: str, project_id: str, include_domain_insights: bool = True) -> Dict[str, Any]:
        """Process query with hybrid LightRAG + domain expertise approach"""
        # Get LightRAG results
        lightrag_result = await self.lightrag.enhanced_query(query, mode="hybrid")
        
        # Get domain-specific insights if requested
        domain_insights = None
        if include_domain_insights:
            domain_insights = await self._get_domain_insights(query, project_id)
            
        # Combine results intelligently
        combined_response = await self._synthesize_hybrid_results(
            query, lightrag_result, domain_insights
        )
        
        return {
            'lightrag_response': combined_response,
            'graph_entities': lightrag_result.get('graph', {}).get('entities', []),
            'graph_relationships': lightrag_result.get('graph', {}).get('relationships', []),
            'confidence_score': lightrag_result.get('combined', {}).get('confidence_score', 0.7),
            'domain_insights': domain_insights,
            'processing_method': 'hybrid'
        }
        
    async def _process_with_enhanced(self, query: str, project_id: str) -> Dict[str, Any]:
        """Process query with enhanced processor"""
        result = await self.enhanced_processor.process_query(query, project_id)
        
        return {
            'lightrag_response': result.get('response', ''),
            'graph_entities': result.get('entities', []),
            'graph_relationships': result.get('relationships', []),
            'confidence_score': result.get('confidence', 0.6),
            'processing_method': 'enhanced_only'
        }
        
    async def _process_with_simple(self, query: str, project_id: str) -> Dict[str, Any]:
        """Process query with simple processor"""
        result = await self.simple_processor.process_query(query, project_id)
        
        return {
            'lightrag_response': result,
            'graph_entities': [],
            'graph_relationships': [],
            'confidence_score': 0.5,
            'processing_method': 'simple_only'
        }
        
    async def _get_domain_insights(self, query: str, project_id: str) -> Dict[str, Any]:
        """Get domain-specific insights using existing processors"""
        try:
            # Get enhanced analysis
            enhanced_result = await self.enhanced_processor.process_query(query, project_id)
            
            # Extract domain-specific patterns
            agricultural_patterns = await self._extract_agricultural_patterns(query, project_id)
            food_industry_patterns = await self._extract_food_industry_patterns(query, project_id)
            
            return {
                'enhanced_analysis': enhanced_result,
                'agricultural_patterns': agricultural_patterns,
                'food_industry_patterns': food_industry_patterns,
                'domain_classification': self._classify_query_domain(query)
            }
            
        except Exception as e:
            logger.error(f"Error getting domain insights: {e}")
            return {}
            
    async def _extract_agricultural_patterns(self, query: str, project_id: str) -> Dict[str, Any]:
        """Extract agricultural-specific patterns and relationships"""
        agricultural_query = """
        SELECT ae.entity_type, ae.entity_value, ae.normalized_value,
               ar.relationship_type, ar.relationship_strength, ar.supporting_evidence
        FROM agricultural_entities ae
        LEFT JOIN agricultural_relationships ar ON ae.id = ar.source_entity_id
        WHERE ae.entity_value ILIKE %s OR ae.normalized_value ILIKE %s
        LIMIT 10
        """
        
        search_term = f"%{query}%"
        result = await self.supabase.execute_query(agricultural_query, [search_term, search_term])
        
        return {
            'matching_entities': len(result),
            'entity_types': list(set(row['entity_type'] for row in result if row['entity_type'])),
            'relationship_types': list(set(row['relationship_type'] for row in result if row['relationship_type'])),
            'top_matches': result[:5]
        }
        
    async def _extract_food_industry_patterns(self, query: str, project_id: str) -> Dict[str, Any]:
        """Extract food industry-specific patterns and relationships"""
        food_query = """
        SELECT fie.food_industry_type, fie.product_category, fie.regulatory_status,
               fie.applications, fie.quality_parameters
        FROM food_industry_entities fie
        WHERE fie.food_industry_type ILIKE %s OR fie.product_category ILIKE %s
        LIMIT 10
        """
        
        search_term = f"%{query}%"
        result = await self.supabase.execute_query(food_query, [search_term, search_term])
        
        return {
            'matching_entities': len(result),
            'industry_types': list(set(row['food_industry_type'] for row in result if row['food_industry_type'])),
            'product_categories': list(set(row['product_category'] for row in result if row['product_category'])),
            'regulatory_statuses': list(set(row['regulatory_status'] for row in result if row['regulatory_status'])),
            'top_matches': result[:5]
        }
        
    def _classify_query_domain(self, query: str) -> Dict[str, float]:
        """Classify query into domain categories with confidence scores"""
        query_lower = query.lower()
        
        # Domain keyword mappings
        agricultural_keywords = ['crop', 'soil', 'fertilizer', 'pest', 'yield', 'farm', 'agriculture', 'harvest']
        food_keywords = ['food', 'nutrition', 'ingredient', 'processing', 'quality', 'safety', 'allergen']
        legal_keywords = ['regulation', 'compliance', 'law', 'legal', 'contract', 'policy', 'requirement']
        
        # Calculate domain scores
        agricultural_score = sum(1 for keyword in agricultural_keywords if keyword in query_lower) / len(agricultural_keywords)
        food_score = sum(1 for keyword in food_keywords if keyword in query_lower) / len(food_keywords)
        legal_score = sum(1 for keyword in legal_keywords if keyword in query_lower) / len(legal_keywords)
        
        # Normalize scores
        total_score = agricultural_score + food_score + legal_score
        if total_score > 0:
            agricultural_score /= total_score
            food_score /= total_score
            legal_score /= total_score
        else:
            # Equal distribution if no matches
            agricultural_score = food_score = legal_score = 0.33
            
        return {
            'agricultural': agricultural_score,
            'food_industry': food_score,
            'legal': legal_score
        }
        
    async def _synthesize_hybrid_results(
        self, 
        query: str, 
        lightrag_result: Dict[str, Any], 
        domain_insights: Optional[Dict[str, Any]]
    ) -> str:
        """Synthesize LightRAG and domain-specific results into coherent response"""
        
        # Start with LightRAG response
        base_response = lightrag_result.get('combined', {}).get('synthesis', 
                                          lightrag_result.get('lightrag', ''))
        
        if not domain_insights:
            return base_response
            
        # Add domain-specific context
        synthesis_parts = [base_response]
        
        # Add agricultural context if relevant
        ag_patterns = domain_insights.get('agricultural_patterns', {})
        if ag_patterns.get('matching_entities', 0) > 0:
            synthesis_parts.append(
                f"\nAgricultural Context: Found {ag_patterns['matching_entities']} related agricultural entities "
                f"including {', '.join(ag_patterns.get('entity_types', [])[:3])}."
            )
            
        # Add food industry context if relevant
        food_patterns = domain_insights.get('food_industry_patterns', {})
        if food_patterns.get('matching_entities', 0) > 0:
            synthesis_parts.append(
                f"\nFood Industry Context: Identified {food_patterns['matching_entities']} relevant food industry entities "
                f"across categories: {', '.join(food_patterns.get('product_categories', [])[:3])}."
            )
            
        # Add domain classification
        domain_classification = domain_insights.get('domain_classification', {})
        primary_domain = max(domain_classification.items(), key=lambda x: x[1])
        if primary_domain[1] > 0.4:  # Only if confidence > 40%
            synthesis_parts.append(
                f"\nPrimary Domain: {primary_domain[0].replace('_', ' ').title()} "
                f"(confidence: {primary_domain[1]:.1%})"
            )
            
        return "\n".join(synthesis_parts)
        
    async def _fallback_processing(self, query: str, project_id: str) -> Dict[str, Any]:
        """Fallback to enhanced processor if LightRAG fails"""
        try:
            result = await self.enhanced_processor.process_query(query, project_id)
            return {
                'lightrag_response': f"[Fallback Mode] {result.get('response', '')}",
                'graph_entities': result.get('entities', []),
                'graph_relationships': result.get('relationships', []),
                'confidence_score': 0.4,  # Lower confidence for fallback
                'processing_method': 'fallback_enhanced'
            }
        except Exception as e:
            logger.error(f"Enhanced processor fallback failed: {e}")
            # Ultimate fallback to simple processor
            simple_result = await self.simple_processor.process_query(query, project_id)
            return {
                'lightrag_response': f"[Simple Fallback] {simple_result}",
                'graph_entities': [],
                'graph_relationships': [],
                'confidence_score': 0.2,  # Very low confidence for ultimate fallback
                'processing_method': 'fallback_simple'
            }
            
    async def _log_query(self, result: LightRAGQueryResult, project_id: str):
        """Log query for analytics and improvement"""
        try:
            log_entry = {
                'project_id': project_id,
                'query': result.query,
                'mode': result.mode,
                'confidence_score': result.confidence_score,
                'processing_time': result.processing_time,
                'fallback_used': result.fallback_used,
                'entities_found': len(result.graph_entities),
                'relationships_found': len(result.graph_relationships),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await self.supabase.supabase.table('lightrag_query_logs').insert(log_entry).execute()
            
        except Exception as e:
            logger.error(f"Error logging query: {e}")
            
    async def get_query_statistics(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get query processing statistics"""
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if project_id:
                conditions.append("project_id = %s")
                params.append(project_id)
                
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            stats_query = f"""
            SELECT 
                mode,
                COUNT(*) as query_count,
                AVG(confidence_score) as avg_confidence,
                AVG(processing_time) as avg_processing_time,
                SUM(CASE WHEN fallback_used THEN 1 ELSE 0 END) as fallback_count
            FROM lightrag_query_logs 
            {where_clause}
            GROUP BY mode
            """
            
            result = await self.supabase.execute_query(stats_query, params)
            
            return {
                'statistics_by_mode': result,
                'lightrag_graph_stats': self.lightrag.get_graph_stats() if self._lightrag_initialized else {},
                'total_queries': sum(row['query_count'] for row in result),
                'overall_avg_confidence': sum(row['avg_confidence'] * row['query_count'] for row in result) / sum(row['query_count'] for row in result) if result else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting query statistics: {e}")
            return {}
            
    async def add_knowledge(self, text: str, domain: str = "general", project_id: Optional[str] = None) -> Dict[str, Any]:
        """Add new knowledge to the LightRAG system"""
        if not self._lightrag_initialized:
            await self.initialize()
            
        result = await self.lightrag.add_new_knowledge(text, domain)
        
        # Log knowledge addition
        if project_id:
            log_entry = {
                'project_id': project_id,
                'action': 'knowledge_addition',
                'domain': domain,
                'text_length': len(text),
                'entities_extracted': result.get('extracted_entities', 0),
                'relationships_extracted': result.get('extracted_relationships', 0),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            try:
                await self.supabase.supabase.table('lightrag_knowledge_logs').insert(log_entry).execute()
            except Exception as e:
                logger.error(f"Error logging knowledge addition: {e}")
                
        return result 