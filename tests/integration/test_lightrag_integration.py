"""
Test LightRAG Integration with Enhanced Legal AI System
Tests the integration between LightRAG and existing domain expertise
"""
import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.storage.lightrag_integration import EnhancedLightRAG, EntityNode, RelationshipEdge, SupabaseKVStorage
from src.retrieval.lightrag_query_processor import LightRAGQueryProcessor, LightRAGQueryResult
from src.storage.db import SupabaseManager

class TestSupabaseKVStorage:
    """Test custom Supabase KV storage for LightRAG"""
    
    @pytest.fixture
    def mock_supabase_manager(self):
        """Mock Supabase manager for testing"""
        manager = Mock(spec=SupabaseManager)
        manager.supabase = Mock()
        return manager
    
    @pytest.fixture
    def kv_storage(self, mock_supabase_manager):
        """Create KV storage instance for testing"""
        return SupabaseKVStorage(mock_supabase_manager)
    
    @pytest.mark.asyncio
    async def test_set_and_get_key(self, kv_storage, mock_supabase_manager):
        """Test setting and getting a key-value pair"""
        # Mock successful set operation
        mock_supabase_manager.supabase.table.return_value.upsert.return_value.execute = AsyncMock()
        
        # Test set
        await kv_storage.set("test_key", "test_value")
        
        # Verify set was called
        mock_supabase_manager.supabase.table.assert_called_with("lightrag_storage")
        
        # Mock successful get operation
        mock_result = Mock()
        mock_result.data = [{"value": "test_value"}]
        mock_supabase_manager.supabase.table.return_value.select.return_value.eq.return_value.execute = AsyncMock(return_value=mock_result)
        
        # Test get
        result = await kv_storage.get("test_key")
        assert result == "test_value"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, kv_storage, mock_supabase_manager):
        """Test getting a non-existent key returns None"""
        # Mock empty result
        mock_result = Mock()
        mock_result.data = []
        mock_supabase_manager.supabase.table.return_value.select.return_value.eq.return_value.execute = AsyncMock(return_value=mock_result)
        
        result = await kv_storage.get("nonexistent_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_key(self, kv_storage, mock_supabase_manager):
        """Test deleting a key"""
        # Mock successful delete operation
        mock_supabase_manager.supabase.table.return_value.delete.return_value.eq.return_value.execute = AsyncMock()
        
        await kv_storage.delete("test_key")
        
        # Verify delete was called
        mock_supabase_manager.supabase.table.assert_called_with("lightrag_storage")

class TestEnhancedLightRAG:
    """Test the EnhancedLightRAG integration class"""
    
    @pytest.fixture
    def mock_supabase_manager(self):
        """Mock Supabase manager for testing"""
        manager = Mock(spec=SupabaseManager)
        manager.execute_query = AsyncMock()
        return manager
    
    @pytest.fixture
    def enhanced_lightrag(self, mock_supabase_manager):
        """Create EnhancedLightRAG instance for testing"""
        with patch('src.storage.lightrag_integration.LightRAG'):
            return EnhancedLightRAG(mock_supabase_manager)
    
    def test_entity_node_creation(self):
        """Test EntityNode dataclass creation"""
        entity = EntityNode(
            id="test-id",
            type="crop",
            value="wheat",
            normalized_value="triticum aestivum",
            properties={"yield": "high"},
            confidence=0.95,
            domain="agricultural"
        )
        
        assert entity.id == "test-id"
        assert entity.type == "crop"
        assert entity.domain == "agricultural"
        assert entity.confidence == 0.95
    
    def test_relationship_edge_creation(self):
        """Test RelationshipEdge dataclass creation"""
        relationship = RelationshipEdge(
            id="rel-id",
            source_id="entity-1",
            target_id="entity-2",
            relationship_type="affects",
            strength=0.8,
            confidence=0.9,
            evidence="Research shows...",
            domain="agricultural",
            quantitative_data={"correlation": 0.85}
        )
        
        assert relationship.relationship_type == "affects"
        assert relationship.strength == 0.8
        assert relationship.quantitative_data["correlation"] == 0.85
    
    @pytest.mark.asyncio
    async def test_load_agricultural_entities(self, enhanced_lightrag, mock_supabase_manager):
        """Test loading agricultural entities from database"""
        # Mock database response
        mock_data = [
            {
                'id': 'entity-1',
                'entity_type': 'crop',
                'entity_value': 'wheat',
                'normalized_value': 'triticum aestivum',
                'entity_subtype': 'cereal',
                'confidence_score': 0.95,
                'additional_properties': {'yield': 'high'},
                'source_context': 'agricultural study'
            }
        ]
        mock_supabase_manager.execute_query.return_value = mock_data
        
        entities = await enhanced_lightrag._load_agricultural_entities()
        
        assert len(entities) == 1
        assert entities[0].type == 'crop'
        assert entities[0].value == 'wheat'
        assert entities[0].domain == 'agricultural'
        assert entities[0].confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_load_agricultural_relationships(self, enhanced_lightrag, mock_supabase_manager):
        """Test loading agricultural relationships from database"""
        # Mock database response
        mock_data = [
            {
                'id': 'rel-1',
                'relationship_type': 'affects',
                'source_entity_id': 'entity-1',
                'target_entity_id': 'entity-2',
                'relationship_strength': 0.8,
                'confidence_score': 0.9,
                'supporting_evidence': 'Research shows positive correlation',
                'quantitative_data': {'correlation': 0.85}
            }
        ]
        mock_supabase_manager.execute_query.return_value = mock_data
        
        relationships = await enhanced_lightrag._load_agricultural_relationships()
        
        assert len(relationships) == 1
        assert relationships[0].relationship_type == 'affects'
        assert relationships[0].strength == 0.8
        assert relationships[0].domain == 'agricultural'
        assert relationships[0].quantitative_data['correlation'] == 0.85
    
    def test_create_entity_document(self, enhanced_lightrag):
        """Test creating structured document from entity"""
        node_data = {
            'value': 'wheat',
            'type': 'crop',
            'domain': 'agricultural',
            'confidence': 0.95,
            'normalized_value': 'triticum aestivum',
            'properties': {'yield': 'high', 'season': 'winter'}
        }
        
        doc = enhanced_lightrag._create_entity_document('entity-1', node_data)
        
        assert 'Entity: wheat' in doc
        assert 'Type: crop' in doc
        assert 'Domain: agricultural' in doc
        assert 'Confidence: 0.95' in doc
        assert 'Normalized: triticum aestivum' in doc
        assert 'Yield: high' in doc
    
    def test_create_relationship_document(self, enhanced_lightrag):
        """Test creating structured document from relationship"""
        # Setup mock knowledge graph
        enhanced_lightrag.knowledge_graph.add_node('entity-1', value='wheat', type='crop')
        enhanced_lightrag.knowledge_graph.add_node('entity-2', value='nitrogen', type='nutrient')
        
        edge_data = {
            'relationship_type': 'requires',
            'strength': 0.9,
            'confidence': 0.85,
            'domain': 'agricultural',
            'evidence': 'Wheat requires nitrogen for growth',
            'quantitative_data': {'optimal_amount': '120kg/ha'}
        }
        
        doc = enhanced_lightrag._create_relationship_document('entity-1', 'entity-2', edge_data)
        
        assert 'Relationship: wheat requires nitrogen' in doc
        assert 'Strength: 0.9' in doc
        assert 'Confidence: 0.85' in doc
        assert 'Domain: agricultural' in doc
        assert 'Evidence: Wheat requires nitrogen for growth' in doc
    
    def test_get_graph_stats(self, enhanced_lightrag):
        """Test getting knowledge graph statistics"""
        # Add some nodes and edges
        enhanced_lightrag.knowledge_graph.add_node('1', type='crop', domain='agricultural')
        enhanced_lightrag.knowledge_graph.add_node('2', type='nutrient', domain='agricultural')
        enhanced_lightrag.knowledge_graph.add_edge('1', '2', relationship_type='requires')
        
        stats = enhanced_lightrag.get_graph_stats()
        
        assert stats['total_nodes'] == 2
        assert stats['total_edges'] == 1
        assert 'crop' in stats['node_types']
        assert 'requires' in stats['edge_types']
        assert 'agricultural' in stats['domains']

class TestLightRAGQueryProcessor:
    """Test the LightRAG query processor"""
    
    @pytest.fixture
    def mock_supabase_manager(self):
        """Mock Supabase manager for testing"""
        manager = Mock(spec=SupabaseManager)
        manager.execute_query = AsyncMock()
        manager.supabase = Mock()
        return manager
    
    @pytest.fixture
    def query_processor(self, mock_supabase_manager):
        """Create query processor for testing"""
        with patch('src.retrieval.lightrag_query_processor.SimpleQueryProcessor'), \
             patch('src.retrieval.lightrag_query_processor.EnhancedQueryProcessor'), \
             patch('src.retrieval.lightrag_query_processor.EnhancedLightRAG'):
            processor = LightRAGQueryProcessor(mock_supabase_manager)
            processor._lightrag_initialized = True  # Skip initialization for tests
            return processor
    
    def test_determine_optimal_mode_simple(self, query_processor):
        """Test query mode determination for simple queries"""
        query = "What is nitrogen?"
        mode = query_processor._determine_optimal_mode(query)
        assert mode == "simple"
    
    def test_determine_optimal_mode_enhanced(self, query_processor):
        """Test query mode determination for enhanced queries"""
        query = "Analyze the relationship between soil pH and nutrient availability"
        mode = query_processor._determine_optimal_mode(query)
        assert mode == "enhanced"
    
    def test_determine_optimal_mode_hybrid(self, query_processor):
        """Test query mode determination for complex queries"""
        query = "Provide comprehensive analysis of multi-step nitrogen cycling processes and their impact on sustainable agriculture practices"
        mode = query_processor._determine_optimal_mode(query)
        assert mode == "hybrid"
    
    def test_classify_query_domain(self, query_processor):
        """Test domain classification of queries"""
        # Agricultural query
        ag_query = "What are the best fertilizers for corn crops?"
        ag_classification = query_processor._classify_query_domain(ag_query)
        assert ag_classification['agricultural'] > ag_classification['food_industry']
        assert ag_classification['agricultural'] > ag_classification['legal']
        
        # Food industry query
        food_query = "What are the nutritional requirements for food processing?"
        food_classification = query_processor._classify_query_domain(food_query)
        assert food_classification['food_industry'] > food_classification['agricultural']
        
        # Legal query
        legal_query = "What are the regulatory compliance requirements for pesticides?"
        legal_classification = query_processor._classify_query_domain(legal_query)
        assert legal_classification['legal'] > 0
    
    @pytest.mark.asyncio
    async def test_extract_agricultural_patterns(self, query_processor, mock_supabase_manager):
        """Test extracting agricultural patterns from database"""
        # Mock database response
        mock_data = [
            {
                'entity_type': 'crop',
                'entity_value': 'wheat',
                'normalized_value': 'triticum aestivum',
                'relationship_type': 'requires',
                'relationship_strength': 0.8,
                'supporting_evidence': 'Studies show wheat requires nitrogen'
            }
        ]
        mock_supabase_manager.execute_query.return_value = mock_data
        
        patterns = await query_processor._extract_agricultural_patterns("wheat nitrogen", "project-1")
        
        assert patterns['matching_entities'] == 1
        assert 'crop' in patterns['entity_types']
        assert 'requires' in patterns['relationship_types']
        assert len(patterns['top_matches']) <= 5
    
    @pytest.mark.asyncio
    async def test_extract_food_industry_patterns(self, query_processor, mock_supabase_manager):
        """Test extracting food industry patterns from database"""
        # Mock database response
        mock_data = [
            {
                'food_industry_type': 'ingredient',
                'product_category': 'protein',
                'regulatory_status': 'approved',
                'applications': ['bakery', 'dairy'],
                'quality_parameters': {'protein_content': '20%'}
            }
        ]
        mock_supabase_manager.execute_query.return_value = mock_data
        
        patterns = await query_processor._extract_food_industry_patterns("protein ingredient", "project-1")
        
        assert patterns['matching_entities'] == 1
        assert 'ingredient' in patterns['industry_types']
        assert 'protein' in patterns['product_categories']
        assert 'approved' in patterns['regulatory_statuses']
    
    @pytest.mark.asyncio
    async def test_synthesize_hybrid_results(self, query_processor):
        """Test synthesis of hybrid results"""
        query = "What affects crop yield?"
        
        lightrag_result = {
            'combined': {
                'synthesis': 'Multiple factors affect crop yield including soil quality, weather, and nutrients.'
            }
        }
        
        domain_insights = {
            'agricultural_patterns': {
                'matching_entities': 5,
                'entity_types': ['crop', 'soil', 'nutrient']
            },
            'food_industry_patterns': {
                'matching_entities': 2,
                'product_categories': ['grain', 'cereal']
            },
            'domain_classification': {
                'agricultural': 0.7,
                'food_industry': 0.2,
                'legal': 0.1
            }
        }
        
        synthesis = await query_processor._synthesize_hybrid_results(
            query, lightrag_result, domain_insights
        )
        
        assert 'Multiple factors affect crop yield' in synthesis
        assert 'Agricultural Context: Found 5 related agricultural entities' in synthesis
        assert 'Food Industry Context: Identified 2 relevant food industry entities' in synthesis
        assert 'Primary Domain: Agricultural' in synthesis
    
    def test_lightrag_query_result_creation(self):
        """Test LightRAGQueryResult dataclass creation"""
        result = LightRAGQueryResult(
            query="Test query",
            mode="hybrid",
            lightrag_response="Test response",
            graph_entities=[{"id": "1", "type": "crop"}],
            graph_relationships=[{"type": "affects"}],
            confidence_score=0.8,
            processing_time=2.5,
            fallback_used=False,
            domain_specific_insights={"domain": "agricultural"}
        )
        
        assert result.query == "Test query"
        assert result.mode == "hybrid"
        assert result.confidence_score == 0.8
        assert result.processing_time == 2.5
        assert not result.fallback_used
        assert len(result.graph_entities) == 1
        assert len(result.graph_relationships) == 1

class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.fixture
    def mock_components(self):
        """Mock all system components for integration testing"""
        supabase_manager = Mock(spec=SupabaseManager)
        supabase_manager.execute_query = AsyncMock()
        supabase_manager.supabase = Mock()
        
        return {
            'supabase_manager': supabase_manager
        }
    
    @pytest.mark.asyncio
    async def test_agricultural_query_flow(self, mock_components):
        """Test complete flow for agricultural query"""
        with patch('src.retrieval.lightrag_query_processor.SimpleQueryProcessor'), \
             patch('src.retrieval.lightrag_query_processor.EnhancedQueryProcessor'), \
             patch('src.retrieval.lightrag_query_processor.EnhancedLightRAG') as mock_lightrag:
            
            # Setup mocks
            processor = LightRAGQueryProcessor(mock_components['supabase_manager'])
            processor._lightrag_initialized = True
            
            # Mock LightRAG response
            mock_lightrag_instance = mock_lightrag.return_value
            mock_lightrag_instance.enhanced_query = AsyncMock(return_value={
                'combined': {
                    'synthesis': 'Nitrogen affects crop yield through soil availability.',
                    'confidence_score': 0.9
                },
                'graph': {
                    'entities': [{'id': '1', 'type': 'nutrient', 'value': 'nitrogen'}],
                    'relationships': [{'type': 'affects', 'source': 'nitrogen', 'target': 'yield'}]
                }
            })
            processor.lightrag = mock_lightrag_instance
            
            # Mock domain insights
            mock_components['supabase_manager'].execute_query.return_value = [
                {
                    'entity_type': 'nutrient',
                    'entity_value': 'nitrogen',
                    'normalized_value': 'N',
                    'relationship_type': 'affects',
                    'relationship_strength': 0.9,
                    'supporting_evidence': 'Essential for protein synthesis'
                }
            ]
            
            # Test query processing
            result = await processor._process_hybrid(
                "How does nitrogen affect crop yield?",
                "project-1",
                include_domain_insights=True
            )
            
            assert 'lightrag_response' in result
            assert result['confidence_score'] == 0.9
            assert len(result['graph_entities']) > 0
            assert result['processing_method'] == 'hybrid'
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, mock_components):
        """Test fallback mechanism when LightRAG fails"""
        with patch('src.retrieval.lightrag_query_processor.SimpleQueryProcessor') as mock_simple, \
             patch('src.retrieval.lightrag_query_processor.EnhancedQueryProcessor') as mock_enhanced, \
             patch('src.retrieval.lightrag_query_processor.EnhancedLightRAG') as mock_lightrag:
            
            # Setup processor
            processor = LightRAGQueryProcessor(mock_components['supabase_manager'])
            processor._lightrag_initialized = True
            
            # Mock LightRAG to fail
            mock_lightrag_instance = mock_lightrag.return_value
            mock_lightrag_instance.enhanced_query = AsyncMock(side_effect=Exception("LightRAG failed"))
            processor.lightrag = mock_lightrag_instance
            
            # Mock enhanced processor success
            mock_enhanced_instance = mock_enhanced.return_value
            mock_enhanced_instance.process_query = AsyncMock(return_value={
                'response': 'Fallback response from enhanced processor',
                'entities': [],
                'relationships': [],
                'confidence': 0.6
            })
            processor.enhanced_processor = mock_enhanced_instance
            
            # Test fallback
            result = await processor._fallback_processing("test query", "project-1")
            
            assert '[Fallback Mode]' in result['lightrag_response']
            assert result['confidence_score'] == 0.4  # Lower confidence for fallback
            assert result['processing_method'] == 'fallback_enhanced'
    
    def test_performance_expectations(self):
        """Test that performance expectations are realistic"""
        # These are the performance targets from the integration guide
        targets = {
            'simple': 0.5,      # seconds
            'enhanced': 2.0,    # seconds
            'lightrag': 5.0,    # seconds
            'hybrid': 7.0       # seconds
        }
        
        # Verify targets are realistic (not too aggressive)
        assert targets['simple'] < targets['enhanced']
        assert targets['enhanced'] < targets['lightrag']
        assert targets['lightrag'] < targets['hybrid']
        assert targets['hybrid'] < 10.0  # Maximum acceptable time
    
    def test_confidence_score_ranges(self):
        """Test that confidence scores are in valid ranges"""
        # Test confidence calculation logic
        test_cases = [
            ({'entities': [], 'relationships': []}, 0.3),  # Low confidence
            ({'entities': [1], 'relationships': []}, 0.6),  # Medium confidence  
            ({'entities': [1], 'relationships': [1]}, 0.9)  # High confidence
        ]
        
        # Mock processor for testing
        processor = LightRAGQueryProcessor(Mock())
        
        for graph_result, expected_confidence in test_cases:
            confidence = processor._calculate_combined_confidence(graph_result)
            assert 0.0 <= confidence <= 1.0
            assert confidence == expected_confidence

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"]) 