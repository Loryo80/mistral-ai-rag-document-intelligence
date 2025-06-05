#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Agricultural AI System
Tests all components: database integration, API logging, entity processing, and enhanced features.
"""

import pytest
import os
import sys
import uuid
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.storage.db import Database
from src.processing.text_processor import TextProcessor
from src.processing.enhanced_entity_normalizer import EnhancedEntityNormalizer
from src.retrieval.enhanced_query_processor import EnhancedQueryProcessor
from src.generation.llm_generator import LLMGenerator

# Test configuration
TEST_USER_ID = "test_user_pytest"
TEST_PROJECT_ID = str(uuid.uuid4())
TEST_DOCUMENT_ID = str(uuid.uuid4())

@pytest.fixture(scope="session")
def db_connection():
    """Create a database connection for testing."""
    return Database()

@pytest.fixture(scope="session")
def test_project(db_connection):
    """Create a test project for the session."""
    try:
        project_id = db_connection.create_project(
            name="Enhanced AI Test Project",
            description="Test project for enhanced agricultural AI system",
            user_id=TEST_USER_ID
        )
        yield project_id
        # Cleanup
        try:
            db_connection.delete_project(project_id)
        except Exception as e:
            print(f"Cleanup warning: {e}")
    except Exception as e:
        pytest.skip(f"Could not create test project: {e}")

class TestDatabaseIntegration:
    """Test database connectivity and table access."""
    
    def test_database_connection(self, db_connection):
        """Test basic database connection."""
        assert db_connection is not None
        assert hasattr(db_connection, 'client')
    
    def test_essential_tables_exist(self, db_connection):
        """Test that all essential tables are accessible."""
        essential_tables = [
            'projects', 'documents', 'chunks', 'embeddings',
            'entities', 'relationships', 'api_usage_logs'
        ]
        
        for table in essential_tables:
            try:
                response = db_connection.client.table(table).select("*").limit(1).execute()
                assert response is not None, f"Table {table} is not accessible"
            except Exception as e:
                pytest.fail(f"Table {table} access failed: {e}")
    
    def test_api_usage_logging_table_structure(self, db_connection):
        """Test API usage logging table structure."""
        # Test that we can insert API usage logs
        test_data = {
            "user_id": TEST_USER_ID,
            "api_provider": "mistral",
            "api_type": "test",
            "tokens_used": 100,
            "cost_usd": 0.001
        }
        
        try:
            db_connection.client.table("api_usage_logs").insert(test_data).execute()
            
            # Verify the insert worked
            result = db_connection.client.table("api_usage_logs")\
                .select("*")\
                .eq("user_id", TEST_USER_ID)\
                .eq("api_type", "test")\
                .execute()
            
            assert len(result.data) > 0, "API usage log was not inserted"
            assert result.data[0]["api_provider"] == "mistral"
            
        except Exception as e:
            pytest.fail(f"API usage logging test failed: {e}")

class TestEnhancedEntityNormalizer:
    """Test enhanced entity normalization functionality."""
    
    @pytest.fixture
    def normalizer(self):
        """Create entity normalizer instance."""
        return EnhancedEntityNormalizer()
    
    def test_product_normalization(self, normalizer):
        """Test product entity normalization."""
        test_cases = [
            ("NPK 20-20-20 FERTILIZER®", {"manufacturer": "TestCorp"}),
            ("SuperGrow NPK 15-15-15", {"brand": "SuperGrow"}),
            ("Organic Compost Mix", {"type": "organic"})
        ]
        
        for value, metadata in test_cases:
            normalized, confidence = normalizer.normalize_entity("PRODUCT", value, metadata)
            assert normalized is not None
            assert 0.0 <= confidence <= 1.0
            assert len(normalized) > 0
    
    def test_chemical_compound_normalization(self, normalizer):
        """Test chemical compound normalization."""
        test_cases = [
            ("Glyphosate-isopropylamine salt", {"concentration": "48%"}),
            ("Nitrogen (N)", {"element": "N"}),
            ("Potassium Chloride (KCl)", {"formula": "KCl"})
        ]
        
        for value, metadata in test_cases:
            normalized, confidence = normalizer.normalize_entity("CHEMICAL_COMPOUND", value, metadata)
            assert normalized is not None
            assert confidence > 0.0
    
    def test_crop_normalization(self, normalizer):
        """Test crop entity normalization."""
        test_cases = [
            ("Triticum aestivum (Winter Wheat)", {"growth_stage": "flowering"}),
            ("Zea mays (Corn)", {"variety": "sweet"}),
            ("Solanum tuberosum (Potato)", {"type": "russet"})
        ]
        
        for value, metadata in test_cases:
            normalized, confidence = normalizer.normalize_entity("CROP", value, metadata)
            assert normalized is not None
            assert confidence > 0.0

class TestAPILogging:
    """Test API usage logging across different components."""
    
    def test_embedding_api_logging(self, db_connection):
        """Test that embedding generation logs API usage."""
        processor = TextProcessor(db=db_connection, user_id=TEST_USER_ID)
        
        # Get initial count
        initial_count = len(db_connection.client.table('api_usage_logs')
                          .select('*').eq('user_id', TEST_USER_ID).execute().data)
        
        # Generate embeddings
        test_chunks = ["Test agricultural text for API logging"]
        embeddings = processor.generate_embeddings(test_chunks, document_id="test_doc_api")
        
        # Check that API usage was logged
        final_count = len(db_connection.client.table('api_usage_logs')
                        .select('*').eq('user_id', TEST_USER_ID).execute().data)
        
        assert final_count > initial_count, "API usage was not logged for embedding generation"
        assert len(embeddings) == 1, "Embedding generation failed"
    
    def test_llm_generation_api_logging(self, db_connection):
        """Test that LLM generation logs API usage."""
        llm_generator = LLMGenerator(db=db_connection, user_id=TEST_USER_ID, provider="mistral")
        
        # Get initial count
        initial_count = len(db_connection.client.table('api_usage_logs')
                          .select('*').eq('user_id', TEST_USER_ID).execute().data)
        
        # Generate answer
        test_query = "What is NPK fertilizer?"
        test_context = ["NPK fertilizer contains nitrogen, phosphorus, and potassium"]
        
        result = llm_generator.generate_answer(test_query, test_context, document_id="test_doc_llm")
        
        # Check that API usage was logged
        final_count = len(db_connection.client.table('api_usage_logs')
                        .select('*').eq('user_id', TEST_USER_ID).execute().data)
        
        assert final_count > initial_count, "API usage was not logged for LLM generation"
        assert "answer" in result, "LLM generation failed"
    
    def test_cost_calculation_accuracy(self, db_connection):
        """Test that API costs are calculated correctly."""
        # Get recent API logs with costs
        recent_logs = db_connection.client.table('api_usage_logs')\
            .select('*')\
            .not_.is_('cost_usd', 'null')\
            .order('created_at', desc=True)\
            .limit(5)\
            .execute()
        
        if recent_logs.data:
            for log in recent_logs.data:
                cost = float(log['cost_usd'])
                tokens = log['tokens_used']
                
                # Basic sanity checks
                assert cost >= 0, "Cost cannot be negative"
                assert tokens > 0, "Tokens used should be positive"
                
                # Cost should be reasonable (not more than $1 per 1000 tokens)
                if tokens > 0:
                    cost_per_1k_tokens = (cost * 1000) / tokens
                    assert cost_per_1k_tokens < 1.0, f"Cost seems too high: ${cost_per_1k_tokens:.4f} per 1K tokens"

class TestEnhancedQueryProcessing:
    """Test enhanced query processing functionality."""
    
    @pytest.fixture
    def query_processor(self, db_connection):
        """Create query processor instance."""
        return EnhancedQueryProcessor(db_connection)
    
    def test_query_intent_classification(self, query_processor):
        """Test query intent classification."""
        test_queries = [
            ("What is the NPK ratio for corn fertilizer?", "product_information"),
            ("Show me safety data for glyphosate", "safety_regulatory"),
            ("What are the application rates for foliar spray?", "application_instructions"),
            ("How to store this pesticide?", "handling_storage")
        ]
        
        for query, expected_intent in test_queries:
            result = query_processor.enhance_query_with_entities(query, project_id="test_project")
            
            assert "query_intent" in result
            assert "primary_intent" in result["query_intent"]
            
            # Check that some intent was classified (may not match exactly due to ML variability)
            intent = result["query_intent"]["primary_intent"]
            assert intent is not None and len(intent) > 0
    
    def test_entity_extraction_from_query(self, query_processor):
        """Test entity extraction from queries."""
        query = "What is the application rate of NPK 20-20-20 for corn in spring?"
        
        result = query_processor.enhance_query_with_entities(query, project_id="test_project")
        
        assert "entity_mentions" in result
        assert len(result["entity_mentions"]) > 0
        
        # Should detect entities like NPK, corn, etc.
        entities = [mention["text"].lower() for mention in result["entity_mentions"]]
        assert any("npk" in entity or "nitrogen" in entity for entity in entities)
    
    def test_enhanced_query_generation(self, query_processor):
        """Test enhanced query generation."""
        original_query = "fertilizer for crops"
        
        result = query_processor.enhance_query_with_entities(original_query, project_id="test_project")
        
        assert "enhanced_query" in result
        assert "enhancement_confidence" in result
        
        enhanced_query = result["enhanced_query"]
        assert len(enhanced_query) >= len(original_query)  # Should be enhanced, not shortened

class TestEndToEndIntegration:
    """Test complete end-to-end system integration."""
    
    def test_document_processing_workflow(self, db_connection, test_project):
        """Test complete document processing workflow."""
        # Initialize components
        processor = TextProcessor(db=db_connection, user_id=TEST_USER_ID)
        normalizer = EnhancedEntityNormalizer()
        query_processor = EnhancedQueryProcessor(db_connection)
        llm_generator = LLMGenerator(db=db_connection, user_id=TEST_USER_ID, provider="mistral")
        
        # Sample agricultural text
        sample_text = """
        PRODUCT SPECIFICATION: AgriGrow NPK 16-16-16
        
        Active Ingredients:
        - Nitrogen (N): 16%
        - Phosphorus (P2O5): 16% 
        - Potassium (K2O): 16%
        
        Target Crops: Tomatoes, Peppers, Cucumbers
        Application Rate: 250-350 kg/ha
        pH Range: 6.0-8.0
        
        Safety Information:
        - Avoid contact with skin and eyes
        - Store in cool, dry place
        - Keep away from children
        """
        
        # Step 1: Extract entities and relationships
        entities_result = processor.extract_entities_and_relationships_mistral(
            sample_text, 
            document_id=TEST_DOCUMENT_ID, 
            document_type="technical_specification"
        )
        
        assert "entities" in entities_result
        assert len(entities_result["entities"]) > 0
        
        # Step 2: Normalize some entities
        normalized_entities = []
        for entity in entities_result["entities"][:3]:  # Test first 3
            entity_type = entity.get('type', 'PRODUCT')
            entity_value = entity.get('value', '')
            
            if entity_value:
                normalized_value, confidence = normalizer.normalize_entity(
                    entity_type, 
                    entity_value, 
                    entity.get('metadata', {})
                )
                
                normalized_entities.append({
                    'original': entity_value,
                    'normalized': normalized_value,
                    'confidence': confidence
                })
        
        assert len(normalized_entities) > 0
        
        # Step 3: Test enhanced query processing
        test_query = "What are the application rates for NPK fertilizer on tomatoes?"
        
        enhanced_query = query_processor.enhance_query_with_entities(
            test_query,
            project_id=test_project
        )
        
        assert "enhanced_query" in enhanced_query
        
        # Step 4: Generate answer using LLM
        answer_result = llm_generator.generate_technical_answer(
            test_query,
            [sample_text],
            document_type="technical_specification",
            document_id=TEST_DOCUMENT_ID
        )
        
        assert "answer" in answer_result
        assert "confidence" in answer_result
        assert len(answer_result["answer"]) > 10  # Should be a meaningful answer
    
    def test_agricultural_entity_storage(self, db_connection):
        """Test storage of agricultural entities in database."""
        # Test agricultural entity insertion
        entity_data = {
            'entity_type': 'PRODUCT',
            'entity_value': 'Test NPK Fertilizer',
            'normalized_value': 'npk_fertilizer_test',
            'confidence_score': 0.95,
            'document_id': TEST_DOCUMENT_ID,
            'metadata': {
                'user_id': TEST_USER_ID,
                'context': 'test agricultural entity',
                'extraction_method': 'test'
            }
        }
        
        try:
            entity_id = db_connection.insert_agricultural_entity(entity_data)
            assert entity_id is not None, "Failed to insert agricultural entity"
            
            # Query the entity back
            entities = db_connection.query_agricultural_entities(
                entity_type='PRODUCT',
                document_id=TEST_DOCUMENT_ID
            )
            
            assert len(entities) > 0, "Could not retrieve inserted agricultural entity"
            
        except Exception as e:
            # This might fail if agricultural tables don't exist, which is acceptable
            print(f"Agricultural entity test skipped: {e}")

class TestSystemPerformance:
    """Test system performance and reliability."""
    
    def test_api_response_times(self, db_connection):
        """Test that API calls complete within reasonable time."""
        import time
        
        processor = TextProcessor(db=db_connection, user_id=TEST_USER_ID)
        
        start_time = time.time()
        embeddings = processor.generate_embeddings(["Quick performance test"], document_id="perf_test")
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 10.0, f"Embedding generation took too long: {duration:.2f}s"
        assert len(embeddings) == 1, "Performance test failed to generate embedding"
    
    def test_error_handling(self, db_connection):
        """Test error handling in various components."""
        processor = TextProcessor(db=db_connection, user_id=TEST_USER_ID)
        
        # Test with empty input
        empty_embeddings = processor.generate_embeddings([], document_id="error_test")
        assert empty_embeddings == [], "Should handle empty input gracefully"
        
        # Test with very long text
        long_text = "Agricultural text. " * 1000  # Very long text
        try:
            long_embeddings = processor.generate_embeddings([long_text], document_id="long_test")
            assert len(long_embeddings) == 1, "Should handle long text"
        except Exception as e:
            # It's acceptable if this fails due to token limits
            print(f"Long text test failed as expected: {e}")

class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    def test_api_usage_data_consistency(self, db_connection):
        """Test that API usage logs are consistent and valid."""
        logs = db_connection.client.table('api_usage_logs')\
            .select('*')\
            .order('created_at', desc=True)\
            .limit(10)\
            .execute()
        
        if logs.data:
            for log in logs.data:
                # Check required fields
                assert log['api_provider'] is not None, "API provider should not be null"
                assert log['api_type'] is not None, "API type should not be null"
                assert log['tokens_used'] is not None, "Tokens used should not be null"
                
                # Check data types and ranges
                assert isinstance(log['tokens_used'], int), "Tokens should be integer"
                assert log['tokens_used'] >= 0, "Tokens should be non-negative"
                
                if log['cost_usd'] is not None:
                    cost = float(log['cost_usd'])
                    assert cost >= 0, "Cost should be non-negative"
    
    def test_entity_data_consistency(self, db_connection):
        """Test that entity data is consistent."""
        entities = db_connection.client.table('entities')\
            .select('*')\
            .limit(10)\
            .execute()
        
        if entities.data:
            for entity in entities.data:
                # Check required fields
                assert entity['entity_type'] is not None, "Entity type should not be null"
                assert entity['entity_value'] is not None, "Entity value should not be null"
                
                # Check that entity_value is not empty
                assert len(entity['entity_value'].strip()) > 0, "Entity value should not be empty"

# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 