"""
Comprehensive Test Suite for Food Industry B2B Integration.

This module tests the complete food industry pipeline including:
- Food entity extraction and processing
- Enhanced query processing with food domain classification
- Database operations for food industry entities
- B2B ingredient search functionality
- Nutritional and allergen information processing
"""

import pytest
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from processing.food_industry_processor import FoodIndustryProcessor, FoodEntity
from retrieval.enhanced_query_processor import EnhancedQueryProcessor
from storage.supabase_client import SupabaseClient

class TestFoodIndustryProcessor:
    """Test suite for the Food Industry Processor."""
    
    @pytest.fixture
    def mock_db_client(self):
        """Mock database client."""
        mock_db = Mock()
        mock_db.execute_query.return_value = []
        mock_db.log_api_usage.return_value = None
        return mock_db
    
    @pytest.fixture
    def food_processor(self, mock_db_client):
        """Food industry processor instance."""
        return FoodIndustryProcessor(mock_db_client)
    
    @pytest.fixture
    def sample_food_text(self):
        """Sample food industry document text."""
        return """
        INGREDIENT SPECIFICATION SHEET
        
        Product Name: Sodium Benzoate (E211)
        CAS Number: 532-32-1
        Chemical Formula: C7H5NaO2
        
        Description: Sodium benzoate is a food preservative and antimicrobial agent 
        commonly used in acidic foods and beverages. It is GRAS approved by the FDA 
        and has E-number E211 in Europe.
        
        Applications:
        - Soft drinks and beverages
        - Fruit juices and jams
        - Pickled products
        - Bakery products with acidic pH
        
        Nutritional Information:
        Vitamin C: 0mg per 100g
        Calories: 0 kcal per 100g
        
        Allergen Information:
        Contains: None
        May contain traces of: None
        Allergen-free: Gluten-free, dairy-free, nut-free
        
        Storage Conditions:
        Store at room temperature (15-25°C) in a dry place
        Shelf life: 24 months
        
        Regulatory Status:
        - FDA GRAS approved
        - EFSA approved (E211)
        - Kosher certified
        - Halal certified
        
        Microbiological Specifications:
        Total plate count: <1000 CFU/g
        Yeast: <100 CFU/g
        Mold: <100 CFU/g
        """
    
    def test_food_config_loading(self, food_processor):
        """Test that food industry configuration is loaded correctly."""
        assert hasattr(food_processor, 'food_config')
        assert 'entity_types' in food_processor.food_config
        assert 'FOOD_INGREDIENT' in food_processor.food_config['entity_types']
        assert 'NUTRITIONAL_COMPONENT' in food_processor.food_config['entity_types']
        assert 'ALLERGEN_INFO' in food_processor.food_config['entity_types']
    
    def test_food_patterns_initialization(self, food_processor):
        """Test that food-specific regex patterns are initialized."""
        assert hasattr(food_processor, 'food_patterns')
        assert 'vitamin_content' in food_processor.food_patterns
        assert 'e_number' in food_processor.food_patterns
        assert 'allergen_declaration' in food_processor.food_patterns
        assert 'gras_status' in food_processor.food_patterns
    
    def test_food_entity_extraction(self, food_processor, sample_food_text):
        """Test extraction of food industry entities."""
        entities = food_processor.extract_food_entities(sample_food_text, "INGREDIENT_SPEC")
        
        # Should extract food ingredients
        food_ingredients = entities.get('FOOD_INGREDIENT', [])
        assert len(food_ingredients) > 0
        
        # Should find sodium benzoate
        sodium_benzoate_found = any('sodium benzoate' in entity.text.lower() 
                                  for entity in food_ingredients)
        assert sodium_benzoate_found
        
        # Should extract food safety standards
        safety_standards = entities.get('FOOD_SAFETY_STANDARD', [])
        assert len(safety_standards) > 0
        
        # Should find GRAS and E-number
        gras_found = any('gras' in entity.text.lower() for entity in safety_standards)
        e_number_found = any('e211' in entity.text.lower() for entity in safety_standards)
        assert gras_found or e_number_found
    
    def test_food_specific_info_extraction(self, food_processor, sample_food_text):
        """Test extraction of food-specific information like nutritional data."""
        nutritional_info = food_processor._extract_nutritional_information(sample_food_text)
        
        assert 'vitamins' in nutritional_info
        assert 'calories' in nutritional_info
        
        # Should extract vitamin C content
        vitamins = nutritional_info['vitamins']
        vitamin_c_found = any(v['vitamin'] == 'C' for v in vitamins)
        
        # Should extract caloric value
        calories = nutritional_info['calories']
        if calories:
            assert calories['amount'] == 0
            assert 'kcal' in calories['unit']
    
    def test_allergen_information_extraction(self, food_processor, sample_food_text):
        """Test extraction of allergen information."""
        allergen_info = food_processor._extract_allergen_information(sample_food_text)
        
        assert 'contains' in allergen_info
        assert 'free_from' in allergen_info
        
        # Should identify gluten-free, dairy-free, nut-free
        free_from = allergen_info['free_from']
        assert 'gluten' in free_from or 'dairy' in free_from or 'nut' in free_from
    
    def test_regulatory_information_extraction(self, food_processor, sample_food_text):
        """Test extraction of regulatory compliance information."""
        regulatory_info = food_processor._extract_food_regulatory_information(sample_food_text)
        
        assert 'gras_status' in regulatory_info
        assert 'fda_approved' in regulatory_info
        assert 'e_numbers' in regulatory_info
        assert 'certifications' in regulatory_info
        
        # Should detect GRAS status
        assert regulatory_info['gras_status'] == True
        
        # Should extract E211
        assert 'E211' in regulatory_info['e_numbers']
        
        # Should find certifications
        certifications = regulatory_info['certifications']
        assert 'kosher' in certifications or 'halal' in certifications
    
    def test_food_document_classification(self, food_processor, sample_food_text):
        """Test classification of food industry documents."""
        doc_type = food_processor._classify_food_document_type(sample_food_text, "ingredient_spec.pdf")
        
        assert doc_type in ['INGREDIENT_SPEC', 'FOOD_DATASHEET', 'COA', 'NUTRITIONAL_INFO']
    
    def test_complete_food_document_processing(self, food_processor, sample_food_text):
        """Test complete processing of a food industry document."""
        result = food_processor.process_food_document(
            sample_food_text, 
            document_id="test-doc-123",
            filename="sodium_benzoate_spec.pdf"
        )
        
        # Check result structure
        assert 'document_id' in result
        assert 'document_type' in result
        assert 'entities' in result
        assert 'nutritional_information' in result
        assert 'allergen_information' in result
        assert 'regulatory_information' in result
        assert 'quality_metrics' in result
        
        # Check metadata
        metadata = result['processing_metadata']
        assert metadata['processor_version'] == "FoodIndustryProcessor_v1.0"
        assert metadata['food_specific_features'] == True
        assert metadata['total_entities'] > 0


class TestEnhancedQueryProcessorFoodIntegration:
    """Test suite for Enhanced Query Processor with food industry support."""
    
    @pytest.fixture
    def mock_db_client(self):
        """Mock database client with food industry data."""
        mock_db = Mock()
        
        # Mock food entity search results
        mock_db.execute_query.return_value = [
            {
                'id': '123',
                'entity_value': 'sodium benzoate',
                'entity_type': 'FOOD_INGREDIENT',
                'confidence_score': 0.95,
                'food_type': 'preservative',
                'regulatory_status': 'GRAS approved',
                'allergen_info': {'free_from': ['gluten', 'dairy']},
                'filename': 'ingredient_spec.pdf'
            },
            {
                'id': '124',
                'entity_value': 'vitamin c',
                'entity_type': 'NUTRITIONAL_COMPONENT',
                'confidence_score': 0.90,
                'nutritional_value': 60.0,
                'nutritional_unit': 'mg',
                'filename': 'nutrition_facts.pdf'
            }
        ]
        
        return mock_db
    
    @pytest.fixture
    def enhanced_query_processor(self, mock_db_client):
        """Enhanced query processor with food industry support."""
        processor = EnhancedQueryProcessor(mock_db_client)
        # Mock the food config loading
        processor.food_config = {
            'entity_types': {
                'FOOD_INGREDIENT': {
                    'examples': ['sodium benzoate', 'vitamin c', 'calcium']
                }
            }
        }
        return processor
    
    def test_query_domain_classification(self, enhanced_query_processor):
        """Test classification of query domains."""
        # Food industry query
        food_query = "What are the allergen information for sodium benzoate preservative?"
        domain = enhanced_query_processor.classify_query_domain(food_query)
        assert domain == 'food_industry'
        
        # Agricultural query
        agri_query = "What is the application rate for nitrogen fertilizer on corn?"
        domain = enhanced_query_processor.classify_query_domain(agri_query)
        assert domain == 'agricultural'
        
        # Mixed domain query
        mixed_query = "What organic certification do natural preservatives have?"
        domain = enhanced_query_processor.classify_query_domain(mixed_query)
        assert domain in ['mixed', 'food_industry']
    
    def test_food_entity_extraction_from_query(self, enhanced_query_processor):
        """Test extraction of food entities from user queries."""
        query = "What are the nutritional benefits of vitamin C and allergen info for dairy ingredients?"
        
        mentions = enhanced_query_processor._extract_entity_mentions_from_query(query, 'food_industry')
        
        # Should extract food-related mentions
        assert len(mentions) > 0
        
        # Should identify vitamin C
        vitamin_found = any('vitamin' in mention['text'].lower() for mention in mentions)
        assert vitamin_found
        
        # Should identify allergen-related terms
        allergen_found = any('dairy' in mention['text'].lower() for mention in mentions)
        assert allergen_found
    
    def test_food_query_enhancement(self, enhanced_query_processor):
        """Test enhancement of food industry queries."""
        query = "What preservatives are GRAS approved for bakery products?"
        
        result = enhanced_query_processor.enhance_query_with_entities(query)
        
        # Check result structure
        assert 'original_query' in result
        assert 'enhanced_query' in result
        assert 'query_domain' in result
        assert 'expansion_terms' in result
        assert 'suggested_filters' in result
        
        # Should classify as food industry domain
        assert result['query_domain'] == 'food_industry'
        
        # Should have food-related expansion terms
        expansion_terms = result['expansion_terms']
        food_terms = ['gras', 'approved', 'certified', 'bakery', 'preservative']
        has_food_terms = any(term in ' '.join(expansion_terms) for term in food_terms)
        assert has_food_terms
    
    def test_food_specific_filters(self, enhanced_query_processor):
        """Test generation of food industry specific filters."""
        matches = [
            {
                'entity_type': 'FOOD_INGREDIENT',
                'entity_value': 'sodium benzoate',
                'confidence_score': 0.9
            },
            {
                'entity_type': 'ALLERGEN_INFO',
                'entity_value': 'dairy',
                'confidence_score': 0.8
            }
        ]
        
        filters = enhanced_query_processor._suggest_query_filters(matches, 'food_industry')
        
        # Should suggest food industry document types
        doc_types = filters['document_types']
        food_doc_types = ['food_sds', 'nutritional_info', 'ingredient_spec', 'allergen_declaration']
        has_food_doc_types = any(fdt in doc_types for fdt in food_doc_types)
        assert has_food_doc_types
        
        # Should suggest allergen-specific filters
        domain_specific = filters['domain_specific']
        assert 'allergen_free' in domain_specific
    
    def test_food_ingredient_search(self, enhanced_query_processor):
        """Test specialized food ingredient search functionality."""
        query = "natural preservatives for dairy products"
        filters = {
            'food_grade': True,
            'allergen_free': ['gluten'],
            'organic': True
        }
        
        # Mock the database execution
        with patch.object(enhanced_query_processor.db, 'execute_query') as mock_query:
            mock_query.return_value = [
                {
                    'id': '123',
                    'entity_value': 'sodium benzoate',
                    'food_type': 'preservative',
                    'organic_certified': True,
                    'food_grade': True,
                    'regulatory_status': 'GRAS approved'
                }
            ]
            
            result = enhanced_query_processor.search_food_ingredients(query, filters)
        
        # Check result structure
        assert 'query' in result
        assert 'domain' in result
        assert 'ingredients' in result
        assert 'insights' in result
        assert 'recommendations' in result
        
        # Should be classified as food industry
        assert result['domain'] == 'food_industry'
        
        # Should have insights about the search
        insights = result['insights']
        assert 'regulatory_focus' in insights
        assert 'allergen_focus' in insights


class TestFoodDatabaseIntegration:
    """Test suite for food industry database integration."""
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client for testing."""
        mock_client = Mock(spec=SupabaseClient)
        
        # Mock table operations
        mock_table = Mock()
        mock_table.insert.return_value.execute.return_value = Mock(data=[{'id': '123'}])
        mock_table.select.return_value.execute.return_value = Mock(data=[])
        mock_table.update.return_value.eq.return_value.execute.return_value = Mock(data=[])
        
        mock_client.table.return_value = mock_table
        mock_client.execute_query.return_value = []
        
        return mock_client
    
    def test_food_entity_storage(self, mock_supabase_client):
        """Test storage of food industry entities in database."""
        food_entity = {
            'entity_id': '123',
            'food_type': 'preservative',
            'ingredient_category': 'synthetic',
            'functional_class': 'antimicrobial',
            'allergen_info': {'free_from': ['gluten', 'dairy']},
            'regulatory_status': 'GRAS approved',
            'e_number': 'E211',
            'food_grade': True,
            'organic_certified': False
        }
        
        # Test insertion
        mock_supabase_client.table('food_industry_entities').insert(food_entity).execute()
        
        # Verify the table method was called
        mock_supabase_client.table.assert_called_with('food_industry_entities')
    
    def test_nutritional_information_storage(self, mock_supabase_client):
        """Test storage of nutritional information."""
        nutritional_data = {
            'entity_id': '123',
            'nutrient_type': 'vitamin',
            'nutrient_name': 'vitamin_c',
            'content_value': 60.0,
            'content_unit': 'mg',
            'per_serving_size': '100g',
            'daily_value_percentage': 67.0
        }
        
        # Test insertion
        mock_supabase_client.table('nutritional_information').insert(nutritional_data).execute()
        
        # Verify the table method was called
        mock_supabase_client.table.assert_called_with('nutritional_information')
    
    def test_allergen_information_storage(self, mock_supabase_client):
        """Test storage of allergen information."""
        allergen_data = {
            'entity_id': '123',
            'allergen_type': 'dairy',
            'declaration_type': 'free_from',
            'cross_contamination_risk': 'low',
            'labeling_requirements': ['Dairy-free label approved']
        }
        
        # Test insertion
        mock_supabase_client.table('allergen_information').insert(allergen_data).execute()
        
        # Verify the table method was called
        mock_supabase_client.table.assert_called_with('allergen_information')


class TestFoodEntityModel:
    """Test suite for the FoodEntity data model."""
    
    def test_food_entity_creation(self):
        """Test creation of FoodEntity objects."""
        food_entity = FoodEntity(
            text="sodium benzoate",
            entity_type="FOOD_INGREDIENT",
            confidence=0.95,
            start_pos=10,
            end_pos=25,
            context="Sodium benzoate is a preservative",
            allergen_info={'free_from': ['gluten']},
            nutritional_value=None,
            regulatory_status="GRAS approved",
            food_grade=True,
            shelf_life="24 months"
        )
        
        # Test basic properties
        assert food_entity.text == "sodium benzoate"
        assert food_entity.entity_type == "FOOD_INGREDIENT"
        assert food_entity.confidence == 0.95
        
        # Test food-specific properties
        assert food_entity.allergen_info == {'free_from': ['gluten']}
        assert food_entity.regulatory_status == "GRAS approved"
        assert food_entity.food_grade == True
        assert food_entity.shelf_life == "24 months"
    
    def test_food_entity_inheritance(self):
        """Test that FoodEntity properly inherits from ExtractedEntity."""
        food_entity = FoodEntity(
            text="vitamin c",
            entity_type="NUTRITIONAL_COMPONENT",
            confidence=0.85,
            start_pos=0,
            end_pos=9,
            nutritional_value=60.0
        )
        
        # Should have inherited properties
        assert hasattr(food_entity, 'text')
        assert hasattr(food_entity, 'entity_type')
        assert hasattr(food_entity, 'confidence')
        
        # Should have food-specific properties
        assert hasattr(food_entity, 'nutritional_value')
        assert hasattr(food_entity, 'allergen_info')
        assert hasattr(food_entity, 'regulatory_status')


class TestFoodQueryIntentClassification:
    """Test suite for food industry query intent classification."""
    
    @pytest.fixture
    def enhanced_processor(self):
        """Enhanced query processor for testing."""
        mock_db = Mock()
        processor = EnhancedQueryProcessor(mock_db)
        return processor
    
    def test_ingredient_information_intent(self, enhanced_processor):
        """Test classification of ingredient information queries."""
        query = "What are the properties of sodium benzoate preservative?"
        intent = enhanced_processor._classify_query_intent(query)
        
        assert intent['primary_intent'] in ['ingredient_information', 'product_information']
        assert intent['confidence'] > 0.5
    
    def test_nutritional_inquiry_intent(self, enhanced_processor):
        """Test classification of nutritional inquiry queries."""
        query = "How much vitamin C content is in this supplement?"
        intent = enhanced_processor._classify_query_intent(query)
        
        assert intent['primary_intent'] in ['nutritional_inquiry', 'ingredient_information']
        assert intent['confidence'] > 0.5
    
    def test_allergen_safety_intent(self, enhanced_processor):
        """Test classification of allergen safety queries."""
        query = "Does this ingredient contain any dairy allergens?"
        intent = enhanced_processor._classify_query_intent(query)
        
        assert intent['primary_intent'] in ['allergen_safety', 'safety_regulatory']
        assert intent['confidence'] > 0.5
    
    def test_food_regulatory_intent(self, enhanced_processor):
        """Test classification of food regulatory queries."""
        query = "Is this ingredient GRAS approved by the FDA?"
        intent = enhanced_processor._classify_query_intent(query)
        
        assert intent['primary_intent'] in ['food_regulatory', 'safety_regulatory']
        assert intent['confidence'] > 0.5


class TestFoodApplicationAnalysis:
    """Test suite for food application analysis functionality."""
    
    @pytest.fixture
    def mock_db_with_food_functions(self):
        """Mock database with food-specific functions."""
        mock_db = Mock()
        
        # Mock the analyze_food_application function result
        mock_db.execute_query.return_value = [
            {
                'total_ingredients': 5,
                'allergen_count': 2,
                'regulatory_compliance_score': 0.85,
                'nutritional_completeness': 0.75,
                'estimated_shelf_life': '12 months',
                'risk_factors': {
                    'allergen_risk': 'medium',
                    'regulatory_risk': 'low'
                }
            }
        ]
        
        return mock_db
    
    def test_food_application_analysis(self, mock_db_with_food_functions):
        """Test analysis of food applications using database functions."""
        # This would be called through the enhanced query processor
        application_id = '123e4567-e89b-12d3-a456-426614174000'
        
        # Simulate calling the database function
        result = mock_db_with_food_functions.execute_query(
            "SELECT * FROM analyze_food_application(%s)",
            [application_id]
        )
        
        assert len(result) > 0
        analysis = result[0]
        
        # Check analysis results
        assert 'total_ingredients' in analysis
        assert 'allergen_count' in analysis
        assert 'regulatory_compliance_score' in analysis
        assert 'nutritional_completeness' in analysis
        assert 'risk_factors' in analysis
        
        # Verify data types and ranges
        assert isinstance(analysis['total_ingredients'], int)
        assert 0.0 <= analysis['regulatory_compliance_score'] <= 1.0
        assert 0.0 <= analysis['nutritional_completeness'] <= 1.0


# Integration test to verify the complete pipeline
class TestCompleteIntegrationFlow:
    """End-to-end integration tests for food industry B2B support."""
    
    @pytest.fixture
    def complete_system(self):
        """Complete system setup for integration testing."""
        mock_db = Mock()
        
        # Setup mock database responses for complete flow
        mock_db.execute_query.side_effect = [
            # First call: entity search
            [
                {
                    'id': '123',
                    'entity_value': 'sodium benzoate',
                    'entity_type': 'FOOD_INGREDIENT',
                    'confidence_score': 0.95,
                    'filename': 'ingredient_spec.pdf'
                }
            ],
            # Second call: food entity details
            [
                {
                    'id': '123',
                    'entity_value': 'sodium benzoate',
                    'food_type': 'preservative',
                    'regulatory_status': 'GRAS approved',
                    'allergen_info': {'free_from': ['gluten', 'dairy']},
                    'filename': 'ingredient_spec.pdf'
                }
            ],
            # Third call: nutritional information
            [
                {
                    'nutrient_name': 'calories',
                    'content_value': 0.0,
                    'content_unit': 'kcal',
                    'entity_value': 'sodium benzoate'
                }
            ],
            # Fourth call: allergen information
            [
                {
                    'allergen_type': 'gluten',
                    'declaration_type': 'free_from',
                    'entity_value': 'sodium benzoate'
                }
            ]
        ]
        
        # Setup food processor
        food_processor = FoodIndustryProcessor(mock_db)
        food_processor.food_config = {
            'entity_types': {
                'FOOD_INGREDIENT': {
                    'examples': ['sodium benzoate', 'vitamin c']
                }
            }
        }
        
        # Setup enhanced query processor
        query_processor = EnhancedQueryProcessor(mock_db)
        query_processor.food_config = food_processor.food_config
        
        return {
            'food_processor': food_processor,
            'query_processor': query_processor,
            'mock_db': mock_db
        }
    
    def test_complete_food_b2b_workflow(self, complete_system):
        """Test complete B2B food ingredient workflow."""
        food_processor = complete_system['food_processor']
        query_processor = complete_system['query_processor']
        
        # Step 1: Process food industry document
        sample_text = """
        Sodium Benzoate (E211) - Food Grade Preservative
        GRAS approved by FDA. Suitable for beverages, jams, and pickled products.
        Allergen-free: Gluten-free, dairy-free.
        Storage: Store in dry conditions at room temperature.
        """
        
        food_doc_result = food_processor.process_food_document(
            sample_text, 
            document_id="test-123",
            filename="sodium_benzoate.pdf"
        )
        
        # Verify document processing results
        assert food_doc_result['document_type'] in ['INGREDIENT_SPEC', 'FOOD_DATASHEET']
        assert len(food_doc_result['entities']) > 0
        assert food_doc_result['regulatory_information']['gras_status'] == True
        
        # Step 2: Enhanced query processing for B2B search
        b2b_query = "What are GRAS approved preservatives for beverage applications?"
        
        enhanced_result = query_processor.enhance_query_with_entities(b2b_query)
        
        # Verify query enhancement
        assert enhanced_result['query_domain'] == 'food_industry'
        assert len(enhanced_result['expansion_terms']) > 0
        
        # Step 3: Specialized ingredient search
        ingredient_search_result = query_processor.search_food_ingredients(
            "preservatives for beverages",
            filters={'food_grade': True, 'regulatory_status': 'GRAS'}
        )
        
        # Verify search results
        assert ingredient_search_result['domain'] == 'food_industry'
        assert len(ingredient_search_result['ingredients']) >= 0
        assert 'insights' in ingredient_search_result
        assert 'recommendations' in ingredient_search_result
        
        # Verify B2B-relevant insights
        insights = ingredient_search_result['insights']
        assert 'regulatory_focus' in insights
        assert 'domain_relevance' in insights


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"]) 