#!/usr/bin/env python3
"""
Test Suite for PDF Extractor API Usage Logging
Tests the API logging functionality that was recently fixed.
"""

import pytest
import os
import sys
import uuid
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.storage.db import Database
from src.extraction.pdf_extractor import PDFExtractor

# Test configuration
TEST_USER_ID = "test_user_pdf_extractor"
TEST_DOCUMENT_ID = str(uuid.uuid4())

@pytest.fixture
def db_connection():
    """Create a database connection for testing."""
    return Database()

@pytest.fixture  
def pdf_extractor(db_connection):
    """Create PDF extractor with database logging."""
    return PDFExtractor(db=db_connection, user_id=TEST_USER_ID)

class TestPDFExtractorAPILogging:
    """Test PDF extractor API usage logging functionality."""
    
    def test_pdf_extractor_initialization_with_db(self, pdf_extractor):
        """Test that PDF extractor initializes correctly with database."""
        assert pdf_extractor.db is not None
        assert pdf_extractor.user_id == TEST_USER_ID
        assert hasattr(pdf_extractor, '_log_api_usage')
    
    @patch('src.extraction.pdf_extractor.convert_from_path')
    @patch.object(PDFExtractor, '_detect_document_language')
    def test_mistral_ocr_api_logging(self, mock_detect_lang, mock_convert, pdf_extractor, db_connection):
        """Test that Mistral OCR API usage is logged correctly."""
        # Skip if no Mistral API key
        if not pdf_extractor.api_key:
            pytest.skip("No Mistral API key available for testing")
        
        # Mock dependencies
        mock_detect_lang.return_value = "en"
        mock_image = Mock()
        mock_convert.return_value = [mock_image]
        
        # Mock the image encoding
        with patch.object(pdf_extractor, '_encode_image_to_base64', return_value="mock_base64"):
            # Mock the Mistral API response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "Test OCR extracted content"
            mock_response.usage = Mock()
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            mock_response.usage.total_tokens = 150
            
            with patch.object(pdf_extractor.client.chat, 'complete', return_value=mock_response):
                # Get initial API log count
                initial_count = len(db_connection.client.table('api_usage_logs')
                                 .select('*').eq('user_id', TEST_USER_ID).execute().data)
                
                # Create a temporary test PDF file
                test_pdf_path = "test_temp.pdf"
                with open(test_pdf_path, "wb") as f:
                    f.write(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n")
                
                try:
                    # Call the extraction method
                    result = pdf_extractor._extract_with_mistral(test_pdf_path, TEST_DOCUMENT_ID)
                    
                    # Verify extraction worked
                    assert "text" in result
                    assert "Test OCR extracted content" in result["text"]
                    
                    # Check that API usage was logged
                    final_count = len(db_connection.client.table('api_usage_logs')
                                    .select('*').eq('user_id', TEST_USER_ID).execute().data)
                    
                    assert final_count > initial_count, "API usage was not logged for Mistral OCR"
                    
                    # Verify the logged data
                    latest_logs = db_connection.client.table('api_usage_logs')\
                        .select('*')\
                        .eq('user_id', TEST_USER_ID)\
                        .order('created_at', desc=True)\
                        .limit(1)\
                        .execute()
                    
                    if latest_logs.data:
                        log = latest_logs.data[0]
                        assert log['api_provider'] == 'mistral'
                        assert log['api_type'] == 'ocr'
                        assert log['tokens_used'] == 150
                        assert float(log['cost_usd']) >= 0
                        
                        # Check request payload contains model info
                        if log['request_payload']:
                            assert 'model' in log['request_payload']
                            assert 'page' in log['request_payload']
                
                finally:
                    # Clean up test file
                    if os.path.exists(test_pdf_path):
                        os.remove(test_pdf_path)
    
    def test_cost_calculation_mistral_ocr(self, pdf_extractor):
        """Test Mistral OCR cost calculation."""
        # Test cost calculation method
        cost = pdf_extractor._calculate_mistral_ocr_cost("pixtral-12b", 1000, 800, 200)
        
        # Should be reasonable cost
        assert cost >= 0
        assert cost < 1.0  # Should be less than $1 for 1000 tokens
        
        # Test with unknown model (should use fallback)
        fallback_cost = pdf_extractor._calculate_mistral_ocr_cost("unknown-model", 1000, 800, 200)
        assert fallback_cost >= 0
    
    def test_log_api_usage_method(self, pdf_extractor, db_connection):
        """Test the _log_api_usage method directly."""
        # Create a mock response
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response content"
        
        # Get initial count
        initial_count = len(db_connection.client.table('api_usage_logs')
                          .select('*').eq('user_id', TEST_USER_ID).execute().data)
        
        # Call the logging method
        pdf_extractor._log_api_usage(
            api_provider="mistral",
            api_type="ocr",
            tokens_used=150,
            response=mock_response,
            document_id=TEST_DOCUMENT_ID,
            page_number=1
        )
        
        # Check that log was created
        final_count = len(db_connection.client.table('api_usage_logs')
                        .select('*').eq('user_id', TEST_USER_ID).execute().data)
        
        assert final_count > initial_count, "API usage log was not created"
        
        # Verify the log details
        latest_log = db_connection.client.table('api_usage_logs')\
            .select('*')\
            .eq('user_id', TEST_USER_ID)\
            .order('created_at', desc=True)\
            .limit(1)\
            .execute()
        
        if latest_log.data:
            log = latest_log.data[0]
            assert log['api_provider'] == 'mistral'
            assert log['api_type'] == 'ocr'
            assert log['tokens_used'] == 150
            
            # Check request payload
            if log['request_payload']:
                assert log['request_payload'].get('page') == 1
            
            # Check response metadata
            if log['response_metadata']:
                assert 'extracted_chars' in log['response_metadata']
                assert log['response_metadata']['page_number'] == 1
    
    def test_api_logging_with_invalid_document_id(self, pdf_extractor, db_connection):
        """Test API logging with non-UUID document ID."""
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test content"
        
        # Get initial count
        initial_count = len(db_connection.client.table('api_usage_logs')
                          .select('*').eq('user_id', TEST_USER_ID).execute().data)
        
        # Call with non-UUID document_id
        pdf_extractor._log_api_usage(
            api_provider="mistral",
            api_type="ocr",
            tokens_used=100,
            response=mock_response,
            document_id="not_a_uuid_string",
            page_number=1
        )
        
        # Should still log successfully (document_id should be None in DB)
        final_count = len(db_connection.client.table('api_usage_logs')
                        .select('*').eq('user_id', TEST_USER_ID).execute().data)
        
        assert final_count > initial_count, "API usage should be logged even with invalid document_id"
        
        # Check that non-UUID document_id was stored in request_payload
        latest_log = db_connection.client.table('api_usage_logs')\
            .select('*')\
            .eq('user_id', TEST_USER_ID)\
            .order('created_at', desc=True)\
            .limit(1)\
            .execute()
        
        if latest_log.data:
            log = latest_log.data[0]
            assert log['document_id'] is None  # Should be None since it's not a valid UUID
            
            # Non-UUID should be stored in request_payload
            if log['request_payload']:
                assert log['request_payload'].get('original_document_id') == "not_a_uuid_string"

class TestPDFExtractorPerformance:
    """Test PDF extractor performance and reliability."""
    
    def test_fallback_extraction_performance(self, pdf_extractor):
        """Test fallback extraction method performance."""
        import time
        
        # Create a simple test PDF content
        test_pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Test content) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000010 00000 n 
0000000079 00000 n 
0000000173 00000 n 
0000000301 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
398
%%EOF"""
        
        # Write test PDF
        test_pdf_path = "test_performance.pdf"
        with open(test_pdf_path, "wb") as f:
            f.write(test_pdf_content)
        
        try:
            start_time = time.time()
            result = pdf_extractor._extract_with_fallback(test_pdf_path)
            end_time = time.time()
            
            duration = end_time - start_time
            assert duration < 5.0, f"Fallback extraction took too long: {duration:.2f}s"
            
            # Check result structure
            assert "text" in result
            assert "tables" in result
            assert "metadata" in result
            assert "source" in result
            
        finally:
            if os.path.exists(test_pdf_path):
                os.remove(test_pdf_path)
    
    def test_error_handling_missing_file(self, pdf_extractor):
        """Test error handling for missing PDF file."""
        with pytest.raises(Exception):
            pdf_extractor.extract_from_pdf("nonexistent_file.pdf")
    
    def test_error_handling_invalid_pdf(self, pdf_extractor):
        """Test error handling for invalid PDF file."""
        # Create invalid PDF file
        invalid_pdf_path = "invalid_test.pdf"
        with open(invalid_pdf_path, "w") as f:
            f.write("This is not a PDF file content")
        
        try:
            # Should handle gracefully and fall back
            result = pdf_extractor.extract_from_pdf(invalid_pdf_path)
            # Should return some result structure even if extraction fails
            assert isinstance(result, dict)
            
        finally:
            if os.path.exists(invalid_pdf_path):
                os.remove(invalid_pdf_path)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 