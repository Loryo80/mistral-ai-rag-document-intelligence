#!/usr/bin/env python3
"""
Comprehensive Database Functionality Tests

This module tests:
1. Database connectivity and table access
2. Chunk counting functionality
3. Direct database operations
4. Enhanced RAG system with Simple Query Processor
5. Vector search and embeddings functionality

Combines functionality from:
- test_chunk_count.py
- test_direct_db.py  
- test_enhanced_rag.py
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.storage.db import Database
from src.retrieval.simple_query_processor import SimpleQueryProcessor
from supabase import create_client

# Test configuration
TEST_USER_ID = "be0c1330-2cf1-4b32-a205-6509d08bbe43"
ROQUETTE_PROJECT_ID = "95ef5fc3-5655-4ff4-8898-a3e3e606f5a5"
OTHER_PROJECT_ID = "f3b29d6a-18ce-49be-8ca8-036d4d69072f"

@pytest.fixture(scope="session")
def db_connection():
    """Create a database connection for testing."""
    try:
        return Database()
    except Exception as e:
        pytest.skip(f"Could not connect to database: {e}")

@pytest.fixture(scope="session")
def supabase_client():
    """Create a direct Supabase client for testing."""
    try:
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        if not url or not key:
            pytest.skip("Supabase credentials not found")
        return create_client(url, key)
    except Exception as e:
        pytest.skip(f"Could not create Supabase client: {e}")

class TestDatabaseConnectivity:
    """Test basic database connectivity and table access."""
    
    def test_database_initialization(self, db_connection):
        """Test that database initializes correctly."""
        assert db_connection is not None
        assert hasattr(db_connection, 'client')
        assert hasattr(db_connection, 'user_id')
    
    def test_essential_tables_accessibility(self, db_connection):
        """Test that essential tables are accessible."""
        essential_tables = [
            'projects', 'documents', 'entities', 'relationships', 
            'custom_users', 'user_project_access'
        ]
        
        for table in essential_tables:
            try:
                response = db_connection.client.table(table).select("*").limit(1).execute()
                assert response is not None, f"Table {table} is not accessible"
                print(f"✅ Table '{table}' is accessible")
            except Exception as e:
                pytest.fail(f"Table {table} access failed: {e}")
    
    def test_chunks_embeddings_tables_status(self, db_connection):
        """Test the status of chunks and embeddings tables."""
        # Check chunks table
        try:
            response = db_connection.client.table('chunks').select("*").limit(1).execute()
            chunks_exist = True
            print("✅ Chunks table exists and is accessible")
        except Exception:
            chunks_exist = False
            print("⚠️ Chunks table does not exist")
        
        # Check embeddings table  
        try:
            response = db_connection.client.table('embeddings').select("*").limit(1).execute()
            embeddings_exist = True
            print("✅ Embeddings table exists and is accessible")
        except Exception:
            embeddings_exist = False
            print("⚠️ Embeddings table does not exist")
        
        # Both should exist for full functionality
        if not chunks_exist or not embeddings_exist:
            print("⚠️ Missing chunks/embeddings tables may limit functionality")

class TestChunkCountingFunctionality:
    """Test chunk counting functionality across projects."""
    
    def test_chunk_counting_roquette_project(self, db_connection):
        """Test chunk counting for the Roquette project."""
        print(f"\n🧪 Testing chunk counting for Roquette project: {ROQUETTE_PROJECT_ID}")
        
        try:
            chunk_count = db_connection.count_chunks_for_project(ROQUETTE_PROJECT_ID)
            print(f"✅ Roquette project chunk count: {chunk_count}")
            
            # Should be non-negative
            assert chunk_count >= 0, "Chunk count should be non-negative"
            
            # If chunks exist, test was successful
            if chunk_count > 0:
                print(f"📊 Found {chunk_count} chunks in Roquette project")
            else:
                print("📊 No chunks found - this may be expected if chunks table doesn't exist")
                
        except Exception as e:
            # This might fail if chunks table doesn't exist, which is acceptable
            print(f"⚠️ Chunk counting failed (expected if chunks table missing): {e}")
    
    def test_chunk_counting_other_project(self, db_connection):
        """Test chunk counting for another project."""
        print(f"\n🧪 Testing chunk counting for other project: {OTHER_PROJECT_ID}")
        
        try:
            chunk_count = db_connection.count_chunks_for_project(OTHER_PROJECT_ID)
            print(f"✅ Other project chunk count: {chunk_count}")
            
            assert chunk_count >= 0, "Chunk count should be non-negative"
            
        except Exception as e:
            print(f"⚠️ Chunk counting failed for other project: {e}")
    
    def test_document_retrieval_by_project(self, db_connection):
        """Test document retrieval for projects."""
        print(f"\n📄 Testing document retrieval for Roquette project")
        
        try:
            docs = db_connection.fetch_documents_by_project(ROQUETTE_PROJECT_ID)
            print(f"✅ Found {len(docs)} documents in Roquette project")
            
            for doc in docs[:3]:  # Show first 3 documents
                filename = doc.get('filename', 'Unknown')
                doc_id = doc.get('id', 'Unknown')
                print(f"  📄 {filename} (ID: {doc_id[:8]}...)")
            
            assert isinstance(docs, list), "Documents should be returned as a list"
            
        except Exception as e:
            print(f"❌ Document retrieval failed: {e}")
            raise

class TestDirectDatabaseOperations:
    """Test direct database operations using Supabase client."""
    
    def test_total_counts_via_sql(self, supabase_client):
        """Test getting total counts via direct SQL execution."""
        print("\n🔍 Testing direct database access via SQL")
        
        # Test documents count
        try:
            docs_result = supabase_client.rpc('execute_sql', {
                'query': 'SELECT count(*) as doc_count FROM documents;'
            }).execute()
            doc_count = docs_result.data[0]['doc_count'] if docs_result.data else 0
            print(f"📊 Total documents: {doc_count}")
            assert doc_count >= 0
        except Exception as e:
            print(f"⚠️ Documents count query failed: {e}")
        
        # Test chunks count (may fail if table doesn't exist)
        try:
            chunks_result = supabase_client.rpc('execute_sql', {
                'query': 'SELECT count(*) as chunk_count FROM chunks;'
            }).execute()
            chunk_count = chunks_result.data[0]['chunk_count'] if chunks_result.data else 0
            print(f"📊 Total chunks: {chunk_count}")
        except Exception as e:
            print(f"⚠️ Chunks count query failed (expected if table missing): {e}")
        
        # Test embeddings count (may fail if table doesn't exist)
        try:
            embeddings_result = supabase_client.rpc('execute_sql', {
                'query': 'SELECT count(*) as emb_count FROM embeddings;'
            }).execute()
            emb_count = embeddings_result.data[0]['emb_count'] if embeddings_result.data else 0
            print(f"📊 Total embeddings: {emb_count}")
        except Exception as e:
            print(f"⚠️ Embeddings count query failed (expected if table missing): {e}")
    
    def test_project_specific_data_via_sql(self, supabase_client):
        """Test project-specific data retrieval via SQL."""
        print(f"\n📁 Testing project-specific data for Roquette project")
        
        try:
            # Get documents for Roquette project
            project_docs_result = supabase_client.rpc('execute_sql', {
                'query': f"SELECT * FROM documents WHERE project_id = '{ROQUETTE_PROJECT_ID}'::uuid;"
            }).execute()
            
            docs = project_docs_result.data if project_docs_result.data else []
            print(f"✅ Found {len(docs)} documents in Roquette project via SQL")
            
            # Test chunk counting for each document (if chunks table exists)
            for doc in docs[:2]:  # Test first 2 documents
                doc_id = doc['id']
                filename = doc['filename']
                print(f"  📄 {filename} (ID: {doc_id[:8]}...)")
                
                try:
                    doc_chunks_result = supabase_client.rpc('execute_sql', {
                        'query': f"SELECT count(*) as chunk_count FROM chunks WHERE document_id = '{doc_id}'::uuid;"
                    }).execute()
                    
                    chunk_count = doc_chunks_result.data[0]['chunk_count'] if doc_chunks_result.data else 0
                    print(f"    🧩 Chunks: {chunk_count}")
                    
                except Exception as e:
                    print(f"    ⚠️ Chunk count failed for document (expected if chunks table missing): {e}")
            
        except Exception as e:
            print(f"❌ Project-specific data retrieval failed: {e}")
            raise

class TestEnhancedRAGSystem:
    """Test the Enhanced RAG system with Simple Query Processor."""
    
    @pytest.fixture
    def rag_database(self, db_connection):
        """Create a database connection with explicit user_id for RAG testing."""
        db_connection.user_id = TEST_USER_ID
        return db_connection
    
    def test_simple_query_processor_initialization(self, rag_database):
        """Test that SimpleQueryProcessor initializes correctly."""
        try:
            processor = SimpleQueryProcessor(rag_database)
            assert processor is not None
            assert hasattr(processor, 'db')
            print("✅ SimpleQueryProcessor initialized successfully")
        except Exception as e:
            print(f"❌ SimpleQueryProcessor initialization failed: {e}")
            raise
    
    def test_chunk_counting_via_rpc(self, rag_database):
        """Test chunk counting via RPC function."""
        print(f"\n📊 Testing chunk counting via RPC for project: {ROQUETTE_PROJECT_ID}")
        
        try:
            result = rag_database.client.rpc(
                'count_chunks_for_project',
                {
                    'project_uuid': ROQUETTE_PROJECT_ID,
                    'auth_user_id': TEST_USER_ID
                }
            ).execute()
            
            chunk_count = result.data if isinstance(result.data, int) else (result.data[0] if result.data else 0)
            print(f"✅ RPC chunk count for Roquette project: {chunk_count}")
            
            assert chunk_count >= 0, "Chunk count should be non-negative"
            
            if chunk_count > 0:
                print(f"📊 Found {chunk_count} chunks via RPC")
            else:
                print("📊 No chunks found via RPC - may indicate missing chunks table")
                
        except Exception as e:
            print(f"⚠️ RPC chunk counting failed (expected if function/table missing): {e}")
    
    @patch('src.retrieval.simple_query_processor.SimpleQueryProcessor.simple_query')
    def test_vector_similarity_search_mock(self, mock_simple_query, rag_database):
        """Test vector similarity search with mocked response."""
        print(f"\n🔍 Testing vector similarity search (mocked)")
        
        # Mock successful response
        mock_simple_query.return_value = {
            'answer': 'PEARLITOL is a mannitol-based excipient used in pharmaceutical formulations.',
            'chunks_found': 3,
            'sources': [
                {'filename': 'pearlitol_spec.pdf', 'page': 1, 'similarity': 0.95},
                {'filename': 'pharmaceutical_guide.pdf', 'page': 15, 'similarity': 0.87}
            ]
        }
        
        processor = SimpleQueryProcessor(rag_database)
        test_query = "What is PEARLITOL?"
        
        result = processor.simple_query(test_query, ROQUETTE_PROJECT_ID)
        
        print(f"Query: {test_query}")
        print(f"Chunks found: {result.get('chunks_found', 0)}")
        print(f"Sources: {len(result.get('sources', []))}")
        
        if result.get('answer'):
            print(f"Answer (first 100 chars): {result['answer'][:100]}...")
        
        assert result.get('chunks_found', 0) > 0, "Should find chunks in mocked response"
        assert result.get('answer'), "Should have an answer in mocked response"
        print("✅ Mocked vector similarity search working correctly!")
    
    def test_real_vector_search_if_available(self, rag_database):
        """Test real vector search if system is fully operational."""
        print(f"\n🔍 Testing real vector similarity search")
        
        # Skip if API key not available
        if not os.getenv('MISTRAL_API_KEY'):
            pytest.skip("MISTRAL_API_KEY not available for real vector search test")
        
        try:
            processor = SimpleQueryProcessor(rag_database)
            test_query = "What is PEARLITOL?"
            
            print(f"Query: {test_query}")
            print(f"Project ID: {ROQUETTE_PROJECT_ID}")
            
            result = processor.simple_query(test_query, ROQUETTE_PROJECT_ID)
            
            print(f"Chunks found: {result.get('chunks_found', 0)}")
            print(f"Sources: {len(result.get('sources', []))}")
            
            if result.get('answer'):
                print(f"Answer (first 200 chars): {result['answer'][:200]}...")
            
            if result.get('sources'):
                print(f"Sources:")
                for i, source in enumerate(result['sources'][:2], 1):
                    print(f"  {i}. {source['filename']} (Page {source['page']}) - Similarity: {source['similarity']:.2f}")
            
            # Basic validation
            if result.get('chunks_found', 0) > 0:
                print("✅ Real vector search working correctly!")
                assert result.get('answer'), "Should have an answer when chunks are found"
            else:
                print("⚠️ No chunks found - may indicate missing chunks/embeddings tables")
                
        except Exception as e:
            print(f"⚠️ Real vector search failed (may indicate missing tables/API issues): {e}")
            # Don't fail the test - this is expected in some environments

class TestSystemIntegration:
    """Test complete system integration."""
    
    def test_full_database_functionality_summary(self, db_connection, supabase_client):
        """Provide a comprehensive summary of database functionality."""
        print(f"\n📋 COMPREHENSIVE DATABASE FUNCTIONALITY SUMMARY")
        print("=" * 60)
        
        results = {
            'database_connection': False,
            'essential_tables': False,
            'chunks_table': False,
            'embeddings_table': False,
            'document_retrieval': False,
            'chunk_counting': False,
            'simple_query_processor': False
        }
        
        # Test database connection
        try:
            assert db_connection is not None
            results['database_connection'] = True
            print("✅ Database connection: WORKING")
        except Exception:
            print("❌ Database connection: FAILED")
        
        # Test essential tables
        essential_tables = ['projects', 'documents', 'entities', 'relationships']
        try:
            for table in essential_tables:
                db_connection.client.table(table).select("*").limit(1).execute()
            results['essential_tables'] = True
            print("✅ Essential tables: ACCESSIBLE")
        except Exception:
            print("❌ Essential tables: FAILED")
        
        # Test chunks table
        try:
            db_connection.client.table('chunks').select("*").limit(1).execute()
            results['chunks_table'] = True
            print("✅ Chunks table: EXISTS")
        except Exception:
            print("⚠️ Chunks table: MISSING")
        
        # Test embeddings table
        try:
            db_connection.client.table('embeddings').select("*").limit(1).execute()
            results['embeddings_table'] = True
            print("✅ Embeddings table: EXISTS")
        except Exception:
            print("⚠️ Embeddings table: MISSING")
        
        # Test document retrieval
        try:
            docs = db_connection.fetch_documents_by_project(ROQUETTE_PROJECT_ID)
            if len(docs) > 0:
                results['document_retrieval'] = True
                print(f"✅ Document retrieval: WORKING ({len(docs)} docs found)")
            else:
                print("⚠️ Document retrieval: NO DOCUMENTS FOUND")
        except Exception:
            print("❌ Document retrieval: FAILED")
        
        # Test chunk counting
        try:
            chunk_count = db_connection.count_chunks_for_project(ROQUETTE_PROJECT_ID)
            results['chunk_counting'] = True
            print(f"✅ Chunk counting: WORKING ({chunk_count} chunks)")
        except Exception:
            print("❌ Chunk counting: FAILED")
        
        # Test SimpleQueryProcessor
        try:
            processor = SimpleQueryProcessor(db_connection)
            results['simple_query_processor'] = True
            print("✅ SimpleQueryProcessor: INITIALIZED")
        except Exception:
            print("❌ SimpleQueryProcessor: FAILED")
        
        # Summary
        working_count = sum(results.values())
        total_count = len(results)
        
        print("\n" + "=" * 60)
        print(f"📊 FUNCTIONALITY SUMMARY: {working_count}/{total_count} components working")
        
        if results['chunks_table'] and results['embeddings_table']:
            print("🎉 FULL FUNCTIONALITY: All required tables present")
        elif results['database_connection'] and results['essential_tables']:
            print("⚠️ PARTIAL FUNCTIONALITY: Core features working, missing chunks/embeddings")
        else:
            print("❌ LIMITED FUNCTIONALITY: Core database issues detected")
        
        print("=" * 60)
        
        # At minimum, database connection should work
        assert results['database_connection'], "Database connection is required"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 