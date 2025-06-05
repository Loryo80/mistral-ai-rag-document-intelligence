#!/usr/bin/env python
import os
import sys
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Supabase credentials directly, not from another module
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def create_tables():
    """Create all required database tables in Supabase."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Supabase URL and Key must be set in environment variables.")
        logger.error(f"SUPABASE_URL set: {bool(SUPABASE_URL)}")
        logger.error(f"SUPABASE_KEY set: {bool(SUPABASE_KEY)}")
        raise ValueError("Supabase URL and Key must be set in environment variables.")
    
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Connected to Supabase.")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
        raise
        
    # Dictionary of table definitions
    tables = {
        "custom_users": """
            CREATE TABLE IF NOT EXISTS custom_users (
                id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT,
                role TEXT DEFAULT 'user',
                api_quota INTEGER DEFAULT 100,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                last_login TIMESTAMP WITH TIME ZONE
            );
        """,
        "projects": """
            CREATE TABLE IF NOT EXISTS projects (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT NOT NULL,
                description TEXT,
                metadata JSONB,
                card_color TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
        """,
        "user_project_access": """
            CREATE TABLE IF NOT EXISTS user_project_access (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
                project_id uuid REFERENCES projects(id) ON DELETE CASCADE,
                access_level TEXT DEFAULT 'read',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                UNIQUE(user_id, project_id)
            );
        """,
        "documents": """
            CREATE TABLE IF NOT EXISTS documents (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                filename TEXT NOT NULL,
                num_pages INTEGER,
                metadata JSONB,
                project_id uuid REFERENCES projects(id) ON DELETE CASCADE,
                version INTEGER DEFAULT 1,
                file_hash TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
        """,
        "chunks": """
            CREATE TABLE IF NOT EXISTS chunks (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
                chunk_text TEXT NOT NULL,
                chunk_order INTEGER,
                page_number INTEGER,
                metadata JSONB
            );
        """,
        "embeddings": """
            CREATE TABLE IF NOT EXISTS embeddings (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                chunk_id uuid REFERENCES chunks(id) ON DELETE CASCADE,
                document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
                embedding vector(1024),
                metadata JSONB
            );
        """,
        "entities": """
            CREATE TABLE IF NOT EXISTS entities (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
                chunk_id uuid REFERENCES chunks(id) ON DELETE SET NULL,
                entity_type TEXT NOT NULL,
                entity_value TEXT NOT NULL,
                start_char INTEGER,
                end_char INTEGER,
                metadata JSONB
            );
        """,
        "relationships": """
            CREATE TABLE IF NOT EXISTS relationships (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
                chunk_id uuid REFERENCES chunks(id) ON DELETE SET NULL,
                entity_1 TEXT NOT NULL,
                entity_2 TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                context TEXT,
                metadata JSONB
            );
        """,
        "api_usage_logs": """
            CREATE TABLE IF NOT EXISTS api_usage_logs (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT,
                document_id uuid REFERENCES documents(id) ON DELETE SET NULL,
                api_provider TEXT,
                api_type TEXT,
                tokens_used INTEGER,
                cost_usd NUMERIC(10,4),
                request_payload JSONB,
                response_metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
        """
    }
    
    # Create each table
    for table_name, sql in tables.items():
        try:
            # Check if table exists
            try:
                client.table(table_name).select("*").limit(1).execute()
                logger.info(f"Table '{table_name}' already exists.")
                continue  # Skip to next table
            except Exception:
                logger.info(f"Table '{table_name}' does not exist, attempting to create.")
            
            # Try to create the table using SQL query
            try:
                # For the custom_users table, be careful with the RPC call as it might already exist
                if table_name == "custom_users":
                    logger.info("Custom_users table requires special handling, attempting creation...")
                    try:
                        response = client.postgrest.rpc('execute_sql', {'query': sql}).execute()
                        if hasattr(response, 'error') and response.error:
                            logger.warning(f"Note about custom_users table: {response.error}")
                        else:
                            logger.info("Custom_users table created or already exists.")
                    except Exception as e:
                        logger.warning(f"Note about custom_users table: {e}")
                else:
                    response = client.postgrest.rpc('execute_sql', {'query': sql}).execute()
                    if hasattr(response, 'error') and response.error:
                        logger.error(f"Error creating table '{table_name}': {response.error}")
                    else:
                        logger.info(f"Table '{table_name}' created successfully.")
            except Exception as e:
                logger.error(f"Failed to create table '{table_name}' via RPC: {e}")
                
        except Exception as e:
            logger.error(f"Error processing table '{table_name}': {e}")
    
    # Try to create the get_current_user_profile function
    try:
        user_profile_function = """
        CREATE OR REPLACE FUNCTION get_current_user_profile()
        RETURNS SETOF custom_users
        LANGUAGE sql
        SECURITY definer
        SET search_path = public
        AS $$
            SELECT * FROM custom_users WHERE id = auth.uid()
        $$;
        """
        
        client.postgrest.rpc('execute_sql', {'query': user_profile_function}).execute()
        logger.info("User profile function created or updated.")
    except Exception as e:
        logger.error(f"Error creating user profile function: {e}")
    
    # Create vector similarity search function (if pgvector extension is enabled)
    try:
        # First try to enable pgvector extension if not already enabled
        try:
            client.postgrest.rpc('execute_sql', {'query': 'CREATE EXTENSION IF NOT EXISTS vector;'}).execute()
            logger.info("Pgvector extension enabled.")
        except Exception as e:
            logger.warning(f"Could not enable pgvector extension: {e}")
        
        vector_function = """
        CREATE OR REPLACE FUNCTION match_embeddings(
            query_embedding vector(1024),
            match_threshold float,
            match_count int
        )
        RETURNS TABLE(
            chunk_id uuid,
            document_id uuid,
            chunk_text text,
            page_number integer,
            similarity float
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                c.id as chunk_id,
                c.document_id,
                c.chunk_text,
                c.page_number,
                1 - (e.embedding <=> query_embedding) as similarity
            FROM
                embeddings e
            JOIN
                chunks c ON e.chunk_id = c.id
            WHERE
                1 - (e.embedding <=> query_embedding) > match_threshold
            ORDER BY
                similarity DESC
            LIMIT
                match_count;
        END;
        $$;
        """
        
        # Try to create the function
        try:
            client.postgrest.rpc('execute_sql', {'query': vector_function}).execute()
            logger.info("Vector similarity search function created.")
        except Exception as e:
            logger.error(f"Failed to create vector similarity function: {e}")
    except Exception as e:
        logger.error(f"Error setting up vector similarity function: {e}")
    
    logger.info("Database setup completed.")

if __name__ == "__main__":
    try:
        create_tables()
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        sys.exit(1) 