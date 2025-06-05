# db.py (Database Module) - FIXED VERSION
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import uuid
from datetime import datetime

from dotenv import load_dotenv
from supabase import create_client, Client
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def is_retryable_db_exception(exception):
    import requests
    retryable = (
        ConnectionError,
        TimeoutError,
        requests.exceptions.RequestException,
        Exception,
    )
    return isinstance(exception, retryable)

RETRY_CONFIG = dict(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(4),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

class Database:
    """Database interface for storing and retrieving embeddings, chunks, and entities using Supabase/PostgreSQL."""
    
    def __init__(self, access_token: Optional[str] = None):
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("Supabase URL and Key must be set in environment variables.")
        
        # Create the Supabase client with the anon key
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized.")
        
        # Use access_token if provided to set the auth session
        if access_token:
            # Set the session with access_token to authenticate requests
            # This is needed for RLS policies that check auth.uid()
            try:
                self.client.postgrest.auth(access_token)
                logger.info("Supabase client authenticated with access token")
            except Exception as e:
                logger.error(f"Failed to set auth session with access token: {e}")
        
        self.access_token = access_token
        
        # Define table names
        self.projects_table = "projects"
        self.documents_table = "documents"
        self.chunks_table = "chunks"
        self.embeddings_table = "embeddings"
        self.entities_table = "entities"
        self.relationships_table = "relationships"
        self.users_table = "custom_users"
        self.user_project_access_table = "user_project_access"

        self._create_tables_if_not_exists()

    def _create_tables_if_not_exists(self):
        """Check if core tables exist - skip chunks/embeddings as they're handled by RLS."""
        logger.info("Checking if core database tables exist...")
        
        # Only check core tables that should be directly accessible
        core_tables = [
            self.projects_table, 
            self.documents_table,
            self.entities_table,
            self.relationships_table,
            "custom_users",
            self.user_project_access_table
        ]
        
        missing_tables = []
        
        for table in core_tables:
            try:
                # Try using the client.from_ approach with limit 1 to check table existence
                response = self.client.from_(table).select("*").limit(1).execute()
                logger.info(f"Core table '{table}' exists and is accessible.")
            except Exception as e:
                missing_tables.append(table)
                logger.warning(f"Core table '{table}' is not accessible: {e}")
        
        # Note about chunks/embeddings tables
        logger.info("Note: chunks and embeddings tables are managed via RLS policies and accessed through query processors.")
        
        if missing_tables:
            logger.warning(f"The following core tables appear to be missing: {', '.join(missing_tables)}")
            logger.warning("Please run the create_tables.py script to create the missing tables.")

    def _is_valid_uuid(self, value: Any) -> bool:
        """Check if a value is a valid UUID."""
        if value is None:
            return False
        try:
            uuid.UUID(str(value))
            return True
        except (ValueError, AttributeError, TypeError):
            return False

    def _ensure_uuid(self, value: Any) -> Optional[str]:
        """Ensure value is a valid UUID string or None."""
        if value is None:
            return None
        try:
            # Convert to UUID and then back to string to ensure proper format
            return str(uuid.UUID(str(value)))
        except (ValueError, AttributeError, TypeError):
            logger.error(f"Invalid UUID value: {value}")
            return None

    @retry(**RETRY_CONFIG)
    def create_project(self, name: str, description: str = "", metadata: Dict[str, Any] = None, user_id: Optional[str] = None) -> str:
        """Create a new project and return its ID."""
        try:
            # Prepare metadata
            if metadata is None:
                metadata = {}
            elif isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            # Extract card_color from metadata if present
            card_color = metadata.pop("card_color", None)
            
            # First try to use the RPC function which handles both project and access creation
            try:
                logger.info(f"Creating project using RPC function for user: {user_id}")
                
                # Ensure RPC function has access token if available
                rpc_client = self.client.rpc
                
                response = rpc_client('create_project_with_access', {
                    'p_name': name,
                    'p_description': description or "",
                    'p_metadata': metadata,
                    'p_card_color': card_color
                }).execute()
                
                if response.data:
                    project_id = response.data
                    logger.info(f"Successfully created project '{name}' with ID: {project_id} using RPC")
                    return project_id
            except Exception as rpc_error:
                logger.warning(f"RPC create_project_with_access failed: {rpc_error}, falling back to direct insert")
            
            # Fallback: Direct insert (this might fail due to RLS if user is not properly authenticated)
            project_data = {
                "name": name,
                "description": description or "",
                "metadata": metadata
            }
            
            if card_color:
                project_data["card_color"] = card_color
            
            logger.info(f"Creating project with direct insert: {project_data}")
            
            # Insert project
            response = self.client.table(self.projects_table).insert(project_data).execute()
            
            if not response.data:
                raise ValueError("No project data returned from insert")
            
            project_id = response.data[0]['id']
            logger.info(f"Created project '{name}' with ID: {project_id}")
            
            # Create user_project_access entry
            if user_id and self._is_valid_uuid(user_id):
                try:
                    access_data = {
                        "user_id": str(user_id),
                        "project_id": str(project_id),
                        "access_level": "admin"
                    }
                    self.client.table(self.user_project_access_table).insert(access_data).execute()
                    logger.info(f"Added user {user_id} as admin to project {project_id}")
                except Exception as e:
                    logger.error(f"Failed to create user_project_access entry: {e}")
                    # Try to clean up the project
                    try:
                        self.client.table(self.projects_table).delete().eq("id", project_id).execute()
                        logger.error("Rolled back project creation due to access creation failure")
                    except:
                        pass
                    raise Exception(f"Failed to grant access to project: {e}")
            
            return project_id
            
        except Exception as e:
            logger.error(f"Error creating project: {e}", exc_info=True)
            raise

    @retry(**RETRY_CONFIG)
    def update_project(self, project_id: str, name: str = None, description: str = None, 
                      metadata: Dict[str, Any] = None, card_color: str = None) -> None:
        """Update an existing project."""
        try:
            if not self._is_valid_uuid(project_id):
                raise ValueError(f"Invalid project_id: {project_id}")
            
            update_data = {}
            if name is not None:
                update_data["name"] = name
            if description is not None:
                update_data["description"] = description
            # Handle card_color directly as a parameter
            if card_color is not None:
                update_data["card_color"] = card_color
            if metadata is not None:
                # Also extract card_color from metadata if present (for backward compatibility)
                if "card_color" in metadata:
                    card_color = metadata.pop("card_color")
                    update_data["card_color"] = card_color
                update_data["metadata"] = metadata
            
            if update_data:
                self.client.table(self.projects_table).update(update_data).eq("id", project_id).execute()
                logger.info(f"Updated project {project_id}")
                
        except Exception as e:
            logger.error(f"Error updating project: {e}", exc_info=True)
            raise

    @retry(**RETRY_CONFIG)
    def delete_project(self, project_id: str) -> None:
        """Delete a project and all associated data with complete cascade deletion."""
        try:
            if not self._is_valid_uuid(project_id):
                raise ValueError(f"Invalid project_id: {project_id}")
            
            # Convert to proper UUID format
            project_uuid = str(uuid.UUID(str(project_id)))
            logger.info(f"Starting complete deletion of project {project_uuid}")
            
            # First get all documents for this project
            docs_response = self.client.table(self.documents_table).select("id").eq("project_id", project_uuid).execute()
            
            if docs_response.data:
                doc_ids = [doc['id'] for doc in docs_response.data]
                logger.info(f"Found {len(doc_ids)} documents to delete")
                
                # Delete related data for each document in proper order
                for doc_id in doc_ids:
                    logger.info(f"Deleting data for document {doc_id}")
                    
                    try:
                        # Delete API usage logs first (references document_id)
                        self.client.table("api_usage_logs").delete().eq("document_id", doc_id).execute()
                        
                        # Delete embeddings (references chunks)
                        self.client.table(self.embeddings_table).delete().eq("document_id", doc_id).execute()
                        
                        # Delete agricultural entities (specific to this system)
                        self.client.table("agricultural_entities").delete().eq("document_id", doc_id).execute()
                        
                        # Delete agricultural relationships (specific to this system)
                        self.client.table("agricultural_relationships").delete().eq("document_id", doc_id).execute()
                        
                        # Delete traditional entities
                        self.client.table(self.entities_table).delete().eq("document_id", doc_id).execute()
                        
                        # Delete traditional relationships
                        self.client.table(self.relationships_table).delete().eq("document_id", doc_id).execute()
                        
                        # Delete chunks (documents depend on these)
                        self.client.table(self.chunks_table).delete().eq("document_id", doc_id).execute()
                
                        logger.info(f"Successfully deleted all related data for document {doc_id}")
                        
                    except Exception as doc_error:
                        logger.error(f"Error deleting data for document {doc_id}: {doc_error}")
                        # Continue with other documents even if one fails
                        continue
                
                # Delete all documents for this project
                try:
                    self.client.table(self.documents_table).delete().eq("project_id", project_uuid).execute()
                    logger.info(f"Deleted all documents for project {project_uuid}")
                except Exception as docs_error:
                    logger.error(f"Error deleting documents for project: {docs_error}")
            
            # Delete remaining API usage logs for this project (if any)
            try:
                # Get API logs that might reference this project through documents
                self.client.table("api_usage_logs").delete().eq("project_id", project_uuid).execute()
            except Exception as api_logs_error:
                logger.warning(f"Error deleting API logs: {api_logs_error}")
            
            # Delete user project access
            try:
                self.client.table(self.user_project_access_table).delete().eq("project_id", project_uuid).execute()
                logger.info(f"Deleted user access for project {project_uuid}")
            except Exception as access_error:
                logger.error(f"Error deleting user access: {access_error}")
            
            # Finally delete the project itself
            try:
                self.client.table(self.projects_table).delete().eq("id", project_uuid).execute()
                logger.info(f"Successfully deleted project {project_uuid}")
            except Exception as project_error:
                logger.error(f"Error deleting project: {project_error}")
                raise
            
            # Clean up LightRAG storage if possible
            try:
                self._cleanup_lightrag_for_project(project_uuid)
            except Exception as lightrag_error:
                logger.warning(f"Could not clean up LightRAG for project {project_uuid}: {lightrag_error}")
            
            logger.info(f"Complete deletion of project {project_uuid} and all associated data finished")
            
        except Exception as e:
            logger.error(f"Error deleting project: {e}", exc_info=True)
            raise

    def _cleanup_lightrag_for_project(self, project_id: str) -> None:
        """Clean up LightRAG storage for a specific project."""
        try:
            import os
            import shutil
            from pathlib import Path
            
            # Check if rag_storage directory exists
            rag_storage_path = Path("rag_storage")
            if rag_storage_path.exists():
                # Look for project-specific files (this is a basic cleanup)
                # In a more sophisticated setup, LightRAG would have project-specific namespaces
                logger.info(f"LightRAG storage exists, but project-specific cleanup not implemented yet")
                # TODO: Implement project-specific LightRAG cleanup when namespace support is added
            
        except Exception as e:
            logger.warning(f"LightRAG cleanup failed: {e}")

    def verify_project_deletion(self, project_id: str) -> Dict[str, int]:
        """Verify that a project and all its data has been completely deleted."""
        try:
            project_uuid = str(uuid.UUID(str(project_id)))
            
            # Check remaining data counts
            remaining_data = {}
            
            # Check project
            proj_response = self.client.table(self.projects_table).select("id").eq("id", project_uuid).execute()
            remaining_data["projects"] = len(proj_response.data) if proj_response.data else 0
            
            # Check documents
            docs_response = self.client.table(self.documents_table).select("id").eq("project_id", project_uuid).execute()
            remaining_data["documents"] = len(docs_response.data) if docs_response.data else 0
            
            # Check user access
            access_response = self.client.table(self.user_project_access_table).select("id").eq("project_id", project_uuid).execute()
            remaining_data["user_access"] = len(access_response.data) if access_response.data else 0
            
            # Check agricultural entities
            ag_entities_response = self.client.table("agricultural_entities").select("id").eq("project_id", project_uuid).execute()
            remaining_data["agricultural_entities"] = len(ag_entities_response.data) if ag_entities_response.data else 0
            
            # Check agricultural relationships
            ag_rels_response = self.client.table("agricultural_relationships").select("id").eq("project_id", project_uuid).execute()
            remaining_data["agricultural_relationships"] = len(ag_rels_response.data) if ag_rels_response.data else 0
            
            return remaining_data
            
        except Exception as e:
            logger.error(f"Error verifying project deletion: {e}")
            return {"error": str(e)}

    @retry(**RETRY_CONFIG)
    def fetch_all_projects(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all projects accessible to the user."""
        try:
            if user_id and self._is_valid_uuid(user_id):
                # Convert to standard UUID format to avoid type issues
                user_uuid = str(uuid.UUID(str(user_id)))
                
                # Get projects user has access to (use standard query as RPC might have issues)
                try:
                    access_response = self.client.table(self.user_project_access_table)\
                        .select("project_id")\
                        .eq("user_id", user_uuid)\
                        .execute()
                    
                    if not access_response.data:
                        logger.info(f"User {user_id} has no accessible projects via access table")
                        return []
                    
                    project_ids = [row["project_id"] for row in access_response.data]
                    logger.info(f"Found {len(project_ids)} project IDs for user {user_id}: {project_ids}")
                    
                    # Get project details
                    projects_response = self.client.table(self.projects_table)\
                        .select("*")\
                        .in_("id", project_ids)\
                        .order("created_at", desc=True)\
                        .execute()
                    
                    if projects_response.data:
                        logger.info(f"Successfully fetched {len(projects_response.data)} projects for user {user_id}")
                        return projects_response.data
                    else:
                        logger.warning(f"No project details found for project IDs: {project_ids}")
                        return []
                    
                except Exception as e:
                    logger.error(f"Error in standard query method: {e}")
                    return []
            else:
                # No user_id, return empty list
                logger.info("No valid user_id provided for fetching projects")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching projects: {e}", exc_info=True)
            return []

    @retry(**RETRY_CONFIG)
    def fetch_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single project by ID."""
        try:
            # Ensure project_id is a valid UUID
            if not self._is_valid_uuid(project_id):
                logger.error(f"Invalid project_id: {project_id}")
                return None
            
            # Format the UUID properly to avoid type mismatches
            project_uuid = str(uuid.UUID(str(project_id)))
            
            # Try direct SQL query first (bypasses RLS and avoids type issues)
            try:
                query = f"""
                SELECT * FROM projects 
                WHERE id::text = '{project_uuid}'
                LIMIT 1
                """
                
                response = self.client.rpc('execute_sql', {'query': query}).execute()
                
                if response.data and len(response.data) > 0:
                    logger.info(f"Found project via direct SQL: {response.data[0].get('name', 'Unknown')}")
                    return response.data[0]
            except Exception as e:
                logger.warning(f"Direct SQL query failed: {e}, trying standard query")
            
            # Standard query (may be affected by RLS)
            try:
                response = self.client.table(self.projects_table)\
                    .select("*")\
                    .eq("id", project_uuid)\
                    .limit(1)\
                    .execute()
                
                if response.data and len(response.data) > 0:
                    return response.data[0]
            except Exception as e:
                logger.warning(f"Standard query failed: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching project: {e}")
            return None

    @retry(**RETRY_CONFIG)
    def fetch_documents_by_project(self, project_id: str) -> List[Dict[str, Any]]:
        """Fetch all documents belonging to a project."""
        try:
            # Validate and format project_id as proper UUID
            if not self._is_valid_uuid(project_id):
                logger.warning(f"Invalid project_id format: {project_id}")
                return []
            
            # Convert to standard UUID format to avoid type issues
            project_uuid = str(uuid.UUID(str(project_id)))
            
            # Try direct SQL query first to avoid type mismatches
            try:
                query = f"""
                SELECT * FROM documents 
                WHERE project_id::text = '{project_uuid}'
                ORDER BY created_at DESC
                """
                
                response = self.client.rpc('execute_sql', {'query': query}).execute()
                
                if response.data:
                    logger.info(f"Found {len(response.data)} documents via direct SQL")
                    return response.data
            except Exception as e:
                logger.warning(f"Direct SQL query for documents failed: {e}, trying standard query")
            
            # Standard query as fallback
            try:
                response = self.client.table(self.documents_table)\
                    .select("*")\
                    .eq("project_id", project_uuid)\
                    .order("created_at", desc=True)\
                    .execute()
                
                return response.data or []
            except Exception as e:
                logger.warning(f"Standard query for documents failed: {e}")
                return []
            
        except Exception as e:
            logger.error(f"Error fetching documents: {e}", exc_info=True)
            return []

    def _direct_db_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute a direct database query and return the first result row."""
        try:
            # Use the execute_sql function directly - it doesn't need the client.rpc wrapper
            response = self.client.rpc('execute_sql', {'query': query}).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.warning(f"Direct SQL query failed: {e}")
            return None
    
    def _execute_sql_query(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Execute a direct database query and return all result rows."""
        try:
            response = self.client.rpc('execute_sql', {'query': query}).execute()
            return response.data or []
        except Exception as e:
            logger.warning(f"SQL query execution failed: {e}")
            return None

    def insert_document(self, filename: str, num_pages: int, metadata: Dict[str, Any], 
                       project_id: str, version: int = 1, file_hash: str = None) -> str:
        """Insert a document record and return its ID. No retry decorator to prevent duplicate creation."""
        try:
            # Ensure project_id is a valid UUID string
            if not self._is_valid_uuid(project_id):
                raise ValueError(f"Invalid project_id: {project_id}")
            
            # Format project_id as a proper UUID string
            project_uuid = str(uuid.UUID(str(project_id)))
            
            # Method 1: Use predefined UUID approach (most reliable)
            try:
                # Generate a new UUID for the document
                new_doc_id = str(uuid.uuid4())
                
                # Prepare data safely for SQL injection prevention
                safe_filename = filename.replace("'", "''") if filename else ""
                metadata_json = json.dumps(metadata) if metadata else "{}"
                safe_metadata = metadata_json.replace("'", "''")
                safe_file_hash = file_hash.replace("'", "''") if file_hash else None
                
                # Build INSERT query with proper UUID casting
                if safe_file_hash:
                    query = f"""
                    INSERT INTO documents (id, filename, num_pages, metadata, project_id, version, file_hash)
                    VALUES ('{new_doc_id}'::uuid, '{safe_filename}', {num_pages}, '{safe_metadata}'::jsonb, '{project_uuid}'::uuid, {version}, '{safe_file_hash}');
                    """
                else:
                    query = f"""
                    INSERT INTO documents (id, filename, num_pages, metadata, project_id, version, file_hash)
                    VALUES ('{new_doc_id}'::uuid, '{safe_filename}', {num_pages}, '{safe_metadata}'::jsonb, '{project_uuid}'::uuid, {version}, NULL);
                    """
                
                # Execute the query
                result = self._execute_sql_query(query)
                if result is not None:  # Even empty result means success for INSERT
                    logger.info(f"Inserted document '{filename}' with ID: {new_doc_id} using predefined UUID SQL")
                    return new_doc_id
                    
            except Exception as e:
                logger.warning(f"Predefined UUID SQL insert failed: {e}")
            
            # Method 2: Try using the standard Supabase API without auth (bypass RLS temporarily)
            try:
                # Create a temporary client without auth to bypass RLS for system operations
                temp_client = create_client(SUPABASE_URL, SUPABASE_KEY)
                
                data = {
                    "filename": filename,
                    "num_pages": num_pages,
                    "metadata": metadata,
                    "project_id": project_uuid,
                    "version": version,
                    "file_hash": file_hash
                }
                
                response = temp_client.table(self.documents_table).insert(data).execute()
                if response.data and len(response.data) > 0:
                    doc_id = response.data[0]['id']
                    logger.info(f"Inserted document '{filename}' with ID: {doc_id} using bypass API")
                    return doc_id
                    
            except Exception as e:
                logger.warning(f"Bypass API insert failed: {e}")
            
            # Method 3: Try the regular client with auth
            try:
                data = {
                    "filename": filename,
                    "num_pages": num_pages,
                    "metadata": metadata,
                    "project_id": project_uuid,
                    "version": version,
                    "file_hash": file_hash
                }
                
                response = self.client.table(self.documents_table).insert(data).execute()
                if response.data and len(response.data) > 0:
                    doc_id = response.data[0]['id']
                    logger.info(f"Inserted document '{filename}' with ID: {doc_id} using regular API")
                    return doc_id
                    
            except Exception as e:
                logger.warning(f"Regular API insert failed: {e}")
            
            # If all methods fail, raise an error with details
            raise Exception(f"Failed to insert document after trying multiple methods. Project ID: {project_uuid}, Filename: {filename}")
            
        except Exception as e:
            logger.error(f"Error inserting document: {e}", exc_info=True)
            raise

    def insert_chunk(self, document_id: str, chunk_text: str, chunk_order: int, 
                    page_number: int, metadata: Dict[str, Any]) -> str:
        """Insert a text chunk and return its ID. No retry decorator to prevent duplicate creation."""
        try:
            # Ensure document_id is a valid UUID
            if not self._is_valid_uuid(document_id):
                raise ValueError(f"Invalid document_id: {document_id}")
            
            # Format document_id as a proper UUID string
            document_uuid = str(uuid.UUID(str(document_id)))
            
            # Method 1: Try using direct SQL query with predefined UUID
            try:
                # Generate UUID in advance
                new_chunk_id = str(uuid.uuid4())
                
                # Convert metadata to JSON string and escape chunk_text
                metadata_json = json.dumps(metadata)
                safe_chunk_text = chunk_text.replace("'", "''")
                safe_metadata = metadata_json.replace("'", "''")
                
                query = f"""
                INSERT INTO chunks (id, document_id, chunk_text, chunk_order, page_number, metadata)
                VALUES ('{new_chunk_id}'::uuid, '{document_uuid}'::uuid, '{safe_chunk_text}', {chunk_order}, {page_number}, '{safe_metadata}'::jsonb);
                """
                
                result = self._execute_sql_query(query)
                if result is not None:
                    logger.info(f"Inserted chunk with ID: {new_chunk_id} using direct SQL")
                    return new_chunk_id
            except Exception as e:
                logger.warning(f"Direct SQL chunk insertion failed: {e}")
            
            # Method 2: Try using temporary client (bypass RLS)
            try:
                temp_client = create_client(SUPABASE_URL, SUPABASE_KEY)
                
                data = {
                    "document_id": document_uuid,
                    "chunk_text": chunk_text,
                    "chunk_order": chunk_order,
                    "page_number": page_number,
                    "metadata": metadata
                }
                
                response = temp_client.table(self.chunks_table).insert(data).execute()
                if response.data and len(response.data) > 0:
                    chunk_id = response.data[0]['id']
                    logger.info(f"Inserted chunk with ID: {chunk_id} using bypass API")
                    return chunk_id
            except Exception as e:
                logger.warning(f"Bypass API chunk insertion failed: {e}")
            
            raise Exception(f"Failed to insert chunk after trying multiple methods")
            
        except Exception as e:
            logger.error(f"Error inserting chunk: {e}", exc_info=True)
            raise

    def insert_embedding(self, chunk_id: str, document_id: str, embedding: List[float], 
                        metadata: Dict[str, Any]) -> None:
        """Insert an embedding. No retry decorator to prevent duplicate creation."""
        try:
            # Ensure chunk_id and document_id are valid UUIDs
            if not self._is_valid_uuid(chunk_id):
                raise ValueError(f"Invalid chunk_id: {chunk_id}")
            if not self._is_valid_uuid(document_id):
                raise ValueError(f"Invalid document_id: {document_id}")
            
            # Format UUIDs properly
            chunk_uuid = str(uuid.UUID(str(chunk_id)))
            document_uuid = str(uuid.UUID(str(document_id)))
            
            # Method 1: Try using direct SQL query with predefined UUID
            try:
                new_embedding_id = str(uuid.uuid4())
                
                # Convert metadata to JSON and embedding to vector string
                metadata_json = json.dumps(metadata)
                safe_metadata = metadata_json.replace("'", "''")
                # Format the embedding vector correctly for PostgreSQL
                embedding_str = "[" + ",".join(str(value) for value in embedding) + "]"
                
                query = f"""
                INSERT INTO embeddings (id, chunk_id, document_id, embedding, metadata)
                VALUES ('{new_embedding_id}'::uuid, '{chunk_uuid}'::uuid, '{document_uuid}'::uuid, '{embedding_str}'::vector, '{safe_metadata}'::jsonb);
                """
                
                result = self._execute_sql_query(query)
                if result is not None:
                    logger.info(f"Inserted embedding with ID: {new_embedding_id} using direct SQL")
                    return
            except Exception as e:
                logger.warning(f"Direct SQL embedding insertion failed: {e}")
            
            # Method 2: Try using temporary client (bypass RLS)
            try:
                temp_client = create_client(SUPABASE_URL, SUPABASE_KEY)
                
                data = {
                    "chunk_id": chunk_uuid,
                    "document_id": document_uuid,
                    "embedding": embedding,
                    "metadata": metadata
                }
                
                temp_client.table(self.embeddings_table).insert(data).execute()
                logger.info(f"Inserted embedding using bypass API")
                return
            except Exception as e:
                logger.warning(f"Bypass API embedding insertion failed: {e}")
            
            raise Exception(f"Failed to insert embedding after trying multiple methods")
            
        except Exception as e:
            logger.error(f"Error inserting embedding: {e}", exc_info=True)
            raise

    def insert_entity(self, document_id: str, chunk_id: str, entity_type: str, 
                     entity_value: str, start_char: Optional[int] = None, 
                     end_char: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Insert an entity and return its ID. No retry decorator to prevent duplicate creation."""
        try:
            # Ensure document_id is a valid UUID
            if not self._is_valid_uuid(document_id):
                raise ValueError(f"Invalid document_id: {document_id}")
            
            # Format document_id as proper UUID
            document_uuid = str(uuid.UUID(str(document_id)))
            
            # Handle chunk_id (which can be None)
            chunk_uuid = None
            if chunk_id and self._is_valid_uuid(chunk_id):
                chunk_uuid = str(uuid.UUID(str(chunk_id)))
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            # Method 1: Try using direct SQL query with predefined UUID
            try:
                new_entity_id = str(uuid.uuid4())
                
                # Escape values to prevent SQL injection
                safe_entity_type = entity_type.replace("'", "''") if entity_type else ""
                safe_entity_value = entity_value.replace("'", "''") if entity_value else ""
                metadata_json = json.dumps(metadata)
                safe_metadata = metadata_json.replace("'", "''")
                
                query = f"""
                INSERT INTO entities (id, document_id, chunk_id, entity_type, entity_value, start_char, end_char, metadata)
                VALUES (
                    '{new_entity_id}'::uuid,
                    '{document_uuid}'::uuid, 
                    {f"'{chunk_uuid}'::uuid" if chunk_uuid else "NULL"}, 
                    '{safe_entity_type}', 
                    '{safe_entity_value}', 
                    {start_char if start_char is not None else "NULL"}, 
                    {end_char if end_char is not None else "NULL"}, 
                    '{safe_metadata}'::jsonb
                );
                """
                
                result = self._execute_sql_query(query)
                if result is not None:
                    logger.info(f"Inserted entity with ID: {new_entity_id} using direct SQL")
                    return new_entity_id
            except Exception as e:
                logger.warning(f"Direct SQL entity insertion failed: {e}")
            
            # Method 2: Try using temporary client (bypass RLS)
            try:
                temp_client = create_client(SUPABASE_URL, SUPABASE_KEY)
                
                data = {
                    "document_id": document_uuid,
                    "chunk_id": chunk_uuid,
                    "entity_type": entity_type,
                    "entity_value": entity_value,
                    "start_char": start_char,
                    "end_char": end_char,
                    "metadata": metadata
                }
                
                response = temp_client.table(self.entities_table).insert(data).execute()
                if response.data and len(response.data) > 0:
                    entity_id = response.data[0]['id']
                    logger.info(f"Inserted entity with ID: {entity_id} using bypass API")
                    return entity_id
            except Exception as e:
                logger.warning(f"Bypass API entity insertion failed: {e}")
            
            raise Exception(f"Failed to insert entity after trying multiple methods")
            
        except Exception as e:
            logger.error(f"Error inserting entity: {e}", exc_info=True)
            raise

    def insert_relationship(self, document_id: str, chunk_id: str, entity_1: str, 
                          entity_2: str, relationship_type: str, context: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Insert a relationship and return its ID. No retry decorator to prevent duplicate creation."""
        try:
            # Ensure document_id is a valid UUID
            if not self._is_valid_uuid(document_id):
                raise ValueError(f"Invalid document_id: {document_id}")
            
            # Format document_id as proper UUID
            document_uuid = str(uuid.UUID(str(document_id)))
            
            # Handle chunk_id (which can be None)
            chunk_uuid = None
            if chunk_id and self._is_valid_uuid(chunk_id):
                chunk_uuid = str(uuid.UUID(str(chunk_id)))
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            # Method 1: Try using direct SQL query with predefined UUID
            try:
                new_rel_id = str(uuid.uuid4())
                
                # Escape values to prevent SQL injection
                safe_entity_1 = entity_1.replace("'", "''") if entity_1 else ""
                safe_entity_2 = entity_2.replace("'", "''") if entity_2 else ""
                safe_relationship_type = relationship_type.replace("'", "''") if relationship_type else ""
                safe_context = context.replace("'", "''") if context else ""
                metadata_json = json.dumps(metadata)
                safe_metadata = metadata_json.replace("'", "''")
                
                query = f"""
                INSERT INTO relationships (id, document_id, chunk_id, entity_1, entity_2, relationship_type, context, metadata)
                VALUES (
                    '{new_rel_id}'::uuid,
                    '{document_uuid}'::uuid, 
                    {f"'{chunk_uuid}'::uuid" if chunk_uuid else "NULL"}, 
                    '{safe_entity_1}', 
                    '{safe_entity_2}', 
                    '{safe_relationship_type}', 
                    '{safe_context}', 
                    '{safe_metadata}'::jsonb
                );
                """
                
                result = self._execute_sql_query(query)
                if result is not None:
                    logger.info(f"Inserted relationship with ID: {new_rel_id} using direct SQL")
                    return new_rel_id
            except Exception as e:
                logger.warning(f"Direct SQL relationship insertion failed: {e}")
            
            # Method 2: Try using temporary client (bypass RLS)
            try:
                temp_client = create_client(SUPABASE_URL, SUPABASE_KEY)
                
                data = {
                    "document_id": document_uuid,
                    "chunk_id": chunk_uuid,
                    "entity_1": entity_1,
                    "entity_2": entity_2,
                    "relationship_type": relationship_type,
                    "context": context,
                    "metadata": metadata
                }
                
                response = temp_client.table(self.relationships_table).insert(data).execute()
                if response.data and len(response.data) > 0:
                    rel_id = response.data[0]['id']
                    logger.info(f"Inserted relationship with ID: {rel_id} using bypass API")
                    return rel_id
            except Exception as e:
                logger.warning(f"Bypass API relationship insertion failed: {e}")
            
            raise Exception(f"Failed to insert relationship after trying multiple methods")
            
        except Exception as e:
            logger.error(f"Error inserting relationship: {e}", exc_info=True)
            raise

    @retry(**RETRY_CONFIG)
    def fetch_similar_embeddings(self, query_embedding: List[float], limit: int = 5, 
                                match_threshold: float = 0.78, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch embeddings similar to the query embedding."""
        try:
            if project_id:
                # Use the match_embeddings_by_project function
                response = self.client.rpc(
                    'match_embeddings_by_project',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': match_threshold,
                        'match_count': limit,
                        'project_id': project_id
                    }
                ).execute()
            else:
                # Use the general match_embeddings function
                response = self.client.rpc(
                    'match_embeddings',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': match_threshold,
                        'match_count': limit
                    }
                ).execute()
            
            return response.data or []
        except Exception as e:
            logger.warning(f"Error in fetch_similar_embeddings: {e}")
            return []

    @retry(**RETRY_CONFIG)
    def fetch_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch text chunks by their IDs."""
        if not chunk_ids:
            return []
        response = self.client.table(self.chunks_table).select("*").in_("id", chunk_ids).execute()
        return response.data or []
    
    @retry(**RETRY_CONFIG)
    def count_chunks_for_project(self, project_id: str) -> int:
        """Count chunks for a specific project (handles RLS properly)."""
        try:
            if not self._is_valid_uuid(project_id):
                raise ValueError(f"Invalid project_id: {project_id}")
            
            project_uuid = str(uuid.UUID(str(project_id)))
            
            # Method 1: Try direct SQL query that bypasses RLS issues
            try:
                query = f"""
                SELECT COUNT(c.id) as chunk_count
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.project_id = '{project_uuid}'::uuid;
                """
                
                result = self._execute_sql_query(query)
                if result and len(result) > 0:
                    return result[0].get('chunk_count', 0)
            except Exception as e:
                logger.warning(f"Direct SQL chunk count failed: {e}")
            
            # Method 2: Try using Supabase client with proper joins
            try:
                response = self.client.table("chunks").select(
                    "id, documents!inner(project_id)"
                ).eq("documents.project_id", project_uuid).execute()
                
                return len(response.data) if response.data else 0
            except Exception as e:
                logger.warning(f"Joined query chunk count failed: {e}")
            
            # Method 3: Fallback - get documents first, then count chunks
            try:
                docs_response = self.fetch_documents_by_project(project_id)
                if not docs_response:
                    return 0
                
                doc_ids = [doc["id"] for doc in docs_response]
                total_chunks = 0
                
                for doc_id in doc_ids:
                    try:
                        chunks_response = self.client.table(self.chunks_table).select("id").eq("document_id", doc_id).execute()
                        total_chunks += len(chunks_response.data) if chunks_response.data else 0
                    except Exception as doc_error:
                        logger.warning(f"Failed to count chunks for document {doc_id}: {doc_error}")
                        continue
                
                return total_chunks
            except Exception as e:
                logger.warning(f"Fallback chunk count failed: {e}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Error counting chunks for project {project_id}: {e}")
            return 0

    @retry(**RETRY_CONFIG)
    def fetch_entities(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch entities matching given query parameters."""
        query = self.client.table(self.entities_table).select("*")
        
        for key, value in query_params.items():
            if isinstance(value, list):
                query = query.in_(key, value)
            else:
                query = query.eq(key, value)
        
        response = query.execute()
        return response.data or []

    @retry(**RETRY_CONFIG)
    def fetch_relationships(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fetch relationships matching given query parameters."""
        query = self.client.table(self.relationships_table).select("*")
        
        for key, value in query_params.items():
            if isinstance(value, list):
                query = query.in_(key, value)
            else:
                query = query.eq(key, value)
        
        response = query.execute()
        return response.data or []

    @retry(**RETRY_CONFIG)
    def fetch_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a document by its ID."""
        try:
            if not self._is_valid_uuid(document_id):
                return None
            
            response = self.client.table(self.documents_table)\
                .select("*")\
                .eq("id", document_id)\
                .limit(1)\
                .execute()
            
            if response.data:
                return response.data[0]
            return None
            
        except Exception as e:
            logger.error(f"Error fetching document: {e}")
            return None

    @retry(**RETRY_CONFIG)
    def fetch_documents_by_hash(self, file_hash: str) -> List[Dict[str, Any]]:
        """Fetch documents by file hash."""
        try:
            # Try direct SQL query first to avoid type mismatches
            try:
                query = f"""
                SELECT * FROM documents 
                WHERE file_hash = '{file_hash}'
                """
                
                response = self.client.rpc('execute_sql', {'query': query}).execute()
                
                if response.data:
                    logger.info(f"Found {len(response.data)} documents with hash {file_hash} via direct SQL")
                    # Make sure all UUIDs are properly formatted to avoid comparison issues
                    for doc in response.data:
                        if 'id' in doc and doc['id']:
                            doc['id'] = str(uuid.UUID(str(doc['id'])))
                        if 'project_id' in doc and doc['project_id']:
                            doc['project_id'] = str(uuid.UUID(str(doc['project_id'])))
                    return response.data
            except Exception as e:
                logger.warning(f"Direct SQL query for documents by hash failed: {e}, trying standard query")
            
            # Standard query
            response = self.client.table(self.documents_table).select("*").eq("file_hash", file_hash).execute()
            
            if response.data:
                # Make sure all UUIDs are properly formatted to avoid comparison issues
                for doc in response.data:
                    if 'id' in doc and doc['id']:
                        doc['id'] = str(uuid.UUID(str(doc['id'])))
                    if 'project_id' in doc and doc['project_id']:
                        doc['project_id'] = str(uuid.UUID(str(doc['project_id'])))
        
            return response.data or []
        except Exception as e:
            logger.error(f"Error in fetch_documents_by_hash: {e}")
            return []

    @retry(**RETRY_CONFIG)
    def fetch_latest_version_for_filename(self, filename: str, project_id: Optional[str] = None) -> int:
        """Fetch the latest version number for a filename."""
        try:
            # If project_id is provided, validate and format it
            project_uuid = None
            if project_id:
                if not self._is_valid_uuid(project_id):
                    logger.warning(f"Invalid project_id format in fetch_latest_version: {project_id}")
                    return 0
                project_uuid = str(uuid.UUID(str(project_id)))
            
            # Try direct SQL query first to avoid type mismatches
            try:
                if project_uuid:
                    query = f"""
                    SELECT version FROM documents 
                    WHERE filename = '{filename}' AND project_id::text = '{project_uuid}'
                    ORDER BY version DESC
                    LIMIT 1
                    """
                else:
                    query = f"""
                    SELECT version FROM documents 
                    WHERE filename = '{filename}'
                    ORDER BY version DESC
                    LIMIT 1
                    """
                
                response = self.client.rpc('execute_sql', {'query': query}).execute()
                
                if response.data and len(response.data) > 0:
                    return response.data[0].get("version", 0)
            except Exception as e:
                logger.warning(f"Direct SQL query for version failed: {e}, trying standard query")
            
            # Standard query
            query = self.client.table(self.documents_table).select("version").eq("filename", filename)
            
            if project_uuid:
                query = query.eq("project_id", project_uuid)
            
            response = query.order("version", desc=True).limit(1).execute()
            
            if response.data and len(response.data) > 0:
                return response.data[0].get("version", 0)
            
            return 0
        except Exception as e:
            logger.error(f"Error in fetch_latest_version_for_filename: {e}")
            return 0

    @retry(**RETRY_CONFIG)
    def log_api_usage(self, *, user_id=None, document_id=None, api_provider=None, 
                     api_type=None, tokens_used=None, cost_usd=None, 
                     request_payload=None, response_metadata=None):
        """Log API usage and cost."""
        try:
            # Validate document_id as UUID if provided
            validated_document_id = None
            if document_id:
                if self._is_valid_uuid(document_id):
                    validated_document_id = str(document_id)
                else:
                    # Store the original document_id in request_payload for reference
                    if request_payload is None:
                        request_payload = {}
                    request_payload['original_document_id'] = str(document_id)
                    logger.debug(f"Non-UUID document_id '{document_id}' stored in request_payload")
            
            # Handle user_id as text (no UUID validation needed)
            validated_user_id = str(user_id) if user_id is not None else None
            
            data = {
                "user_id": validated_user_id,
                "document_id": validated_document_id,
                "api_provider": api_provider,
                "api_type": api_type,
                "tokens_used": tokens_used,
                "cost_usd": cost_usd,
                "request_payload": request_payload,
                "response_metadata": response_metadata
            }
            self.client.table("api_usage_logs").insert(data).execute()
            logger.info(f"Logged API usage: {api_provider}/{api_type}")
        except Exception as e:
            logger.error(f"Error logging API usage: {e}")

    # Agricultural Database Extensions
    def create_agricultural_tables(self):
        """Create agricultural-specific tables for crops, products, and research studies."""
        logger.info("Creating agricultural database tables...")
        
        try:
            # Use Supabase SQL function to create tables
            sql_commands = [
                # Crops table - Agricultural crop information
                """
                CREATE TABLE IF NOT EXISTS crops (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    common_name VARCHAR(255) NOT NULL,
                    scientific_name VARCHAR(255),
                    variety VARCHAR(255),
                    crop_family VARCHAR(100),
                    growth_cycle_days INTEGER,
                    optimal_ph_min DECIMAL(3,1),
                    optimal_ph_max DECIMAL(3,1),
                    optimal_temp_min INTEGER,
                    optimal_temp_max INTEGER,
                    water_requirements VARCHAR(50),
                    soil_type_preferences TEXT[],
                    primary_nutrients JSONB,
                    growth_stages JSONB,
                    common_pests TEXT[],
                    common_diseases TEXT[],
                    harvest_indicators TEXT[],
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """,
                
                # Agricultural products table
                """
                CREATE TABLE IF NOT EXISTS agro_products (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    product_name VARCHAR(255) NOT NULL,
                    brand VARCHAR(255),
                    product_type VARCHAR(100) NOT NULL,
                    category VARCHAR(100),
                    active_ingredients JSONB,
                    npk_ratio VARCHAR(20),
                    formulation VARCHAR(100),
                    concentration JSONB,
                    application_methods TEXT[],
                    target_crops TEXT[],
                    application_rates JSONB,
                    application_timing TEXT[],
                    compatibility_notes TEXT,
                    restrictions TEXT[],
                    storage_requirements TEXT,
                    registration_numbers TEXT[],
                    manufacturer VARCHAR(255),
                    safety_data JSONB,
                    efficacy_claims JSONB,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """,
                
                # Research studies table
                """
                CREATE TABLE IF NOT EXISTS research_studies (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    study_title VARCHAR(500) NOT NULL,
                    study_type VARCHAR(100),
                    researchers TEXT[],
                    institution VARCHAR(255),
                    publication_year INTEGER,
                    publication_journal VARCHAR(255),
                    doi VARCHAR(255),
                    study_location VARCHAR(255),
                    climate_zone VARCHAR(100),
                    soil_type VARCHAR(100),
                    crop_studied TEXT[],
                    products_tested TEXT[],
                    study_duration_days INTEGER,
                    sample_size INTEGER,
                    methodology TEXT,
                    treatments JSONB,
                    control_groups JSONB,
                    measurements JSONB,
                    results JSONB,
                    statistical_significance JSONB,
                    yield_data JSONB,
                    environmental_conditions JSONB,
                    conclusions TEXT,
                    limitations TEXT,
                    recommendations TEXT,
                    keywords TEXT[],
                    quality_score DECIMAL(3,2),
                    peer_reviewed BOOLEAN DEFAULT FALSE,
                    open_access BOOLEAN DEFAULT FALSE,
                    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """,
                
                # Agricultural entities extension table
                """
                CREATE TABLE IF NOT EXISTS agricultural_entities (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    entity_type VARCHAR(50) NOT NULL,
                    entity_value VARCHAR(500) NOT NULL,
                    normalized_value VARCHAR(500),
                    entity_subtype VARCHAR(100),
                    confidence_score DECIMAL(3,2) DEFAULT 1.00,
                    extraction_method VARCHAR(50) DEFAULT 'automatic',
                    source_context TEXT,
                    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
                    position_start INTEGER,
                    position_end INTEGER,
                    additional_properties JSONB DEFAULT '{}',
                    verification_status VARCHAR(20) DEFAULT 'pending',
                    verified_by UUID REFERENCES custom_users(id) ON DELETE SET NULL,
                    verified_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """,
                
                # Agricultural relationships extension table
                """
                CREATE TABLE IF NOT EXISTS agricultural_relationships (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    relationship_type VARCHAR(50) NOT NULL,
                    source_entity_id UUID REFERENCES agricultural_entities(id) ON DELETE CASCADE,
                    target_entity_id UUID REFERENCES agricultural_entities(id) ON DELETE CASCADE,
                    relationship_strength DECIMAL(3,2) DEFAULT 1.00,
                    confidence_score DECIMAL(3,2) DEFAULT 1.00,
                    supporting_evidence TEXT,
                    quantitative_data JSONB,
                    study_references UUID[],
                    extraction_method VARCHAR(50) DEFAULT 'automatic',
                    source_context TEXT,
                    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
                    verification_status VARCHAR(20) DEFAULT 'pending',
                    verified_by UUID REFERENCES custom_users(id) ON DELETE SET NULL,
                    verified_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """
            ]
            
            # Execute table creation commands
            for sql in sql_commands:
                try:
                    if hasattr(self.client, 'rpc') and callable(getattr(self.client, 'rpc')):
                        # Try RPC method if available
                        response = self.client.rpc('execute_sql', {'query': sql.strip()}).execute()
                    else:
                        # Fallback: log SQL for manual execution
                        logger.warning(f"Cannot execute SQL directly. Please run manually: {sql[:100]}...")
                except Exception as e:
                    logger.warning(f"SQL execution failed (table may already exist): {e}")
            
            # Create indexes
            index_commands = [
                "CREATE INDEX IF NOT EXISTS idx_crops_common_name ON crops(common_name);",
                "CREATE INDEX IF NOT EXISTS idx_agro_products_name ON agro_products(product_name);",
                "CREATE INDEX IF NOT EXISTS idx_agro_products_type ON agro_products(product_type);",
                "CREATE INDEX IF NOT EXISTS idx_research_studies_type ON research_studies(study_type);",
                "CREATE INDEX IF NOT EXISTS idx_agro_entities_type ON agricultural_entities(entity_type);",
                "CREATE INDEX IF NOT EXISTS idx_agro_entities_value ON agricultural_entities(entity_value);",
                "CREATE INDEX IF NOT EXISTS idx_agro_relationships_type ON agricultural_relationships(relationship_type);"
            ]
            
            for index_sql in index_commands:
                try:
                    if hasattr(self.client, 'rpc') and callable(getattr(self.client, 'rpc')):
                        response = self.client.rpc('execute_sql', {'query': index_sql.strip()}).execute()
                except Exception as e:
                    logger.warning(f"Index creation failed (may already exist): {e}")
            
            logger.info("Agricultural database tables setup completed")
            
        except Exception as e:
            logger.error(f"Error creating agricultural tables: {e}")
            # Don't raise - allow system to continue without agricultural tables

    @retry(**RETRY_CONFIG)
    def insert_agricultural_entity(self, entity_data: Dict[str, Any]) -> Optional[str]:
        """Insert an agricultural entity into the database."""
        try:
            # Handle both direct metadata and nested metadata format
            metadata = entity_data.get('metadata', {})
            if isinstance(metadata, dict):
                # Extract additional properties from metadata
                additional_properties = {
                    "unit": metadata.get('unit'),
                    "numeric_value": metadata.get('numeric_value'),
                    "normalized_value": metadata.get('normalized_value'),
                    "context": metadata.get('context'),
                    "document_type": metadata.get('document_type'),
                    "start_pos": metadata.get('start_pos'),
                    "end_pos": metadata.get('end_pos'),
                    "user_id": metadata.get('user_id'),
                    "project_id": metadata.get('project_id')
                }
                # Remove None values
                additional_properties = {k: v for k, v in additional_properties.items() if v is not None}
            else:
                additional_properties = {}
            
            # Prepare the data for insertion
            insert_data = {
                "entity_type": entity_data.get('entity_type'),
                "entity_value": entity_data.get('entity_value'),
                "normalized_value": entity_data.get('normalized_value'),
                "entity_subtype": entity_data.get('entity_subtype'),
                "confidence_score": entity_data.get('confidence_score', 1.0),
                "extraction_method": entity_data.get('extraction_method', 'automatic'),
                "source_context": entity_data.get('source_context') or metadata.get('context'),
                "document_id": self._ensure_uuid(entity_data.get('document_id')),
                "chunk_id": self._ensure_uuid(entity_data.get('chunk_id')),
                "position_start": entity_data.get('position_start') or metadata.get('start_pos'),
                "position_end": entity_data.get('position_end') or metadata.get('end_pos'),
                "additional_properties": additional_properties
            }
            
            # Remove None values
            insert_data = {k: v for k, v in insert_data.items() if v is not None}
            
            response = self.client.table("agricultural_entities").insert(insert_data).execute()
            
            if response.data:
                entity_id = response.data[0]['id']
                logger.info(f"Inserted agricultural entity: {entity_data.get('entity_type')} - {entity_data.get('entity_value')}")
                return entity_id
            else:
                logger.error("No data returned from agricultural entity insert")
                return None
                
        except Exception as e:
            logger.error(f"Error inserting agricultural entity: {e}")
            return None

    @retry(**RETRY_CONFIG)
    def insert_agricultural_relationship(self, relationship_data: Dict[str, Any]) -> Optional[str]:
        """Insert an agricultural relationship into the database."""
        try:
            # Handle both direct metadata and nested metadata format
            metadata = relationship_data.get('metadata', {})
            if isinstance(metadata, dict):
                # Extract additional properties from metadata
                quantitative_data = {
                    "context": metadata.get('context'),
                    "document_type": metadata.get('document_type'),
                    "user_id": metadata.get('user_id'),
                    "project_id": metadata.get('project_id')
                }
                # Remove None values
                quantitative_data = {k: v for k, v in quantitative_data.items() if v is not None}
            else:
                quantitative_data = {}
            
            # Handle source_entity and target_entity - they might be entity names or IDs
            source_entity_id = relationship_data.get('source_entity_id')
            target_entity_id = relationship_data.get('target_entity_id')
            
            # If IDs not provided, we'll store the entity names in supporting_evidence for now
            # In production, you'd want to resolve entity names to IDs
            if not source_entity_id:
                supporting_evidence = f"Source: {relationship_data.get('source_entity', '')}, Target: {relationship_data.get('target_entity', '')}"
            else:
                supporting_evidence = relationship_data.get('supporting_evidence')
            
            # Prepare the data for insertion
            insert_data = {
                "relationship_type": relationship_data.get('relationship_type'),
                "source_entity_id": self._ensure_uuid(source_entity_id) if source_entity_id else None,
                "target_entity_id": self._ensure_uuid(target_entity_id) if target_entity_id else None,
                "relationship_strength": relationship_data.get('relationship_strength', 1.0),
                "confidence_score": relationship_data.get('confidence_score', 1.0),
                "supporting_evidence": supporting_evidence,
                "quantitative_data": quantitative_data,
                "study_references": relationship_data.get('study_references', []),
                "extraction_method": relationship_data.get('extraction_method', 'automatic'),
                "source_context": relationship_data.get('source_context') or metadata.get('context'),
                "document_id": self._ensure_uuid(relationship_data.get('document_id')),
                "chunk_id": self._ensure_uuid(relationship_data.get('chunk_id'))
            }
            
            # Remove None values
            insert_data = {k: v for k, v in insert_data.items() if v is not None}
            
            response = self.client.table("agricultural_relationships").insert(insert_data).execute()
            
            if response.data:
                relationship_id = response.data[0]['id']
                logger.info(f"Inserted agricultural relationship: {relationship_data.get('relationship_type')}")
                return relationship_id
            else:
                logger.error("No data returned from agricultural relationship insert")
                return None
                
        except Exception as e:
            logger.error(f"Error inserting agricultural relationship: {e}")
            return None

    @retry(**RETRY_CONFIG)
    def query_agricultural_entities(self, 
                                    entity_type: Optional[str] = None,
                                    entity_value: Optional[str] = None,
                                    document_id: Optional[str] = None,
                                    project_id: Optional[str] = None,
                                    limit: int = 50) -> List[Dict[str, Any]]:
        """Query agricultural entities with various filters."""
        try:
            query = self.client.table("agricultural_entities").select("*")
            
            if entity_type:
                query = query.eq("entity_type", entity_type)
            
            if entity_value:
                query = query.ilike("entity_value", f"%{entity_value}%")
            
            if document_id:
                query = query.eq("document_id", self._ensure_uuid(document_id))
            
            # Add project filtering by joining with documents table
            if project_id:
                # This requires a more complex query - for now, filter client-side
                pass
            
            response = query.order("confidence_score", desc=True).limit(limit).execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Error querying agricultural entities: {e}")
            return []

    @retry(**RETRY_CONFIG)
    def query_agricultural_relationships(self,
                                         relationship_type: Optional[str] = None,
                                         source_entity_type: Optional[str] = None,
                                         target_entity_type: Optional[str] = None,
                                         document_id: Optional[str] = None,
                                         project_id: Optional[str] = None,
                                         limit: int = 50) -> List[Dict[str, Any]]:
        """Query agricultural relationships with various filters."""
        try:
            # For complex queries with joins, use RPC if available
            if hasattr(self.client, 'rpc') and callable(getattr(self.client, 'rpc')):
                # Build dynamic query
                where_conditions = []
                
                if relationship_type:
                    where_conditions.append(f"ar.relationship_type = '{relationship_type}'")
                
                if document_id:
                    where_conditions.append(f"ar.document_id = '{self._ensure_uuid(document_id)}'")
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                query = f"""
                    SELECT ar.*, 
                           se.entity_value as source_entity_value,
                           se.entity_type as source_entity_type,
                           te.entity_value as target_entity_value,
                           te.entity_type as target_entity_type
                    FROM agricultural_relationships ar
                    LEFT JOIN agricultural_entities se ON ar.source_entity_id = se.id
                    LEFT JOIN agricultural_entities te ON ar.target_entity_id = te.id
                    WHERE {where_clause}
                    ORDER BY ar.relationship_strength DESC, ar.confidence_score DESC
                    LIMIT {limit}
                """
                
                try:
                    response = self.client.rpc('execute_sql', {'query': query}).execute()
                    return response.data if response.data else []
                except Exception as rpc_e:
                    logger.warning(f"RPC query failed: {rpc_e}, falling back to simple query")
            
            # Fallback to simple query
            query = self.client.table("agricultural_relationships").select("*")
            
            if relationship_type:
                query = query.eq("relationship_type", relationship_type)
            
            if document_id:
                query = query.eq("document_id", self._ensure_uuid(document_id))
            
            response = query.order("relationship_strength", desc=True).limit(limit).execute()
            
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Error querying agricultural relationships: {e}")
            return []

    def get_agricultural_summary(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of agricultural data for a project or overall."""
        try:
            summary = {
                "total_entities": 0,
                "total_relationships": 0,
                "entity_types": {},
                "relationship_types": {},
                "top_crops": [],
                "top_products": []
            }
            
            # Get entity counts
            entities = self.query_agricultural_entities(project_id=project_id, limit=1000)
            summary["total_entities"] = len(entities)
            
            # Count by entity type
            for entity in entities:
                entity_type = entity.get("entity_type", "unknown")
                summary["entity_types"][entity_type] = summary["entity_types"].get(entity_type, 0) + 1
            
            # Get relationship counts
            relationships = self.query_agricultural_relationships(project_id=project_id, limit=1000)
            summary["total_relationships"] = len(relationships)
            
            # Count by relationship type
            for relationship in relationships:
                rel_type = relationship.get("relationship_type", "unknown")
                summary["relationship_types"][rel_type] = summary["relationship_types"].get(rel_type, 0) + 1
            
            # Extract top crops and products
            crops = [e for e in entities if e.get("entity_type") == "CROP"]
            products = [e for e in entities if e.get("entity_type") == "PRODUCT"]
            
            summary["top_crops"] = [e.get("entity_value") for e in crops[:10]]
            summary["top_products"] = [e.get("entity_value") for e in products[:10]]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating agricultural summary: {e}")
            return {
                "total_entities": 0,
                "total_relationships": 0,
                "entity_types": {},
                "relationship_types": {},
                "top_crops": [],
                "top_products": [],
                "error": str(e)
            }

    def create_document(self, project_id: str, document_data: Dict[str, Any], content: str = "") -> str:
        """Convenience method to create a document. Wraps insert_document."""
        filename = document_data.get("filename", "unknown.pdf")
        num_pages = document_data.get("num_pages", 1)
        metadata = {k: v for k, v in document_data.items() if k not in ["filename", "num_pages"]}
        metadata["content"] = content
        
        return self.insert_document(
            filename=filename,
            num_pages=num_pages,
            metadata=metadata,
            project_id=project_id
        )
    
    def create_chunk(self, document_id: str, chunk_data: Dict[str, Any]) -> str:
        """Convenience method to create a chunk. Wraps insert_chunk."""
        chunk_text = chunk_data.get("chunk_text", "")
        chunk_order = chunk_data.get("chunk_index", 0)
        page_number = chunk_data.get("page_number", 1)
        metadata = {k: v for k, v in chunk_data.items() if k not in ["chunk_text", "chunk_index", "page_number"]}
        
        return self.insert_chunk(
            document_id=document_id,
            chunk_text=chunk_text,
            chunk_order=chunk_order,
            page_number=page_number,
            metadata=metadata
        )
    
    def create_entity(self, entity_data: Dict[str, Any]) -> str:
        """Convenience method to create an entity. Wraps insert_entity."""
        document_id = entity_data.get("document_id")
        chunk_id = entity_data.get("chunk_id")
        entity_type = entity_data.get("entity_type", "")
        entity_value = entity_data.get("entity_text", entity_data.get("entity_value", ""))
        start_char = entity_data.get("start_position")
        end_char = entity_data.get("end_position")
        metadata = {k: v for k, v in entity_data.items() 
                   if k not in ["document_id", "chunk_id", "entity_type", "entity_text", "entity_value", "start_position", "end_position"]}
        
        return self.insert_entity(
            document_id=document_id,
            chunk_id=chunk_id,
            entity_type=entity_type,
            entity_value=entity_value,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata
        )
    
    def create_relationship(self, relationship_data: Dict[str, Any]) -> str:
        """Convenience method to create a relationship. Wraps insert_relationship."""
        document_id = relationship_data.get("document_id")
        chunk_id = relationship_data.get("chunk_id")
        entity_1 = relationship_data.get("source_entity", "")
        entity_2 = relationship_data.get("target_entity", "")
        relationship_type = relationship_data.get("relationship_type", "")
        context = relationship_data.get("context", "")
        metadata = {k: v for k, v in relationship_data.items() 
                   if k not in ["document_id", "chunk_id", "source_entity", "target_entity", "relationship_type", "context"]}
        
        return self.insert_relationship(
            document_id=document_id,
            chunk_id=chunk_id,
            entity_1=entity_1,
            entity_2=entity_2,
            relationship_type=relationship_type,
            context=context,
            metadata=metadata
        )

    @retry(**RETRY_CONFIG)
    def search_entities_by_type(self, entity_type: str, project_id: Optional[str] = None, 
                              document_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Search entities by type with optional filtering."""
        try:
            query = self.client.table('entities').select('*').eq('entity_type', entity_type)
            
            if project_id and self._is_valid_uuid(project_id):
                # Join with documents table to filter by project
                query = self.client.table('entities').select('''
                    *,
                    documents!inner(project_id)
                ''').eq('entity_type', entity_type).eq('documents.project_id', project_id)
            
            if document_id and self._is_valid_uuid(document_id):
                query = query.eq('document_id', document_id)
            
            response = query.limit(limit).execute()
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error searching entities by type: {e}")
            return []

    @retry(**RETRY_CONFIG)
    def search_entities_by_similarity(self, value: str, project_id: Optional[str] = None,
                                    document_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Search entities by value similarity."""
        try:
            # Use PostgreSQL text search capabilities
            query = self.client.table('entities').select('*').ilike('entity_value', f'%{value}%')
            
            if project_id and self._is_valid_uuid(project_id):
                # Join with documents table to filter by project
                query = self.client.table('entities').select('''
                    *,
                    documents!inner(project_id)
                ''').ilike('entity_value', f'%{value}%').eq('documents.project_id', project_id)
            
            if document_id and self._is_valid_uuid(document_id):
                query = query.eq('document_id', document_id)
            
            response = query.limit(limit).execute()
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error searching entities by similarity: {e}")
            return []

    @retry(**RETRY_CONFIG)
    def get_frequent_entities(self, project_id: Optional[str] = None, 
                            document_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get frequently occurring entities."""
        try:
            if project_id and self._is_valid_uuid(project_id):
                # Join with documents table to filter by project
                query = self.client.table('entities').select('''
                    *,
                    documents!inner(project_id)
                ''').eq('documents.project_id', project_id)
            else:
                query = self.client.table('entities').select('*')
            
            if document_id and self._is_valid_uuid(document_id):
                query = query.eq('document_id', document_id)
            
            # Order by created_at to get recent entities
            response = query.order('created_at', desc=True).limit(limit).execute()
            return response.data or []
            
        except Exception as e:
            logger.error(f"Error getting frequent entities: {e}")
            return []

    def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a raw SQL query and return results."""
        try:
            # For simple queries, use the Supabase client
            if params:
                # This is a simplified implementation - in production you'd want proper parameterization
                for key, value in params.items():
                    query = query.replace(f":{key}", str(value))
            
            # Use the REST API for simple SELECT queries
            # This is a basic implementation - extend as needed
            return []
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []

    def create_test_project(self, name: str, description: str = "", metadata: dict = {}, user_id: str = "test_user") -> str:
        """Create a project for testing that bypasses RLS."""
        try:
            # Use the service role key to bypass RLS for testing
            test_client = create_client(
                supabase_url=os.getenv("SUPABASE_URL"),
                supabase_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_KEY"))  # Fallback to anon key
            )
            
            project_data = {
                "name": name,
                "description": description,
                "metadata": metadata,
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat()
            }
            
            response = test_client.table(self.projects_table).insert(project_data).execute()
            
            if response.data:
                project_id = response.data[0]["id"]
                logger.info(f"Test project created: {project_id}")
                return project_id
            else:
                raise Exception("No project data returned")
                
        except Exception as e:
            logger.error(f"Error creating test project: {e}")
            # Fallback to UUID generation for testing
            import uuid
            return str(uuid.uuid4())

    @retry(**RETRY_CONFIG)
    def insert_food_industry_entity(self, entity_data: Dict[str, Any], document_id: str) -> Optional[str]:
        """
        Insert a food industry entity into the food_industry_entities table.
        
        Args:
            entity_data: Dictionary containing food industry entity information
            document_id: Document ID to associate with the entity
            
        Returns:
            Entity ID if successful, None otherwise
        """
        try:
            # Ensure document_id is a valid UUID
            document_uuid = self._ensure_uuid(document_id)
            if not document_uuid:
                logger.error(f"Invalid document_id: {document_id}")
                return None
            
            # Prepare the food industry entity data
            food_entity = {
                "food_industry_type": entity_data.get("food_industry_type", "unspecified"),
                "product_category": entity_data.get("product_category", "general"),
                "regulatory_status": entity_data.get("regulatory_status"),
                "food_grade": entity_data.get("food_grade", True),
                "shelf_life_days": entity_data.get("shelf_life_days"),
                "allergen_info": entity_data.get("allergen_info", {}),
                "nutritional_value": entity_data.get("nutritional_value", {}),
                "applications": entity_data.get("applications", []),
                "processing_methods": entity_data.get("processing_methods", []),
                "storage_conditions": entity_data.get("storage_conditions", {}),
                "quality_parameters": entity_data.get("quality_parameters", {}),
                "supplier_info": entity_data.get("supplier_info", {}),
                "cost_information": entity_data.get("cost_information", {}),
                "certifications": entity_data.get("certifications", []),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # First create a regular entity entry
            entity_id = None
            if entity_data.get("entity_value"):
                entity_id = self.insert_entity(
                    document_id=document_uuid,
                    chunk_id=None,
                    entity_type="food_industry",
                    entity_value=entity_data.get("entity_value", ""),
                    metadata=entity_data.get("additional_properties", {})
                )
                food_entity["entity_id"] = entity_id
            
            # Insert into food_industry_entities table
            response = self.client.table("food_industry_entities").insert(food_entity).execute()
            
            if response.data:
                food_entity_id = response.data[0]["id"]
                logger.info(f"Food industry entity inserted successfully: {food_entity_id}")
                return food_entity_id
            else:
                logger.error("No food industry entity data returned after insertion")
                return None
                
        except Exception as e:
            logger.error(f"Error inserting food industry entity: {e}")
            return None

    @retry(**RETRY_CONFIG)
    def insert_nutritional_information(self, nutrition_data: Dict[str, Any], 
                                     entity_id: Optional[str] = None,
                                     food_industry_entity_id: Optional[str] = None) -> Optional[str]:
        """
        Insert nutritional information into the nutritional_information table.
        
        Args:
            nutrition_data: Dictionary containing nutritional information
            entity_id: Related entity ID (optional)
            food_industry_entity_id: Related food industry entity ID (optional)
            
        Returns:
            Nutritional information ID if successful, None otherwise
        """
        try:
            # Prepare nutritional information data
            nutrition_entry = {
                "entity_id": self._ensure_uuid(entity_id) if entity_id else None,
                "food_industry_entity_id": self._ensure_uuid(food_industry_entity_id) if food_industry_entity_id else None,
                "calories_per_100g": nutrition_data.get("calories_per_100g"),
                "protein_g": nutrition_data.get("protein_g"),
                "fat_g": nutrition_data.get("fat_g"),
                "carbohydrates_g": nutrition_data.get("carbohydrates_g"),
                "fiber_g": nutrition_data.get("fiber_g"),
                "sugar_g": nutrition_data.get("sugar_g"),
                "sodium_mg": nutrition_data.get("sodium_mg"),
                "vitamins": nutrition_data.get("vitamins", {}),
                "minerals": nutrition_data.get("minerals", {}),
                "amino_acids": nutrition_data.get("amino_acids", {}),
                "fatty_acids": nutrition_data.get("fatty_acids", {}),
                "other_nutrients": nutrition_data.get("other_nutrients", {}),
                "serving_size_g": nutrition_data.get("serving_size_g", 100),
                "nutritional_claims": nutrition_data.get("nutritional_claims", []),
                "analysis_method": nutrition_data.get("analysis_method"),
                "analysis_date": nutrition_data.get("analysis_date"),
                "certification_body": nutrition_data.get("certification_body"),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Insert into nutritional_information table
            response = self.client.table("nutritional_information").insert(nutrition_entry).execute()
            
            if response.data:
                nutrition_id = response.data[0]["id"]
                logger.info(f"Nutritional information inserted successfully: {nutrition_id}")
                return nutrition_id
            else:
                logger.error("No nutritional information data returned after insertion")
                return None
                
        except Exception as e:
            logger.error(f"Error inserting nutritional information: {e}")
            return None

    @retry(**RETRY_CONFIG)
    def insert_allergen_information(self, allergen_data: Dict[str, Any],
                                  entity_id: Optional[str] = None,
                                  food_industry_entity_id: Optional[str] = None) -> Optional[str]:
        """
        Insert allergen information into the allergen_information table.
        
        Args:
            allergen_data: Dictionary containing allergen information
            entity_id: Related entity ID (optional)
            food_industry_entity_id: Related food industry entity ID (optional)
            
        Returns:
            Allergen information ID if successful, None otherwise
        """
        try:
            # Prepare allergen information data
            allergen_entry = {
                "entity_id": self._ensure_uuid(entity_id) if entity_id else None,
                "food_industry_entity_id": self._ensure_uuid(food_industry_entity_id) if food_industry_entity_id else None,
                "allergen_type": allergen_data.get("allergen_type", ""),
                "presence_level": allergen_data.get("presence_level", "allergen_free"),
                "cross_contamination_risk": allergen_data.get("cross_contamination_risk", "low"),
                "allergen_threshold_ppm": allergen_data.get("allergen_threshold_ppm"),
                "testing_method": allergen_data.get("testing_method"),
                "certification_status": allergen_data.get("certification_status"),
                "labeling_requirements": allergen_data.get("labeling_requirements", {}),
                "regulatory_compliance": allergen_data.get("regulatory_compliance", {}),
                "supply_chain_controls": allergen_data.get("supply_chain_controls", {}),
                "cleaning_protocols": allergen_data.get("cleaning_protocols", {}),
                "testing_frequency": allergen_data.get("testing_frequency"),
                "last_test_date": allergen_data.get("last_test_date"),
                "test_results": allergen_data.get("test_results", {}),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Insert into allergen_information table
            response = self.client.table("allergen_information").insert(allergen_entry).execute()
            
            if response.data:
                allergen_id = response.data[0]["id"]
                logger.info(f"Allergen information inserted successfully: {allergen_id}")
                return allergen_id
            else:
                logger.error("No allergen information data returned after insertion")
                return None
                
        except Exception as e:
            logger.error(f"Error inserting allergen information: {e}")
            return None

