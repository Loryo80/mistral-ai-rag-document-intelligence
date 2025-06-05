"""
Simple Query Processor for Basic RAG Implementation
Following the basic RAG pattern from Mistral AI documentation
"""

import os
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from mistralai import Mistral

logger = logging.getLogger(__name__)

class SimpleQueryProcessor:
    """Simple RAG implementation using Mistral AI following their basic guide."""
    
    def __init__(self, db_instance):
        """Initialize with database connection and Mistral client."""
        self.db = db_instance
        
        # Initialize Mistral client
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        
        self.client = Mistral(api_key=api_key)
        
        # Simple chunk size for splitting (if needed)
        self.chunk_size = 2048
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text using Mistral's embedding model."""
        try:
            embeddings_response = self.client.embeddings.create(
                model="mistral-embed",
                inputs=[text]
            )
            return embeddings_response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def retrieve_similar_chunks(self, query: str, project_id: Optional[str] = None, 
                               limit: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Retrieve similar chunks from the database using vector similarity.
        
        Args:
            query: User's question
            project_id: Optional project filter
            limit: Number of chunks to retrieve
            threshold: Similarity threshold
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Step 1: Generate embedding for the query
            logger.info(f"Generating embedding for query: {query}")
            query_embedding = self.get_text_embedding(query)
            
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Step 2: Search for similar embeddings in database
            logger.info("Searching for similar embeddings in database")
            
            # Use the new enhanced database function that bypasses RLS issues
            try:
                if project_id:
                    # Call the new Supabase function that handles RLS properly
                    # Get user_id from database connection if available
                    user_id = getattr(self.db, 'user_id', None)
                    
                    params = {
                        'query_embedding': query_embedding,
                        'project_uuid': project_id,
                        'similarity_threshold': threshold,
                        'match_limit': limit
                    }
                    
                    # Add user_id if available
                    if user_id:
                        params['auth_user_id'] = user_id
                    
                    result = self.db.client.rpc(
                        'get_similar_chunks_by_project',
                        params
                    ).execute()
                    
                    if result.data:
                        logger.info(f"Found {len(result.data)} similar chunks using enhanced function")
                        
                        # Transform the result to match expected format
                        result_chunks = []
                        for item in result.data:
                            result_chunks.append({
                                "chunk_text": item.get("chunk_text", ""),
                                "chunk_id": item.get("chunk_id"),
                                "document_id": item.get("document_id"),
                                "page_number": item.get("page_number", 1),
                                "similarity_score": float(item.get("similarity_score", 0.0)),
                                "filename": item.get("filename", "Unknown"),
                                "project_id": item.get("project_id")
                            })
                        
                        logger.info(f"Retrieved {len(result_chunks)} relevant chunks")
                        return result_chunks
                    else:
                        logger.info("No results from enhanced similarity search")
                        similar_embeddings = []
                else:
                    logger.warning("No project_id provided for similarity search")
                    similar_embeddings = []
            except Exception as e:
                logger.warning(f"Enhanced vector similarity search failed: {e}")
                # Fallback to original method
                try:
                    similar_embeddings = self.db.fetch_similar_embeddings(
                        query_embedding=query_embedding,
                        limit=limit,
                        match_threshold=threshold,
                        project_id=project_id
                    )
                except Exception as e2:
                    logger.error(f"All embedding searches failed: {e2}")
                    similar_embeddings = []
            
            if not similar_embeddings:
                logger.info("No similar embeddings found, trying fallback text search")
                return self._fallback_text_search(query, project_id, limit)
            
            # Step 3: Get the corresponding chunks
            chunk_ids = [emb.get("chunk_id") for emb in similar_embeddings if emb.get("chunk_id")]
            
            if not chunk_ids:
                logger.info("No valid chunk IDs found")
                return []
            
            # Fetch chunks by IDs
            chunks = self.db.fetch_chunks_by_ids(chunk_ids)
            logger.info(f"Fetched {len(chunks)} chunks for chunk_ids: {chunk_ids[:3]}...")  # Show first 3 IDs
            
            # Step 4: Combine chunk data with similarity scores
            result_chunks = []
            for chunk in chunks:
                # Find corresponding similarity score
                similarity_score = 0.0
                for emb in similar_embeddings:
                    if emb.get("chunk_id") == chunk.get("id"):
                        similarity_score = emb.get("similarity", 0.0)
                        break
                
                # Get document info
                document_info = {}
                try:
                    doc_result = self.db.client.table("documents").select("filename,project_id").eq("id", chunk.get("document_id")).execute()
                    if doc_result.data:
                        document_info = doc_result.data[0]
                except Exception as e:
                    logger.warning(f"Could not fetch document info: {e}")
                
                result_chunks.append({
                    "chunk_text": chunk.get("chunk_text", ""),
                    "chunk_id": chunk.get("id"),
                    "document_id": chunk.get("document_id"),
                    "page_number": chunk.get("page_number", 1),
                    "similarity_score": similarity_score,
                    "filename": document_info.get("filename", "Unknown"),
                    "project_id": document_info.get("project_id")
                })
            
            # Sort by similarity score (descending)
            result_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            logger.info(f"Retrieved {len(result_chunks)} relevant chunks")
            return result_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving similar chunks: {e}")
            return []
    
    def _fallback_text_search(self, query: str, project_id: Optional[str] = None, 
                             limit: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback text-based search when vector search fails.
        Uses basic text matching against documents.
        """
        try:
            logger.info("Using fallback text search")
            
            # Get documents for the project
            if project_id:
                documents = self.db.fetch_documents_by_project(project_id)
            else:
                # This would need a different method to get all documents
                documents = []
            
            if not documents:
                logger.info("No documents found for fallback search")
                return []
            
            # Simple keyword matching approach
            query_words = query.lower().split()
            result_chunks = []
            
            for doc in documents:
                filename = doc.get('filename', 'Unknown')
                doc_id = doc.get('id')
                
                # Create a simple "chunk" from document filename and metadata
                # This is a very basic fallback
                doc_text = f"{filename} {doc.get('metadata', {})}"
                
                # Check if any query words match
                matches = sum(1 for word in query_words if word in doc_text.lower())
                if matches > 0:
                    result_chunks.append({
                        "chunk_text": f"Document: {filename}\nThis document may contain information related to your query.",
                        "chunk_id": f"fallback_{doc_id}",
                        "document_id": doc_id,
                        "page_number": 1,
                        "similarity_score": matches / len(query_words),
                        "filename": filename,
                        "project_id": project_id
                    })
            
            # Sort by matches
            result_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
            return result_chunks[:limit]
            
        except Exception as e:
            logger.error(f"Error in fallback text search: {e}")
            return []
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate answer using Mistral's chat completion with retrieved context.
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Generated answer
        """
        try:
            # Step 1: Build context from retrieved chunks
            context_parts = []
            is_fallback_search = any(str(chunk.get("chunk_id", "")).startswith("fallback_") for chunk in context_chunks)
            
            for i, chunk in enumerate(context_chunks[:3], 1):  # Use top 3 chunks
                filename = chunk.get("filename", "Unknown")
                page = chunk.get("page_number", "Unknown")
                text = chunk.get("chunk_text", "")
                
                # Truncate very long chunks
                if len(text) > 500:
                    text = text[:500] + "..."
                
                context_parts.append(f"[Document: {filename}, Page: {page}]\n{text}")
            
            if not context_parts:
                return "I couldn't find relevant information in your documents to answer this question. Please try rephrasing your question or ensure the relevant documents are uploaded."
            
            # Add note if using fallback search
            fallback_note = ""
            if is_fallback_search:
                fallback_note = "\n\n*Note: Advanced document search is currently unavailable, so I'm providing a basic response based on document names and metadata. For detailed content analysis, please check if your documents have been fully processed.*"
            
            context_text = "\n\n".join(context_parts)
            
            # Step 2: Build enhanced prompt for better, humanized responses
            prompt = f"""You are a helpful AI assistant specializing in document analysis. Based on the provided context from documents, give a comprehensive and informative answer to the user's question.

Context Information:
---------------------
{context_text}
---------------------

Instructions:
- Provide a detailed, accurate answer based ONLY on the information in the context above
- Write in a natural, conversational tone that's easy to understand
- If the information is technical, explain it in accessible terms while maintaining accuracy
- Structure your response with clear explanations and specific details from the documents
- If you need to make reasonable inferences from the context, clearly indicate this
- If the context doesn't contain enough information to fully answer the question, say so

User Question: {query}

Answer:"""
            
            # Step 3: Generate response using Mistral
            logger.info("Generating response with Mistral")
            
            messages = [
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            chat_response = self.client.chat.complete(
                model="mistral-large-latest",
                messages=messages,
                max_tokens=1000,
                temperature=0.1  # Low temperature for factual responses
            )
            
            answer = chat_response.choices[0].message.content
            
            # Step 4: Add source information
            sources = []
            for chunk in context_chunks[:3]:
                filename = chunk.get("filename", "Unknown")
                page = chunk.get("page_number", "Unknown")
                sources.append(f"{filename} (Page {page})")
            
            if sources:
                answer += f"\n\n**Sources:** {', '.join(sources)}"
            
            # Add fallback note if applicable
            answer += fallback_note
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I encountered an error while generating the answer: {str(e)}. Please try again."
    
    def simple_query(self, query: str, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main query method that combines retrieval and generation.
        
        Args:
            query: User's question
            project_id: Optional project filter
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Processing simple query: {query}")
            
            # Step 1: Retrieve relevant chunks
            relevant_chunks = self.retrieve_similar_chunks(query, project_id)
            
            if not relevant_chunks:
                return {
                    "answer": "I couldn't find relevant information in your documents to answer this question. Please try rephrasing your question or ensure the relevant documents are uploaded to the selected project.",
                    "chunks_found": 0,
                    "sources": [],
                    "project_id": project_id
                }
            
            # Step 2: Generate answer
            answer = self.generate_answer(query, relevant_chunks)
            
            # Step 3: Prepare response metadata
            sources = []
            for chunk in relevant_chunks:
                filename = chunk.get("filename", "Unknown")
                page = chunk.get("page_number", "Unknown")
                similarity = chunk.get("similarity_score", 0.0)
                sources.append({
                    "filename": filename,
                    "page": page,
                    "similarity": similarity,
                    "chunk_id": chunk.get("chunk_id")
                })
            
            return {
                "answer": answer,
                "chunks_found": len(relevant_chunks),
                "sources": sources,
                "project_id": project_id,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error in simple query: {e}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "chunks_found": 0,
                "sources": [],
                "project_id": project_id,
                "error": str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status."""
        return {
            "mistral_client_available": self.client is not None,
            "database_connected": self.db is not None,
            "chunk_size": self.chunk_size
        } 