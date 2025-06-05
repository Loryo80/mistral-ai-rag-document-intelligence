# Code for llm_generator.py (LLM Response Generation Module)
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log

# Import Mistral or OpenAI clients as needed
from mistralai import Mistral
from openai import OpenAI # Use OpenAI for gpt models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def is_retryable_llm_exception(exception):
    import openai
    import requests
    retryable = (
        ConnectionError,
        TimeoutError,
        requests.exceptions.RequestException,
        openai.OpenAIError,
        Exception,  # Optionally catch all for now
    )
    return isinstance(exception, retryable)

RETRY_LLM_CONFIG = dict(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(4),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

class LLMGenerator:
    """Generate answers using LLM models."""
    def __init__(self, db=None, user_id=None, provider="openai"):
        self.model = LLM_MODEL
        self.db = db
        self.user_id = user_id
        self.provider = provider

        if provider == "mistral" and MISTRAL_API_KEY:
            self.client = Mistral(api_key=MISTRAL_API_KEY)
            self.model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
            logger.info(f"Initialized Mistral LLM client with model: {self.model}")
        elif provider == "openai" and OPENAI_API_KEY:
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = LLM_MODEL
            logger.info(f"Initialized OpenAI LLM client with model: {self.model}")
        else:
            raise ValueError(f"No API key found for provider {provider}. Please set API keys in your .env file.")

    @retry(**RETRY_LLM_CONFIG)
    def generate_answer(self, query: str, context: List[str], citations: Optional[List[Dict[str, Any]]] = None, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an answer from the LLM given a query and context.
        The LLM is instructed to use the context and provide citations.

        Args:
            query: User's question, potentially including retrieved entities
            context: List of relevant document chunks. Each chunk is a string.
            citations: Optional list of citation metadata {document, section, page}.
                       These are passed to the LLM to help it cite correctly.
            document_id: Document ID for logging (optional)

        Returns:
            Dict with answer, citations, and confidence score
        """
        if self.provider == "mistral":
            return self.generate_answer_with_mistral(query, context, citations, document_id)
        else:
            return self.generate_answer_with_openai(query, context, citations, document_id)

    @retry(**RETRY_LLM_CONFIG)
    def generate_answer_with_mistral(self, query: str, context: List[str], citations: Optional[List[Dict[str, Any]]] = None, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate answer using Mistral API."""
        messages = [
            {"role": "system", "content": "You are a legal AI assistant. Provide concise and accurate answers based *only* on the provided context. When referencing information, cite the source by including the document name and page number directly in your answer, e.g., (Source: Contract.pdf, Page 3). If the answer is not in the context, state that you cannot answer based on the provided information. Avoid external knowledge."},
        ]

        # Construct the user message with context and query
        context_str = "\n\n".join([f"--- Context (Doc: {c.get('document', 'Unknown')}, Page: {c.get('page', 'N/A')}) ---\n{c['text']}" if isinstance(c, dict) and 'text' in c else c for c in context])

        full_query = f"Based on the following context, answer the question:\n\nContext:\n{context_str}\n\nQuestion: {query}"

        messages.append({"role": "user", "content": full_query})

        try:
            # Updated Mistral API call for newer version (1.8.0)
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=500,
            )
            
            answer = response.choices[0].message.content
            api_provider = "mistral"
            
            # --- API Usage Logging ---
            if self.db and api_provider:
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else None
                cost_usd = None
                self.db.log_api_usage(
                    user_id=self.user_id,
                    document_id=document_id,
                    api_provider=api_provider,
                    api_type="llm",
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    request_payload={"model": self.model, "messages_len": len(messages)},
                    response_metadata={"response_id": getattr(response, 'id', None)}
                )

            confidence = 0.9 if context and len(context_str) > 100 else 0.5

            return {
                "answer": answer,
                "citations": citations or [],
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Error during Mistral LLM generation: {e}", exc_info=True)
            raise

    @retry(**RETRY_LLM_CONFIG)
    def generate_answer_with_openai(self, query: str, context: List[str], citations: Optional[List[Dict[str, Any]]] = None, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate answer using OpenAI API."""
        messages = [
            {"role": "system", "content": "You are a legal AI assistant. Provide concise and accurate answers based *only* on the provided context. When referencing information, cite the source by including the document name and page number directly in your answer, e.g., (Source: Contract.pdf, Page 3). If the answer is not in the context, state that you cannot answer based on the provided information. Avoid external knowledge."},
        ]

        # Construct the user message with context and query
        context_str = "\n\n".join([f"--- Context (Doc: {c.get('document', 'Unknown')}, Page: {c.get('page', 'N/A')}) ---\n{c['text']}" if isinstance(c, dict) and 'text' in c else c for c in context])

        full_query = f"Based on the following context, answer the question:\n\nContext:\n{context_str}\n\nQuestion: {query}"

        messages.append({"role": "user", "content": full_query})

        try:
            if self.client:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1, # Keep low for factual answers
                    max_tokens=500,
                )
                answer = response.choices[0].message.content
                api_provider = "openai"
            else:
                answer = "[Error: LLM client not initialized.]"
                api_provider = None

            # --- API Usage Logging ---
            if self.db and api_provider:
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else None
                cost_usd = None # Optionally estimate based on provider pricing
                self.db.log_api_usage(
                    user_id=self.user_id,
                    document_id=document_id,
                    api_provider=api_provider,
                    api_type="llm",
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    request_payload={"model": self.model, "messages_len": len(messages)},
                    response_metadata={"response_id": getattr(response, 'id', None)}
                )

            # Simple heuristic for confidence: based on presence of context.
            # In a real system, this would be more sophisticated (e.g., parsing LLM's self-assessment, or cross-checking facts).
            confidence = 0.9 if context and len(context_str) > 100 else 0.5

            return {
                "answer": answer,
                "citations": citations or [], # LLM is instructed to put citations in answer text, but we still pass along original citations for structured display
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}", exc_info=True)
            return {
                "answer": f"I apologize, but I encountered an error while generating a response: {e}. Please try again.",
                "citations": [],
                "confidence": 0
            }

    @retry(**RETRY_LLM_CONFIG)
    def generate_technical_answer(self, query: str, context: List[str], document_type: str = "technical", citations: Optional[List[Dict[str, Any]]] = None, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a specialized answer for technical/chemical document queries.
        
        Args:
            query: User's question about technical/chemical information
            context: List of relevant document chunks
            document_type: Type of document ("technical", "chemical", "sds", etc.)
            citations: Optional citation metadata
            document_id: Document ID for logging
            
        Returns:
            Dict with answer, citations, and confidence score
        """
        # Select the appropriate system prompt based on document type
        if document_type.lower() in ["chemical", "sds", "safety_data_sheet"]:
            system_prompt = """You are a specialized chemical documentation assistant. Answer questions about chemical products, 
            safety data sheets, and technical specifications. Provide concise and technically accurate answers based ONLY on the 
            provided context. Include relevant parameters with their units when applicable. When referencing technical information, 
            cite the source document and page number. If the information is not in the context, clearly state this limitation."""
        elif document_type.lower() in ["technical", "specification", "datasheet"]:
            system_prompt = """You are a technical documentation specialist. Answer questions about product specifications,
            technical parameters, and product characteristics. Provide concise and technically accurate answers based ONLY on the
            provided context. Be precise with measurements, units, and technical terminology. Cite the source document and page number
            for any technical information. If the information is not in the context, clearly state this limitation."""
        else:
            system_prompt = """You are a documentation assistant. Provide concise and accurate answers based *only* on the provided context.
            When referencing information, cite the source by including the document name and page number. If the answer is not in the context,
            state that you cannot answer based on the provided information. Avoid external knowledge."""

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Format context with enhanced technical information
        formatted_context = []
        for chunk in context:
            if isinstance(chunk, dict) and 'text' in chunk:
                doc_name = chunk.get('document', 'Unknown')
                page_num = chunk.get('page', 'N/A')
                doc_type = chunk.get('document_type', '')
                doc_type_info = f", Type: {doc_type}" if doc_type else ""
                
                formatted_chunk = f"--- Context (Doc: {doc_name}, Page: {page_num}{doc_type_info}) ---\n{chunk['text']}"
                formatted_context.append(formatted_chunk)
            else:
                formatted_context.append(chunk)
        
        context_str = "\n\n".join(formatted_context)
        
        # Enhance the query with technical prompt
        full_query = f"Based on the following technical documentation, answer this question with precision and technical accuracy:\n\nContext:\n{context_str}\n\nQuestion: {query}"
        
        messages.append({"role": "user", "content": full_query})
        
        # Use the appropriate LLM provider
        if self.provider == "mistral":
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=600,
                )
                
                answer = response.choices[0].message.content
                api_provider = "mistral"
                
            except Exception as e:
                logger.error(f"Error during Mistral technical generation: {e}", exc_info=True)
                return {
                    "answer": f"I encountered an error while generating a technical response: {e}. Please try again.",
                    "citations": [],
                    "confidence": 0
                }
        else:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=600,
                )
                
                answer = response.choices[0].message.content
                api_provider = "openai"
                
            except Exception as e:
                logger.error(f"Error during OpenAI technical generation: {e}", exc_info=True)
                return {
                    "answer": f"I encountered an error while generating a technical response: {e}. Please try again.",
                    "citations": [],
                    "confidence": 0
                }
        
        # Log API usage
        if self.db and api_provider:
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else None
            self.db.log_api_usage(
                user_id=self.user_id,
                document_id=document_id,
                api_provider=api_provider,
                api_type="technical_llm",
                tokens_used=tokens_used,
                cost_usd=None,
                request_payload={"model": self.model, "document_type": document_type},
                response_metadata={"response_id": getattr(response, 'id', None)}
            )
        
        # Enhanced confidence calculation based on document type and context match
        if context and len(context_str) > 100:
            confidence = 0.9
            # Look for technical terms in the answer that match the context
            technical_terms = ["specification", "parameter", "value", "standard", "technical", "chemical", "product"]
            if any(term in answer.lower() for term in technical_terms):
                confidence += 0.05
            
            # Cap confidence at 0.95
            confidence = min(confidence, 0.95)
        else:
            confidence = 0.5
        
        return {
            "answer": answer,
            "citations": citations or [],
            "confidence": confidence,
            "document_type": document_type
        }

if __name__ == "__main__":
    # Example usage for testing
    # Ensure MISTRAL_API_KEY or OPENAI_API_KEY is set in your .env file
    
    # Simulate a context with citation info
    mock_context_chunks = [
        {"text": "Section 3.1 outlines the indemnification clause, stating Party A shall indemnify Party B against all losses arising from breach of warranty.", "document": "Contract A.pdf", "page": 3},
        {"text": "The agreement terminates on December 31, 2025, as per Clause 7.2.", "document": "Contract A.pdf", "page": 7}
    ]
    
    mock_citations = [
        {"document": "Contract A.pdf", "section": "Section 3.1", "page": 3},
        {"document": "Contract A.pdf", "section": "Clause 7.2", "page": 7}
    ]

    generator = LLMGenerator()

    # Test query
    query = "What are the key terms of the indemnification clause and the termination date?"
    
    response = generator.generate_answer(query, mock_context_chunks, mock_citations)
    
    print("\n--- Generated Response ---")
    print(f"Answer: {response['answer']}")
    print(f"Confidence: {response['confidence']:.2f}")
    print(f"Citations (original): {response['citations']}")

    # Test with no context
    print("\n--- Generated Response (No Context) ---")
    query_no_context = "Tell me about the history of contract law."
    response_no_context = generator.generate_answer(query_no_context, [])
    print(f"Answer: {response_no_context['answer']}")
    print(f"Confidence: {response_no_context['confidence']:.2f}")
