#!/usr/bin/env python3
"""
Test script to verify API cost calculations for OpenAI and Mistral models.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.processing.text_processor import TextProcessor
from src.extraction.pdf_extractor import PDFExtractor

def test_openai_cost_calculations():
    """Test OpenAI cost calculations with various models and token counts."""
    processor = TextProcessor()
    
    print("=== Testing OpenAI Cost Calculations ===")
    
    test_cases = [
        # (model, total_tokens, input_tokens, output_tokens, expected_description)
        ("gpt-4o", 1000, 800, 200, "GPT-4o with separate input/output tokens"),
        ("gpt-4o-mini", 1000, None, None, "GPT-4o-mini with total tokens only"),
        ("gpt-4", 500, 400, 100, "GPT-4 with separate tokens"),
        ("gpt-3.5-turbo", 2000, 1500, 500, "GPT-3.5-turbo"),
        ("gpt-4.1-nano", 1000, 700, 300, "Custom nano model"),
        ("unknown-model", 1000, 700, 300, "Unknown model fallback"),
    ]
    
    for model, total_tokens, input_tokens, output_tokens, description in test_cases:
        cost = processor._calculate_openai_cost(model, total_tokens, input_tokens, output_tokens)
        print(f"{description}:")
        print(f"  Model: {model}")
        print(f"  Tokens: {total_tokens} total, {input_tokens} input, {output_tokens} output")
        print(f"  Cost: ${cost:.6f}")
        print()

def test_mistral_cost_calculations():
    """Test Mistral cost calculations with various models."""
    processor = TextProcessor()
    extractor = PDFExtractor()
    
    print("=== Testing Mistral Cost Calculations ===")
    
    # Test embedding model
    embedding_cases = [
        ("mistral-embed", 1000, "Mistral embedding model"),
        ("mistral-small", 500, "Mistral small chat model"),
        ("unknown-mistral", 1000, "Unknown Mistral model"),
    ]
    
    for model, tokens, description in embedding_cases:
        cost = processor._calculate_mistral_cost(model, tokens)
        print(f"{description}:")
        print(f"  Model: {model}")
        print(f"  Tokens: {tokens}")
        print(f"  Cost: ${cost:.6f}")
        print()
    
    # Test OCR model
    ocr_cases = [
        ("pixtral-12b-2409", 2000, "Pixtral OCR model"),
        ("pixtral-large", 1500, "Pixtral large model"),
        ("unknown-ocr", 1000, "Unknown OCR model"),
    ]
    
    for model, tokens, description in ocr_cases:
        cost = extractor._calculate_mistral_ocr_cost(model, tokens)
        print(f"{description}:")
        print(f"  Model: {model}")
        print(f"  Tokens: {tokens}")
        print(f"  Cost: ${cost:.6f}")
        print()

def test_real_world_scenarios():
    """Test with realistic token usage scenarios."""
    processor = TextProcessor()
    
    print("=== Real-World Scenario Tests ===")
    
    # Large document processing
    large_doc_input = 15000
    large_doc_output = 3000
    large_doc_total = large_doc_input + large_doc_output
    
    gpt4o_cost = processor._calculate_openai_cost("gpt-4o", large_doc_total, large_doc_input, large_doc_output)
    gpt4o_mini_cost = processor._calculate_openai_cost("gpt-4o-mini", large_doc_total, large_doc_input, large_doc_output)
    
    print(f"Large document processing ({large_doc_input} input + {large_doc_output} output tokens):")
    print(f"  GPT-4o: ${gpt4o_cost:.4f}")
    print(f"  GPT-4o-mini: ${gpt4o_mini_cost:.4f}")
    print(f"  Savings with mini: ${(gpt4o_cost - gpt4o_mini_cost):.4f} ({((gpt4o_cost - gpt4o_mini_cost) / gpt4o_cost * 100):.1f}%)")
    print()
    
    # Embedding generation
    chunks = 50
    tokens_per_chunk = 300
    total_embedding_tokens = chunks * tokens_per_chunk
    
    mistral_embed_cost = processor._calculate_mistral_cost("mistral-embed", total_embedding_tokens)
    
    print(f"Embedding generation ({chunks} chunks × {tokens_per_chunk} tokens = {total_embedding_tokens} total):")
    print(f"  Mistral Embed: ${mistral_embed_cost:.4f}")
    print()

def compare_old_vs_new_pricing():
    """Compare old flat pricing vs new per-million pricing."""
    processor = TextProcessor()
    
    print("=== Old vs New Pricing Comparison ===")
    
    # Test with 10,000 tokens (typical document)
    tokens = 10000
    input_tokens = 7000
    output_tokens = 3000
    
    # New pricing
    new_cost_gpt4o = processor._calculate_openai_cost("gpt-4o", tokens, input_tokens, output_tokens)
    new_cost_mini = processor._calculate_openai_cost("gpt-4o-mini", tokens, input_tokens, output_tokens)
    
    # Old pricing simulation (flat rate per 1K tokens)
    old_gpt4o_rate = 0.005  # Old rate
    old_mini_rate = 0.00015  # Old rate
    old_cost_gpt4o = (tokens / 1000) * old_gpt4o_rate
    old_cost_mini = (tokens / 1000) * old_mini_rate
    
    print(f"For {tokens} tokens ({input_tokens} input + {output_tokens} output):")
    print(f"GPT-4o:")
    print(f"  Old pricing: ${old_cost_gpt4o:.6f}")
    print(f"  New pricing: ${new_cost_gpt4o:.6f}")
    print(f"  Difference: ${(new_cost_gpt4o - old_cost_gpt4o):.6f}")
    print()
    print(f"GPT-4o-mini:")
    print(f"  Old pricing: ${old_cost_mini:.6f}")
    print(f"  New pricing: ${new_cost_mini:.6f}")
    print(f"  Difference: ${(new_cost_mini - old_cost_mini):.6f}")
    print()

if __name__ == "__main__":
    print("API Cost Calculation Test Suite")
    print("=" * 50)
    print()
    
    test_openai_cost_calculations()
    test_mistral_cost_calculations()
    test_real_world_scenarios()
    compare_old_vs_new_pricing()
    
    print("Test completed!") 