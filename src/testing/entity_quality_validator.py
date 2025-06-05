"""Entity Quality Validation Framework for Enhanced Legal AI System"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class EntityQualityValidator:
    """Comprehensive validator for entity quality improvements"""
    
    def __init__(self, db_manager=None):
        """Initialize the validator"""
        self.db_manager = db_manager
        self.text_processor = None
        self.quality_filter = None
        self._init_processors()
    
    def _init_processors(self):
        """Initialize processing components"""
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            
            from processing.text_processor import TextProcessor
            from processing.entity_quality_filter import EntityQualityFilter
            
            self.text_processor = TextProcessor()
            self.quality_filter = EntityQualityFilter()
            
            logger.info("Entity quality validation processors initialized")
        except Exception as e:
            logger.error(f"Failed to initialize processors: {e}")
    
    def validate_document_processing(self, document_text: str, 
                                   document_type: str = "general") -> Dict[str, Any]:
        """Validate quality improvements on a real document"""
        if not self.text_processor or not self.quality_filter:
            return {"error": "Processors not initialized"}
        
        try:
            # Extract entities
            baseline_result = self.text_processor.extract_entities_and_relationships_openai(
                document_text,
                document_type=document_type
            )
            original_entities = baseline_result.get('entities', [])
            
            # Apply quality filtering
            filtered_entities, quality_metrics = self.quality_filter.filter_entities(
                original_entities,
                document_type=document_type,
                language="auto"
            )
            
            # Generate quality statistics
            quality_stats = self.quality_filter.get_quality_statistics(quality_metrics)
            
            return {
                'original_entities': len(original_entities),
                'filtered_entities': len(filtered_entities),
                'quality_statistics': quality_stats,
                'sample_filtered_entities': filtered_entities[:5],
                'sample_rejected_entities': [
                    {
                        'entity': original_entities[i].get('value', ''),
                        'reason': quality_metrics[i].rejection_reason
                    }
                    for i in range(min(5, len(original_entities)))
                    if i < len(quality_metrics) and not quality_metrics[i].passed_filter
                ]
            }
            
        except Exception as e:
            logger.error(f"Document validation failed: {e}")
            return {"error": str(e)} 
