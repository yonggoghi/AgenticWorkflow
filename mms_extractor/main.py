"""
Main script demonstrating the usage of MMS extractor modules.

This script shows a complete workflow for processing MMS data:
1. Loading and preprocessing data
2. Entity matching
3. Text processing
4. LLM-based information extraction
5. Result validation and saving
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

from .core.entity_matcher import KoreanEntityMatcher, find_entities_in_text
from .core.text_processor import TextProcessor
from .core.llm_processor import LLMProcessor, LLMResponse
from .models.schemas import (
    ProgramInfo,
    ExtractionResult,
    ExtractionSchema,
    save_extraction_results
)
from .utils.data_utils import (
    load_csv,
    load_json,
    save_json,
    create_backup,
    filter_dataframe
)
from .utils.text_utils import normalize_text, extract_keywords
from .config.settings import (
    DATA_PATHS,
    ENTITY_MATCHING_CONFIG,
    TEXT_PROCESSING_CONFIG,
    LLM_PROCESSING_CONFIG,
    API_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default model settings from original code
DEFAULT_MODELS = {
    "openai": "skt/a.x-3-lg",
    "anthropic": "skt/claude-3-5-sonnet-20241022"
}

DEFAULT_MAX_TOKENS = 100

class MMSExtractor:
    """
    Main class for MMS data extraction and processing.
    
    This class orchestrates the entire workflow of:
    1. Loading and preprocessing data
    2. Entity matching
    3. Text processing
    4. LLM-based information extraction
    5. Result validation and saving
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the MMS extractor.
        
        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            output_dir: Directory for saving results
        """
        # Initialize components
        self.entity_matcher = KoreanEntityMatcher()
        self.text_processor = TextProcessor()
        self.llm_processor = LLMProcessor(
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key
        )
        
        # Set up output directory
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self._load_data()
        
    def _load_data(self) -> None:
        """Load required data files."""
        try:
            # Load entity lists
            self.item_info = load_csv(DATA_PATHS["item_info"])
            self.mms_data = load_csv(DATA_PATHS["mms_data"])
            self.pgm_tags = load_csv(DATA_PATHS["pgm_tags"])
            
            # Load rules and stop words
            self.alias_rules = load_json(DATA_PATHS["alias_rules"])
            self.stop_words = load_json(DATA_PATHS["stop_words"])
            
            # Build entity matcher
            self._build_entity_matcher()
            
            logger.info("Successfully loaded all data files")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _build_entity_matcher(self) -> None:
        """Build entity matcher with loaded data."""
        # Prepare entity list from item info
        entities = [
            (row["item_nm"], {"type": "item", "data": row.to_dict()})
            for _, row in self.item_info.iterrows()
        ]
        
        # Add program tags
        entities.extend([
            (row["tag_nm"], {"type": "tag", "data": row.to_dict()})
            for _, row in self.pgm_tags.iterrows()
        ])
        
        # Build matcher
        self.entity_matcher.build_from_list(entities)
        logger.info(f"Built entity matcher with {len(entities)} entities")
    
    def process_text(
        self,
        text: str,
        extract_entities: bool = True,
        extract_info: bool = True
    ) -> ExtractionResult:
        """
        Process a single text input.
        
        Args:
            text: Input text to process
            extract_entities: Whether to perform entity matching
            extract_info: Whether to extract structured information
            
        Returns:
            ExtractionResult containing processed information
        """
        start_time = time.time()
        error = None
        
        try:
            # Clean and normalize text
            cleaned_text = self.text_processor.clean_text(text)
            
            # Initialize program info
            program_info = ProgramInfo(
                title="",
                source_text=cleaned_text
            )
            
            # Extract entities if requested
            if extract_entities:
                matches = find_entities_in_text(
                    cleaned_text,
                    self.entity_matcher.entities,
                    min_similarity=ENTITY_MATCHING_CONFIG["min_similarity"]
                )
                
                # Update program info with matched entities
                for match in matches:
                    entity_type = match["entity"]["type"]
                    if entity_type == "item":
                        program_info.product = match["text"]
                    elif entity_type == "tag":
                        if not program_info.keywords:
                            program_info.keywords = []
                        program_info.keywords.append(match["text"])
            
            # Extract structured information if requested
            if extract_info:
                try:
                    # Process with LLM
                    response = self.llm_processor.extract_structured_data(
                        cleaned_text,
                        "program_info"
                    )
                    
                    # Convert to ProgramInfo
                    extracted_info = ExtractionSchema.to_program_info(response)
                    
                    # Update program info with extracted data
                    for field in extracted_info.__dataclass_fields__:
                        if getattr(extracted_info, field) is not None:
                            setattr(program_info, field, getattr(extracted_info, field))
                    
                except Exception as e:
                    logger.warning(f"LLM extraction failed: {str(e)}")
                    error = f"LLM extraction error: {str(e)}"
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return ExtractionResult(
                program_info=program_info,
                processing_time=processing_time,
                error=error
            )
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return ExtractionResult(
                program_info=ProgramInfo(title="", source_text=text),
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def process_batch(
        self,
        texts: List[str],
        batch_size: int = LLM_PROCESSING_CONFIG["batch_size"]
    ) -> List[ExtractionResult]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of texts to process
            batch_size: Number of texts to process in parallel
            
        Returns:
            List of ExtractionResults
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            # Process each text in batch
            batch_results = []
            for text in batch:
                result = self.process_text(text)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Save intermediate results
            self._save_results(batch_results, f"batch_{i//batch_size + 1}")
        
        return results
    
    def _save_results(
        self,
        results: List[ExtractionResult],
        suffix: Optional[str] = None
    ) -> None:
        """
        Save extraction results.
        
        Args:
            results: List of extraction results
            suffix: Optional suffix for filename
        """
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"extraction_results_{timestamp}"
        if suffix:
            filename = f"{filename}_{suffix}"
        filename = f"{filename}.json"
        
        # Save results
        output_path = self.output_dir / filename
        save_extraction_results(results, str(output_path))
        logger.info(f"Saved results to {output_path}")
    
    def run_pipeline(
        self,
        input_file: str,
        batch_size: int = LLM_PROCESSING_CONFIG["batch_size"]
    ) -> List[ExtractionResult]:
        """
        Run the complete extraction pipeline.
        
        Args:
            input_file: Path to input file (CSV or JSON)
            batch_size: Number of texts to process in parallel
            
        Returns:
            List of extraction results
        """
        try:
            # Load input data
            if input_file.endswith('.csv'):
                data = load_csv(input_file)
                texts = data['text'].tolist()  # Adjust column name as needed
            else:
                data = load_json(input_file)
                texts = [item['text'] for item in data]  # Adjust key as needed
            
            # Create backup of input file
            create_backup(input_file, self.output_dir / "backups")
            
            # Process texts
            results = self.process_batch(texts, batch_size)
            
            # Save final results
            self._save_results(results, "final")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            raise

def main():
    """Example usage of the MMS extractor."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MMS Data Extraction Pipeline")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input file (CSV or JSON)"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for saving results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=LLM_PROCESSING_CONFIG["batch_size"],
        help="Batch size for processing"
    )
    parser.add_argument(
        "--openai-key",
        default=API_CONFIG.get("openai_api_key"),
        help="OpenAI API key (defaults to config value)"
    )
    parser.add_argument(
        "--anthropic-key",
        default=API_CONFIG.get("anthropic_api_key"),
        help="Anthropic API key (defaults to config value)"
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_MODELS["openai"],
        help=f"OpenAI model to use (default: {DEFAULT_MODELS['openai']})"
    )
    parser.add_argument(
        "--anthropic-model",
        default=DEFAULT_MODELS["anthropic"],
        help=f"Anthropic model to use (default: {DEFAULT_MODELS['anthropic']})"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens for LLM response (default: {DEFAULT_MAX_TOKENS})"
    )
    parser.add_argument(
        "--min-similarity",
        type=int,
        default=ENTITY_MATCHING_CONFIG["min_similarity"],
        help=f"Minimum similarity for entity matching (default: {ENTITY_MATCHING_CONFIG['min_similarity']})"
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=ENTITY_MATCHING_CONFIG["ngram_size"],
        help=f"N-gram size for entity matching (default: {ENTITY_MATCHING_CONFIG['ngram_size']})"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor with custom model settings
        extractor = MMSExtractor(
            openai_api_key=args.openai_key,
            anthropic_api_key=args.anthropic_key,
            output_dir=args.output_dir
        )
        
        # Update LLM processor settings
        extractor.llm_processor.default_model = args.openai_model
        extractor.llm_processor.max_tokens = args.max_tokens
        
        # Update entity matcher settings
        extractor.entity_matcher.min_similarity = args.min_similarity
        extractor.entity_matcher.ngram_size = args.ngram_size
        
        # Run pipeline
        results = extractor.run_pipeline(
            args.input,
            batch_size=args.batch_size
        )
        
        # Print summary
        success_count = sum(1 for r in results if not r.error)
        logger.info(f"Processing complete:")
        logger.info(f"Total texts: {len(results)}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {len(results) - success_count}")
        
        # Print model settings used
        logger.info("\nModel settings used:")
        logger.info(f"OpenAI model: {args.openai_model}")
        logger.info(f"Anthropic model: {args.anthropic_model}")
        logger.info(f"Max tokens: {args.max_tokens}")
        logger.info(f"Min similarity: {args.min_similarity}")
        logger.info(f"N-gram size: {args.ngram_size}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 