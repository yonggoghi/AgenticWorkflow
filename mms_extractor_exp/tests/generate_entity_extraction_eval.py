#!/usr/bin/env python3
"""
Entity Extraction Evaluation Data Generator
============================================

Generates CSV files for human evaluation of LLM entity extraction.
Uses ProductExtractionTracer to capture 1st stage HYBRID_DAG_EXTRACTION_PROMPT results.

Purpose:
    Improve extraction logic focusing on LLM prompts by comparing
    extracted_entities to correct_entities (human annotated).

Output Format:
    CSV with columns:
    - mms: Original MMS message text
    - extracted_entities_{model}: 1st stage entities from HYBRID_DAG_EXTRACTION_PROMPT
    - linked_entities_{model}: Final entities after ID linking (canonical product names)
    - correct_extracted_entities: Human annotated correct extracted entities
    - correct_linked_entities: Human annotated correct linked entities

    Re-evaluation accumulates columns (keeps all previous results):
    - Initial (ax):     mms, extracted_entities_ax, linked_entities_ax, correct_extracted_entities, correct_linked_entities
    - + gem:            mms, extracted_entities_ax, linked_entities_ax, extracted_entities_gem, linked_entities_gem, ...
    - + ax again:       ... extracted_entities_ax_v2, linked_entities_ax_v2, ...

Usage:
    # New evaluation from text file (one MMS per line, no sampling)
    /Users/yongwook/workspace/AgenticWorkflow/venv/bin/python \
        tests/generate_entity_extraction_eval.py \
        --input-file data/reg_test.txt \
        --output-dir outputs/ \
        --llm-model ax

    # New evaluation from CSV file (with stratified sampling)
    /Users/yongwook/workspace/AgenticWorkflow/venv/bin/python \
        tests/generate_entity_extraction_eval.py \
        --input-file data/mms_data_251001_260205.csv \
        --output-dir outputs/ \
        --sample-size 50 \
        --llm-model ax \
        --random-seed 42

    # Re-evaluate existing evaluation file (preserves correct_entities)
    /Users/yongwook/workspace/AgenticWorkflow/venv/bin/python \
        tests/generate_entity_extraction_eval.py \
        --re-evaluate outputs/entity_extraction_eval_20260205_120019.csv \
        --output-dir outputs/ \
        --llm-model ax
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.trace_product_extraction import ProductExtractionTracer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def categorize_msg_nm(msg_nm: str) -> str:
    """
    Categorize message by msg_nm patterns for stratified sampling.

    Categories:
        - 0 day 혜택: 0 day promotions
        - T day 혜택: T day promotions
        - 대리점: Store/dealer announcements
        - T 우주: T Universe subscription
        - special T/장기고객: Loyalty programs
        - 서비스/요금제: Services and rate plans
        - 통화서비스: Call-related services (컬러링, 콜키퍼)
        - 단말기: Device promotions (iPhone, Galaxy)
        - 이벤트/프로모션: General events and promotions
        - 기타: Others
    """
    if pd.isna(msg_nm):
        return '기타'

    msg_nm = str(msg_nm)

    if '0 day' in msg_nm or '0day' in msg_nm:
        return '0 day 혜택'
    elif 'T day' in msg_nm or 'Tday' in msg_nm:
        return 'T day 혜택'
    elif '대리점' in msg_nm:
        return '대리점'
    elif 'T 우주' in msg_nm or '우주패스' in msg_nm:
        return 'T 우주'
    elif 'special T' in msg_nm or '장기' in msg_nm:
        return 'special T/장기고객'
    elif '요금제' in msg_nm or '서비스' in msg_nm or '부가' in msg_nm:
        return '서비스/요금제'
    elif '컬러링' in msg_nm or '콜키퍼' in msg_nm or '통화' in msg_nm:
        return '통화서비스'
    elif 'iPhone' in msg_nm or '아이폰' in msg_nm or '갤럭시' in msg_nm:
        return '단말기'
    elif '이벤트' in msg_nm or '혜택' in msg_nm:
        return '이벤트/프로모션'
    else:
        return '기타'


def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """
    Perform stratified sampling based on msg_nm categories.

    Args:
        df: Input DataFrame with msg_nm column
        sample_size: Target total sample size
        seed: Random seed for reproducibility

    Returns:
        Stratified sample DataFrame
    """
    # Add category column
    df = df.copy()
    df['category'] = df['msg_nm'].apply(categorize_msg_nm)

    # Calculate samples per category (proportional)
    category_counts = df['category'].value_counts()
    total = len(df)

    logger.info("Category distribution in source data:")
    for cat, count in category_counts.items():
        logger.info(f"  {cat}: {count} ({count/total*100:.1f}%)")

    # Sample proportionally, with minimum 1 per category if possible
    samples = []
    remaining = sample_size

    for category in category_counts.index:
        cat_df = df[df['category'] == category]
        # Proportional allocation
        n_samples = max(1, int(sample_size * len(cat_df) / total))
        n_samples = min(n_samples, len(cat_df), remaining)

        if n_samples > 0:
            cat_sample = cat_df.sample(n=n_samples, random_state=seed)
            samples.append(cat_sample)
            remaining -= n_samples
            logger.info(f"  Sampled {n_samples} from '{category}'")

    result = pd.concat(samples, ignore_index=True)

    # If we still need more samples, fill from largest categories
    if len(result) < sample_size:
        already_sampled = set(result.index)
        remaining_df = df[~df.index.isin(already_sampled)]
        extra_needed = sample_size - len(result)
        if len(remaining_df) >= extra_needed:
            extra = remaining_df.sample(n=extra_needed, random_state=seed)
            result = pd.concat([result, extra], ignore_index=True)

    logger.info(f"Total sampled: {len(result)}")
    return result


def extract_entities(
    tracer: ProductExtractionTracer,
    mms_text: str,
    message_id: str = "#"
) -> Tuple[str, List[str], str, List[str]]:
    """
    Run tracer and return both 1st stage entities and linked entities.

    Args:
        tracer: ProductExtractionTracer instance
        mms_text: MMS message text
        message_id: Message identifier

    Returns:
        Tuple of (extracted_entities_str, extracted_entities_list,
                  linked_entities_str, linked_entities_list)
    """
    try:
        # Run the full trace
        trace = tracer.trace_message(mms_text, message_id)

        # Get 1st stage entities from HYBRID_DAG_EXTRACTION_PROMPT
        extracted = tracer._entity_trace.first_stage_entities or []
        sorted_extracted = sorted(extracted)
        extracted_str = " | ".join(sorted_extracted)

        # Get linked entities from final products
        # Mark unlinked entities (item_id == '#') with (#) suffix
        final_products = trace.final_products or []
        linked = []
        for p in final_products:
            name = p.get('item_nm', '')
            if not name:
                continue
            item_id = p.get('item_id', [])
            # item_id can be ['#'] (list) or '#' (string) for unlinked entities
            if item_id == ['#'] or item_id == '#':
                linked.append(f"{name}(#)")
            else:
                linked.append(name)
        sorted_linked = sorted(set(linked))  # unique and sorted
        linked_str = " | ".join(sorted_linked)

        return extracted_str, sorted_extracted, linked_str, sorted_linked

    except Exception as e:
        logger.error(f"Error extracting entities for {message_id}: {e}")
        return "", [], "", []


def generate_evaluation_csv(
    input_file: str,
    output_dir: str,
    sample_size: int,
    llm_model: str,
    random_seed: int,
    context_mode: str = 'dag'
) -> str:
    """
    Generate evaluation CSV file.

    Args:
        input_file: Path to input CSV file
        output_dir: Output directory path
        sample_size: Number of samples to generate
        llm_model: LLM model to use for extraction
        random_seed: Random seed for sampling
        context_mode: Entity extraction context mode ('dag', 'pairing', 'none', 'ont')

    Returns:
        Path to generated CSV file
    """
    # Load data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} rows")

    # Stratified sampling
    logger.info(f"Performing stratified sampling (n={sample_size}, seed={random_seed})")
    samples = stratified_sample(df, sample_size, random_seed)

    # Initialize tracer with default extractor kwargs (same as cli.py defaults)
    # Using 'local' data source to avoid Oracle DB dependency
    extractor_kwargs = {
        'llm_model': llm_model,
        'entity_llm_model': llm_model,
        'entity_extraction_mode': 'llm',
        'offer_info_data_src': 'local',  # Use local CSV files instead of DB
        'product_info_extraction_mode': 'llm',
        'extract_entity_dag': False,
        'entity_extraction_context_mode': context_mode,
    }

    logger.info(f"Initializing ProductExtractionTracer with config: {extractor_kwargs}")
    tracer = ProductExtractionTracer(extractor_kwargs)

    # Process each sample
    results = []
    total = len(samples)

    for idx, row in samples.iterrows():
        i = len(results) + 1
        mms_text = row['mms_phrs']
        msg_nm = row.get('msg_nm', '')
        offer_dt = row.get('offer_dt', '')

        logger.info(f"[{i}/{total}] Processing: {msg_nm[:50]}...")

        # Extract entities (both 1st stage and linked)
        extracted_str, extracted_list, linked_str, linked_list = extract_entities(
            tracer,
            mms_text,
            message_id=f"eval_{i}"
        )

        logger.info(f"  -> Extracted: {extracted_str[:80]}..." if len(extracted_str) > 80 else f"  -> Extracted: {extracted_str}")
        logger.info(f"  -> Linked: {linked_str[:80]}..." if len(linked_str) > 80 else f"  -> Linked: {linked_str}")

        results.append({
            'mms': mms_text,
            f'extracted_entities_{llm_model}_{context_mode}': extracted_str,
            f'linked_entities_{llm_model}_{context_mode}': linked_str,
            'correct_extracted_entities': '',  # Empty for human annotator
            'correct_linked_entities': ''  # Empty for human annotator
        })

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"entity_extraction_eval_{timestamp}.csv"

    # Save to CSV with UTF-8 BOM for Excel compatibility
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    logger.info(f"Saved {len(results)} samples to {output_file}")

    return str(output_file)


def generate_evaluation_from_text(
    input_file: str,
    output_dir: str,
    llm_model: str,
    context_mode: str = 'dag'
) -> str:
    """
    Generate evaluation CSV file from a text file (one MMS per line).
    Processes all lines without sampling.

    Args:
        input_file: Path to input text file (one MMS message per line)
        output_dir: Output directory path
        llm_model: LLM model to use for extraction
        context_mode: Entity extraction context mode ('dag', 'pairing', 'none', 'ont')

    Returns:
        Path to generated CSV file
    """
    # Load data from text file
    logger.info(f"Loading data from text file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        mms_lines = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(mms_lines)} MMS messages")

    # Initialize tracer with default extractor kwargs (same as cli.py defaults)
    # Using 'local' data source to avoid Oracle DB dependency
    extractor_kwargs = {
        'llm_model': llm_model,
        'entity_llm_model': llm_model,
        'entity_extraction_mode': 'llm',
        'offer_info_data_src': 'local',  # Use local CSV files instead of DB
        'product_info_extraction_mode': 'llm',
        'extract_entity_dag': False,
        'entity_extraction_context_mode': context_mode,
    }

    logger.info(f"Initializing ProductExtractionTracer with config: {extractor_kwargs}")
    tracer = ProductExtractionTracer(extractor_kwargs)

    # Process each MMS message
    results = []
    total = len(mms_lines)

    for i, mms_text in enumerate(mms_lines, 1):
        logger.info(f"[{i}/{total}] Processing: {mms_text[:50]}...")

        # Extract entities (both 1st stage and linked)
        extracted_str, extracted_list, linked_str, linked_list = extract_entities(
            tracer,
            mms_text,
            message_id=f"eval_{i}"
        )

        logger.info(f"  -> Extracted: {extracted_str[:80]}..." if len(extracted_str) > 80 else f"  -> Extracted: {extracted_str}")
        logger.info(f"  -> Linked: {linked_str[:80]}..." if len(linked_str) > 80 else f"  -> Linked: {linked_str}")

        results.append({
            'mms': mms_text,
            f'extracted_entities_{llm_model}_{context_mode}': extracted_str,
            f'linked_entities_{llm_model}_{context_mode}': linked_str,
            'correct_extracted_entities': '',  # Empty for human annotator
            'correct_linked_entities': ''  # Empty for human annotator
        })

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"entity_extraction_eval_{timestamp}.csv"

    # Save to CSV with UTF-8 BOM for Excel compatibility
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    logger.info(f"Saved {len(results)} samples to {output_file}")

    return str(output_file)


def get_next_column_name(df: pd.DataFrame, llm_model: str, context_mode: str, col_type: str = 'extracted') -> str:
    """
    Determine the column name for new results.

    If model+context hasn't been used before: {col_type}_entities_{model}_{context_mode}
    If used before: {col_type}_entities_{model}_{context_mode}_v{n+1}

    Args:
        df: DataFrame with existing columns
        llm_model: LLM model name
        context_mode: Entity extraction context mode
        col_type: 'extracted' or 'linked'

    Returns:
        Column name to use for new results
    """
    base_col = f'{col_type}_entities_{llm_model}_{context_mode}'

    # Find all existing columns for this model
    existing = [col for col in df.columns if col.startswith(base_col)]

    if not existing:
        return base_col  # First time: extracted_entities_ax or linked_entities_ax

    # Find highest version
    max_version = 1
    for col in existing:
        if col == base_col:
            continue  # This is v1 (no suffix)
        # Parse version from extracted_entities_ax_v2, extracted_entities_ax_v3, etc.
        if '_v' in col:
            try:
                version = int(col.split('_v')[-1])
                max_version = max(max_version, version)
            except ValueError:
                pass

    return f'{base_col}_v{max_version + 1}'


def re_evaluate_csv(
    eval_file: str,
    output_dir: str,
    llm_model: str,
    context_mode: str = 'dag'
) -> str:
    """
    Re-evaluate an existing evaluation CSV file.
    Accumulates all results - keeps all existing extracted_entities_* and linked_entities_* columns
    and adds new columns for the current model.

    If the same model is used again, adds version suffix (v2, v3, etc.)

    Args:
        eval_file: Path to existing evaluation CSV file
        output_dir: Output directory path
        llm_model: LLM model to use for extraction
        context_mode: Entity extraction context mode ('dag', 'pairing', 'none', 'ont')

    Returns:
        Path to generated CSV file
    """
    # Load existing evaluation file
    logger.info(f"Loading existing evaluation file: {eval_file}")
    df = pd.read_csv(eval_file)
    logger.info(f"Loaded {len(df)} rows")

    # Validate required columns
    if 'mms' not in df.columns:
        raise ValueError("Input file must have 'mms' column")

    # Determine the new column names (with version suffix if needed)
    new_extracted_col = get_next_column_name(df, llm_model, context_mode, 'extracted')
    new_linked_col = get_next_column_name(df, llm_model, context_mode, 'linked')
    logger.info(f"New columns: '{new_extracted_col}', '{new_linked_col}'")

    # Find all existing extracted_entities and linked_entities columns to preserve
    existing_extracted_cols = [col for col in df.columns if col.startswith('extracted_entities')]
    existing_linked_cols = [col for col in df.columns if col.startswith('linked_entities')]
    logger.info(f"Existing extracted columns: {existing_extracted_cols}")
    logger.info(f"Existing linked columns: {existing_linked_cols}")

    # Check for correct_* columns (new format) or correct_entities (old format)
    has_correct_extracted = 'correct_extracted_entities' in df.columns
    has_correct_linked = 'correct_linked_entities' in df.columns
    has_old_correct = 'correct_entities' in df.columns

    # Initialize tracer
    extractor_kwargs = {
        'llm_model': llm_model,
        'entity_llm_model': llm_model,
        'entity_extraction_mode': 'llm',
        'offer_info_data_src': 'local',
        'product_info_extraction_mode': 'llm',
        'extract_entity_dag': False,
        'entity_extraction_context_mode': context_mode,
    }

    logger.info(f"Initializing ProductExtractionTracer with config: {extractor_kwargs}")
    tracer = ProductExtractionTracer(extractor_kwargs)

    # Process each row
    results = []
    total = len(df)

    for idx, row in df.iterrows():
        i = idx + 1
        mms_text = row['mms']

        logger.info(f"[{i}/{total}] Re-evaluating...")

        # Extract entities (both 1st stage and linked)
        extracted_str, extracted_list, linked_str, linked_list = extract_entities(
            tracer,
            mms_text,
            message_id=f"reeval_{i}"
        )

        logger.info(f"  -> Extracted: {extracted_str[:80]}..." if len(extracted_str) > 80 else f"  -> Extracted: {extracted_str}")
        logger.info(f"  -> Linked: {linked_str[:80]}..." if len(linked_str) > 80 else f"  -> Linked: {linked_str}")

        # Build result row: mms + all existing columns + new columns + correct columns
        result_row = {'mms': mms_text}

        # Preserve all existing extracted_entities columns
        for col in existing_extracted_cols:
            value = row.get(col, '')
            result_row[col] = '' if pd.isna(value) else value

        # Preserve all existing linked_entities columns
        for col in existing_linked_cols:
            value = row.get(col, '')
            result_row[col] = '' if pd.isna(value) else value

        # Add new extraction results
        result_row[new_extracted_col] = extracted_str
        result_row[new_linked_col] = linked_str

        # Preserve correct_extracted_entities
        if has_correct_extracted:
            value = row.get('correct_extracted_entities', '')
            result_row['correct_extracted_entities'] = '' if pd.isna(value) else value
        elif has_old_correct:
            # Migrate from old format
            value = row.get('correct_entities', '')
            result_row['correct_extracted_entities'] = '' if pd.isna(value) else value
        else:
            result_row['correct_extracted_entities'] = ''

        # Preserve correct_linked_entities
        if has_correct_linked:
            value = row.get('correct_linked_entities', '')
            result_row['correct_linked_entities'] = '' if pd.isna(value) else value
        else:
            result_row['correct_linked_entities'] = ''

        results.append(result_row)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"entity_extraction_eval_{timestamp}.csv"

    # Save to CSV with UTF-8 BOM for Excel compatibility
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    logger.info(f"Saved {len(results)} samples to {output_file}")

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Generate entity extraction evaluation CSV for human annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # New evaluation from text file (one MMS per line, no sampling)
    python tests/generate_entity_extraction_eval.py \\
        --input-file data/reg_test.txt \\
        --output-dir outputs/ \\
        --llm-model ax

    # New evaluation from CSV file (with stratified sampling)
    python tests/generate_entity_extraction_eval.py \\
        --input-file data/mms_data_251001_260205.csv \\
        --output-dir outputs/ \\
        --sample-size 50 \\
        --llm-model ax \\
        --random-seed 42

    # Re-evaluate existing file (preserves correct_entities)
    python tests/generate_entity_extraction_eval.py \\
        --re-evaluate outputs/entity_extraction_eval_20260205_120019.csv \\
        --output-dir outputs/ \\
        --llm-model ax
        """
    )

    parser.add_argument(
        "--input-file", "-i",
        type=str,
        default="data/reg_test.txt",
        help="Input file path (.txt for one MMS per line without sampling, .csv for stratified sampling)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs/",
        help="Output directory (default: outputs/)"
    )
    parser.add_argument(
        "--sample-size", "-n",
        type=int,
        default=50,
        help="Number of samples to generate (default: 50)"
    )
    parser.add_argument(
        "--llm-model", "-m",
        type=str,
        choices=['gem', 'ax', 'cld', 'gen', 'gpt', 'opus'],
        default='ax',
        help="LLM model for extraction (default: ax)"
    )
    parser.add_argument(
        "--random-seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--re-evaluate", "-r",
        type=str,
        default=None,
        help="Re-evaluate an existing evaluation CSV file (preserves correct_entities)"
    )
    parser.add_argument(
        "--context-mode", "-c",
        type=str,
        choices=['dag', 'pairing', 'none', 'ont', 'typed'],
        default='dag',
        help="Entity extraction context mode (default: dag)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Entity Extraction Evaluation Data Generator")
    print("=" * 60)

    if args.re_evaluate:
        # Re-evaluation mode
        print(f"Mode: RE-EVALUATE")
        print(f"Input eval file: {args.re_evaluate}")
        print(f"Output dir: {args.output_dir}")
        print(f"LLM model: {args.llm_model}")
        print(f"Context mode: {args.context_mode}")
        print("=" * 60)

        output_file = re_evaluate_csv(
            eval_file=args.re_evaluate,
            output_dir=args.output_dir,
            llm_model=args.llm_model,
            context_mode=args.context_mode
        )
    else:
        # New evaluation mode - detect file type
        input_path = Path(args.input_file)
        is_text_file = input_path.suffix.lower() == '.txt'

        if is_text_file:
            # Text file mode: one MMS per line, no sampling
            print(f"Mode: NEW EVALUATION (text file, no sampling)")
            print(f"Input file: {args.input_file}")
            print(f"Output dir: {args.output_dir}")
            print(f"LLM model: {args.llm_model}")
            print(f"Context mode: {args.context_mode}")
            print("=" * 60)

            output_file = generate_evaluation_from_text(
                input_file=args.input_file,
                output_dir=args.output_dir,
                llm_model=args.llm_model,
                context_mode=args.context_mode
            )
        else:
            # CSV file mode: stratified sampling
            print(f"Mode: NEW EVALUATION (CSV file, stratified sampling)")
            print(f"Input file: {args.input_file}")
            print(f"Output dir: {args.output_dir}")
            print(f"Sample size: {args.sample_size}")
            print(f"LLM model: {args.llm_model}")
            print(f"Context mode: {args.context_mode}")
            print(f"Random seed: {args.random_seed}")
            print("=" * 60)

            output_file = generate_evaluation_csv(
                input_file=args.input_file,
                output_dir=args.output_dir,
                sample_size=args.sample_size,
                llm_model=args.llm_model,
                random_seed=args.random_seed,
                context_mode=args.context_mode
            )

    print("=" * 60)
    print(f"✅ Done! Output saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
