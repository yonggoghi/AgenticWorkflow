"""
Data utility functions for the MMS extractor.
"""

from typing import List, Dict, Any, Optional, Union, Set
import json
import csv
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json(
    file_path: Union[str, Path],
    encoding: str = 'utf-8'
) -> Any:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        encoding: File encoding
        
    Returns:
        Loaded data
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        raise

def save_json(
    data: Any,
    file_path: Union[str, Path],
    indent: int = 2,
    encoding: str = 'utf-8',
    ensure_ascii: bool = False
) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to output file
        indent: JSON indentation
        encoding: File encoding
        ensure_ascii: Whether to escape non-ASCII characters
    """
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(
                data,
                f,
                indent=indent,
                ensure_ascii=ensure_ascii
            )
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        raise

def load_csv(
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    **kwargs
) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        file_path: Path to CSV file
        encoding: File encoding
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        Loaded data as DataFrame
    """
    try:
        return pd.read_csv(
            file_path,
            encoding=encoding,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {str(e)}")
        raise

def save_csv(
    data: pd.DataFrame,
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    index: bool = False,
    **kwargs
) -> None:
    """
    Save data to CSV file.
    
    Args:
        data: Data to save
        file_path: Path to output file
        encoding: File encoding
        index: Whether to save index
        **kwargs: Additional arguments for pd.to_csv
    """
    try:
        data.to_csv(
            file_path,
            encoding=encoding,
            index=index,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Error saving CSV file {file_path}: {str(e)}")
        raise

def load_stop_words(
    file_path: Union[str, Path],
    encoding: str = 'utf-8'
) -> Set[str]:
    """
    Load stop words from file.
    
    Args:
        file_path: Path to stop words file
        encoding: File encoding
        
    Returns:
        Set of stop words
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return {
                line.strip().lower()
                for line in f
                if line.strip()
            }
    except Exception as e:
        logger.error(f"Error loading stop words file {file_path}: {str(e)}")
        raise

def save_stop_words(
    stop_words: Set[str],
    file_path: Union[str, Path],
    encoding: str = 'utf-8'
) -> None:
    """
    Save stop words to file.
    
    Args:
        stop_words: Set of stop words
        file_path: Path to output file
        encoding: File encoding
    """
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            for word in sorted(stop_words):
                f.write(f"{word}\n")
    except Exception as e:
        logger.error(f"Error saving stop words file {file_path}: {str(e)}")
        raise

def load_entity_rules(
    file_path: Union[str, Path],
    encoding: str = 'utf-8'
) -> Dict[str, List[str]]:
    """
    Load entity rules from file.
    
    Args:
        file_path: Path to entity rules file
        encoding: File encoding
        
    Returns:
        Dictionary of (entity_type, patterns) pairs
    """
    try:
        rules = {}
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                entity_type = row.get('entity_type', '').strip()
                pattern = row.get('pattern', '').strip()
                if entity_type and pattern:
                    if entity_type not in rules:
                        rules[entity_type] = []
                    rules[entity_type].append(pattern)
        return rules
    except Exception as e:
        logger.error(f"Error loading entity rules file {file_path}: {str(e)}")
        raise

def save_entity_rules(
    rules: Dict[str, List[str]],
    file_path: Union[str, Path],
    encoding: str = 'utf-8'
) -> None:
    """
    Save entity rules to file.
    
    Args:
        rules: Dictionary of (entity_type, patterns) pairs
        file_path: Path to output file
        encoding: File encoding
    """
    try:
        with open(file_path, 'w', encoding=encoding, newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['entity_type', 'pattern']
            )
            writer.writeheader()
            for entity_type, patterns in rules.items():
                for pattern in patterns:
                    writer.writerow({
                        'entity_type': entity_type,
                        'pattern': pattern
                    })
    except Exception as e:
        logger.error(f"Error saving entity rules file {file_path}: {str(e)}")
        raise

def merge_dataframes(
    dfs: List[pd.DataFrame],
    on: Optional[Union[str, List[str]]] = None,
    how: str = 'outer',
    **kwargs
) -> pd.DataFrame:
    """
    Merge multiple DataFrames.
    
    Args:
        dfs: List of DataFrames to merge
        on: Column(s) to merge on
        how: Merge method
        **kwargs: Additional arguments for pd.merge
        
    Returns:
        Merged DataFrame
    """
    if not dfs:
        return pd.DataFrame()
    
    result = dfs[0]
    for df in dfs[1:]:
        result = pd.merge(
            result,
            df,
            on=on,
            how=how,
            **kwargs
        )
    
    return result

def filter_dataframe(
    df: pd.DataFrame,
    conditions: Dict[str, Any],
    operator: str = 'and'
) -> pd.DataFrame:
    """
    Filter DataFrame using multiple conditions.
    
    Args:
        df: Input DataFrame
        conditions: Dictionary of (column, value) pairs
        operator: Logical operator ('and' or 'or')
        
    Returns:
        Filtered DataFrame
    """
    if not conditions:
        return df
    
    # Build filter masks
    masks = []
    for column, value in conditions.items():
        if isinstance(value, (list, tuple, set)):
            mask = df[column].isin(value)
        else:
            mask = df[column] == value
        masks.append(mask)
    
    # Combine masks
    if operator == 'and':
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = final_mask & mask
    elif operator == 'or':
        final_mask = masks[0]
        for mask in masks[1:]:
            final_mask = final_mask | mask
    else:
        raise ValueError(f"Unknown operator: {operator}")
    
    return df[final_mask]

def aggregate_dataframe(
    df: pd.DataFrame,
    group_by: Union[str, List[str]],
    aggregations: Dict[str, Union[str, List[str]]],
    **kwargs
) -> pd.DataFrame:
    """
    Aggregate DataFrame using multiple aggregations.
    
    Args:
        df: Input DataFrame
        group_by: Column(s) to group by
        aggregations: Dictionary of (column, agg_func) pairs
        **kwargs: Additional arguments for pd.groupby
        
    Returns:
        Aggregated DataFrame
    """
    if not isinstance(group_by, list):
        group_by = [group_by]
    
    # Build aggregation dictionary
    agg_dict = {}
    for column, agg_func in aggregations.items():
        if isinstance(agg_func, (list, tuple)):
            agg_dict[column] = agg_func
        else:
            agg_dict[column] = [agg_func]
    
    # Perform aggregation
    return df.groupby(
        group_by,
        **kwargs
    ).agg(agg_dict).reset_index()

def create_backup(
    file_path: Union[str, Path],
    backup_dir: Optional[Union[str, Path]] = None,
    suffix: str = '.bak'
) -> Path:
    """
    Create backup of file.
    
    Args:
        file_path: Path to file
        backup_dir: Backup directory
        suffix: Backup file suffix
        
    Returns:
        Path to backup file
    """
    file_path = Path(file_path)
    
    # Create backup directory
    if backup_dir:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    else:
        backup_dir = file_path.parent
    
    # Generate backup filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"{file_path.stem}_{timestamp}{suffix}"
    backup_path = backup_dir / backup_name
    
    # Copy file
    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Error creating backup of {file_path}: {str(e)}")
        raise

def restore_backup(
    backup_path: Union[str, Path],
    target_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Restore file from backup.
    
    Args:
        backup_path: Path to backup file
        target_path: Path to restore to
        
    Returns:
        Path to restored file
    """
    backup_path = Path(backup_path)
    
    # Determine target path
    if target_path:
        target_path = Path(target_path)
    else:
        # Remove timestamp and suffix from backup name
        name_parts = backup_path.stem.split('_')
        if len(name_parts) > 1:
            target_name = '_'.join(name_parts[:-1])
        else:
            target_name = name_parts[0]
        target_path = backup_path.parent / f"{target_name}{backup_path.suffix}"
    
    # Restore file
    try:
        import shutil
        shutil.copy2(backup_path, target_path)
        logger.info(f"Restored backup to: {target_path}")
        return target_path
    except Exception as e:
        logger.error(f"Error restoring backup {backup_path}: {str(e)}")
        raise 