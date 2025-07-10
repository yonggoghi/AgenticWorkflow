"""
Text processing utilities for MMS extraction.
"""
import re
import json
import ast
from typing import List, Dict, Any, Union, Tuple
import pandas as pd
from difflib import SequenceMatcher


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters that don't affect fine-tuning.
    Preserves important structural elements like quotes, brackets, and JSON syntax.
    Specifically handles Korean text (Hangul) properly.
    
    Args:
        text: The input text to clean
        
    Returns:
        Cleaned text ready for fine-tuning
    """
    if not isinstance(text, str):
        return ""
    
    # Preserve JSON structural elements with placeholders
    placeholders = {
        '"': "DQUOTE_TOKEN",
        "'": "SQUOTE_TOKEN",
        "{": "OCURLY_TOKEN",
        "}": "CCURLY_TOKEN",
        "[": "OSQUARE_TOKEN",
        "]": "CSQUARE_TOKEN",
        ":": "COLON_TOKEN",
        ",": "COMMA_TOKEN"
    }
    
    for char, placeholder in placeholders.items():
        text = text.replace(char, placeholder)
    
    # Remove control characters (except newlines, carriage returns, and tabs)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize all types of newlines to \n
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # Remove zero-width characters and other invisible unicode
    text = re.sub(r'[\u200B-\u200D\uFEFF\u00A0]', '', text)
    
    # Keep Korean characters and other useful character sets
    allowed_chars_pattern = r'[^\x00-\x7F\u0080-\u00FF\u0100-\u024F\u0370-\u03FF\u0400-\u04FF' + \
                           r'\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\u3000-\u303F' + \
                           r'\uAC00-\uD7A3\uFF00-\uFFEF\u4E00-\u9FFF\n\r\t ]'
    text = re.sub(allowed_chars_pattern, '', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    # Restore original JSON structural elements
    for char, placeholder in placeholders.items():
        text = text.replace(placeholder, char)
    
    # Fix common JSON syntax issues
    text = re.sub(r'"\s+:', r'":', text)
    text = re.sub(r',\s*]', r']', text)
    text = re.sub(r',\s*}', r'}', text)
    
    return text


def preprocess_text(text: str) -> str:
    """
    Preprocess text by converting special characters to spaces.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert special characters to spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing spaces
    return text.strip()


def remove_urls(text: str) -> str:
    """
    Remove URLs from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with URLs removed
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)


def remove_custom_pattern(text: str, keyword: str = "바로가기") -> str:
    """
    Remove custom pattern from text.
    
    Args:
        text: Input text
        keyword: Keyword to match and remove
        
    Returns:
        Text with custom pattern removed
    """
    escaped_keyword = re.escape(keyword)
    pattern = re.compile(r'.*? ' + escaped_keyword)
    return pattern.sub('', text)


def filter_specific_terms(strings: List[str]) -> List[str]:
    """
    Select the most comprehensive string from a list of overlapping strings.
    
    Args:
        strings: List of strings to filter
        
    Returns:
        List of most comprehensive strings
    """
    if not strings:
        return []
    
    unique_strings = list(set(strings))
    unique_strings.sort(key=len, reverse=True)
    
    result = []
    for current in unique_strings:
        is_contained = any(current in existing for existing in result)
        
        if not is_contained:
            result = [r for r in result if r not in current]
            result.append(current)
    
    return result


def clean_segment(segment: str) -> str:
    """
    Clean a quoted segment by removing inner quotes.
    
    Args:
        segment: Input segment
        
    Returns:
        Cleaned segment
    """
    segment = segment.strip()
    if len(segment) >= 2 and segment[0] in ['"', "'"] and segment[-1] == segment[0]:
        q = segment[0]
        inner = segment[1:-1].replace(q, '')
        return q + inner + q
    return segment


def split_key_value(text: str) -> Tuple[str, str]:
    """
    Split text into key and value based on first colon outside quotes.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (key, value)
    """
    in_quote = False
    quote_char = ''
    
    for i, char in enumerate(text):
        if char in ['"', "'"]:
            if in_quote:
                if char == quote_char:
                    in_quote = False
                    quote_char = ''
            else:
                in_quote = True
                quote_char = char
        elif char == ':' and not in_quote:
            return text[:i], text[i+1:]
    
    return text, ''


def split_outside_quotes(text: str, delimiter: str = ',') -> List[str]:
    """
    Split text on delimiter only if outside quoted segments.
    
    Args:
        text: Input text
        delimiter: Delimiter to split on
        
    Returns:
        List of split parts
    """
    parts = []
    current = []
    in_quote = False
    quote_char = ''
    
    for char in text:
        if char in ['"', "'"]:
            if in_quote:
                if char == quote_char:
                    in_quote = False
                    quote_char = ''
            else:
                in_quote = True
                quote_char = char
            current.append(char)
        elif char == delimiter and not in_quote:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    
    if current:
        parts.append(''.join(current).strip())
    
    return parts


def clean_ill_structured_json(text: str) -> str:
    """
    Clean ill-structured JSON-like text.
    
    Args:
        text: Input JSON-like text
        
    Returns:
        Cleaned text
    """
    parts = split_outside_quotes(text, delimiter=',')
    
    cleaned_parts = []
    for part in parts:
        key, value = split_key_value(part)
        key_clean = clean_segment(key)
        value_clean = clean_segment(value) if value.strip() != "" else ""
        
        if value_clean:
            cleaned_parts.append(f"{key_clean}: {value_clean}")
        else:
            cleaned_parts.append(key_clean)
    
    return ', '.join(cleaned_parts)


def repair_json(broken_json):
    """
    More advanced JSON repair that handles edge cases better
    """
    json_str = broken_json
    
    # Fix unquoted keys
    json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1 "\2":', json_str)
    
    # Fix unquoted values more carefully
    # Split on quotes to avoid modifying content inside strings
    parts = json_str.split('"')
    
    for i in range(0, len(parts), 2):  # Only process parts outside quotes (even indices)
        # Fix unquoted values in this part
        parts[i] = re.sub(r':\s*([a-zA-Z0-9_]+)(?=\s*[,\]\}])', r': "\1"', parts[i])
    
    json_str = '"'.join(parts)
    
    # Fix trailing commas
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    return json_str


def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """
    Extract JSON objects from text.
    
    Args:
        text: Input text containing JSON
        
    Returns:
        List of extracted JSON objects
    """
    pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
    
    result = []
    for match in re.finditer(pattern, text):
        potential_json = match.group(0)
        try:
            json_obj = ast.literal_eval(clean_ill_structured_json(repair_json(potential_json)))
            result.append(json_obj)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            pass
    
    return result


def extract_between(text: str, start_marker: str, end_marker: str) -> str:
    """
    Extract text between markers.
    
    Args:
        text: Input text
        start_marker: Start marker
        end_marker: End marker
        
    Returns:
        Extracted text or None if not found
    """
    start_index = text.find(start_marker)
    if start_index == -1:
        return None
    
    start_index += len(start_marker)
    end_index = text.find(end_marker, start_index)
    if end_index == -1:
        return None
    
    return text[start_index:end_index]


def extract_content(text: str, tag_name: str) -> List[str]:
    """
    Extract content between XML-like tags.
    
    Args:
        text: Input text
        tag_name: Tag name to extract
        
    Returns:
        List of extracted content
    """
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def convert_df_to_json_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert DataFrame to specific JSON structure.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of JSON objects
    """
    result = []
    
    grouped = df.groupby('item_name_in_msg')
    
    for item_name_in_msg, group in grouped:
        item_dict = {
            'item_name_in_msg': item_name_in_msg,
            'item_in_voca': []
        }
        
        item_nm_groups = group.groupby('item_nm')
        
        for item_nm, item_group in item_nm_groups:
            item_ids = list(item_group['item_id'].unique())
            
            voca_item = {
                'item_nm': item_nm,
                'item_id': item_ids
            }
            item_dict['item_in_voca'].append(voca_item)
        
        result.append(item_dict)
    
    return result


def dataframe_to_markdown_prompt(df: pd.DataFrame, max_rows: int = None) -> str:
    """
    Convert DataFrame to markdown format for prompts.
    
    Args:
        df: Input DataFrame
        max_rows: Maximum number of rows to display
        
    Returns:
        Markdown formatted string
    """
    if max_rows is not None and len(df) > max_rows:
        display_df = df.head(max_rows)
        truncation_note = f"\n[Note: Only showing first {max_rows} of {len(df)} rows]"
    else:
        display_df = df
        truncation_note = ""
    
    df_markdown = display_df.to_markdown()
    
    return f"\n\n{df_markdown}{truncation_note}\n\n"


def clean_bad_text(text: str) -> str:
    """
    Clean bad text by removing URLs, emails, and unwanted characters.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs and emails
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Keep Korean, alphanumeric, spaces, and specific punctuation
    text = re.sub(r'[^\uAC00-\uD7A3\u1100-\u11FF\w\s\.\?!,]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def replace_special_chars_comprehensive(text: str) -> str:
    """
    More comprehensive: Handle various types of special characters.
    
    Args:
        text: Input text
        
    Returns:
        Text with special characters replaced
    """
    # Replace common punctuation with space
    punctuation_pattern = r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>?/~`]'
    text = re.sub(punctuation_pattern, ' ', text)
    
    # Replace other special symbols
    symbol_pattern = r'[₩＄￦※◆▲▼◀▶★☆♪♫♬♩♭♯]'
    text = re.sub(symbol_pattern, ' ', text)
    
    # Replace various dashes and quotes
    dash_quote_pattern = r'[—–‒―""''‚„‹›«»]'
    text = re.sub(dash_quote_pattern, ' ', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def select_most_comprehensive(strings: List[str]) -> List[str]:
    """
    Select the most comprehensive string from a list of overlapping strings.
    Returns the longest string that contains other strings as substrings.
    
    Args:
        strings: List of strings to filter
        
    Returns:
        List of most comprehensive strings (usually one, but could be multiple if no containment)
    """
    if not strings:
        return []
    
    # Remove duplicates and sort by length (longest first)
    unique_strings = list(set(strings))
    unique_strings.sort(key=len, reverse=True)
    
    result = []
    
    for current in unique_strings:
        # Check if current string contains any of the strings already in result
        is_contained = any(current in existing for existing in result)
        
        # Check if current string contains other strings not yet in result
        contains_others = any(other in current for other in unique_strings if other != current and other not in result)
        
        # If current is not contained by existing results and either:
        # 1. It contains other strings, or 
        # 2. No strings contain each other (keep all unique)
        if not is_contained:
            # Remove any strings from result that are contained in current
            result = [r for r in result if r not in current]
            result.append(current)
    
    return result


def remove_control_characters(text: str) -> str:
    """
    Remove control characters from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with control characters removed
    """
    if isinstance(text, str):
        # Remove control characters except commonly used whitespace
        return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    return text


def replace_strings(text: str, replacements: Dict[str, str]) -> str:
    """
    Replace multiple strings in text.
    
    Args:
        text: Input text
        replacements: Dictionary of {old: new} replacements
        
    Returns:
        Text with replacements applied
    """
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def is_list_of_dicts(var: Any) -> bool:
    """
    Check if variable is a list of dictionaries.
    
    Args:
        var: Variable to check
        
    Returns:
        True if variable is a list of dictionaries
    """
    # Check if the variable is a list
    if not isinstance(var, list):
        return False
    
    # Check if the list is not empty and all elements are dictionaries
    if not var:  # Empty list
        return False
        
    # Check that all elements are dictionaries
    return all(isinstance(item, dict) for item in var)


def remove_duplicate_dicts(dict_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate dictionaries from a list.
    
    Args:
        dict_list: List of dictionaries
        
    Returns:
        List with duplicate dictionaries removed
    """
    result = []
    seen = set()
    for d in dict_list:
        # Convert dictionary to a hashable tuple of items
        t = tuple(sorted(d.items()))
        if t not in seen:
            seen.add(t)
            result.append(d)
    return result


def convert_to_custom_format(json_items: List[Dict[str, Any]]) -> str:
    """
    Convert JSON items to custom format string.
    
    Args:
        json_items: List of JSON item dictionaries
        
    Returns:
        Custom formatted string
    """
    custom_format = []
    
    for item in json_items:
        item_name = item.get("item_name_in_message", "")
        item_id = item.get("item_id", "")
        category = item.get("category", "")
        
        # Create custom format for each item
        custom_line = f"[Item Name] {item_name} [Item ID] {item_id} [Item Category] {category}"
        custom_format.append(custom_line)
    
    return "\n".join(custom_format) 