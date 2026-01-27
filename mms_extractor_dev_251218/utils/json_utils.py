import json
import re
import ast
from typing import List, Dict, Any, Union
import pandas as pd

def dataframe_to_markdown_prompt(df, max_rows=None):
    """DataFrame을 마크다운 형식의 프롬프트로 변환"""
    if max_rows is not None and len(df) > max_rows:
        display_df = df.head(max_rows)
        truncation_note = f"\n[Note: Only showing first {max_rows} of {len(df)} rows]"
    else:
        display_df = df
        truncation_note = ""
    df_markdown = display_df.to_markdown()
    prompt = f"\n\n    {df_markdown}\n    {truncation_note}\n\n    "
    return prompt

def escape_quotes_in_value(value):
    """값 내부의 따옴표를 이스케이프하거나 제거"""
    value = value.strip()
    if not value:
        return value
    
    # 값이 따옴표로 시작하는 경우
    if value[0] in ['"', "'"]:
        quote_char = value[0]
        # 닫는 따옴표 찾기
        end_idx = -1
        for i in range(1, len(value)):
            if value[i] == quote_char:
                end_idx = i
                break
        
        if end_idx > 0:
            # 따옴표로 감싸진 부분과 나머지 분리
            quoted_part = value[1:end_idx]
            remaining = value[end_idx+1:].strip()
            
            # 나머지가 있으면 합치기 (따옴표 내부의 내용으로 통합)
            if remaining:
                combined = quoted_part + ' ' + remaining
                return quote_char + combined + quote_char
            else:
                return quote_char + quoted_part + quote_char
        else:
            # 닫는 따옴표가 없으면 전체를 따옴표로 감싸기
            return quote_char + value[1:] + quote_char
    
    # 따옴표로 시작하지 않으면 전체를 따옴표로 감싸기
    return '"' + value + '"'

def split_key_value(text):
    """따옴표 외부의 첫 번째 콜론을 기준으로 키-값 분리 (개선된 버전)"""
    in_quote = False
    quote_char = ''
    escape_next = False
    
    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\':
            escape_next = True
            continue
            
        if char in ['"', "'"]:
            if in_quote:
                if char == quote_char:
                    in_quote = False
                    quote_char = ''
            else:
                in_quote = True
                quote_char = char
        elif char == ':' and not in_quote:
            return text[:i].strip(), text[i+1:].strip()
    
    return text.strip(), ''

def split_outside_quotes(text, delimiter=','):
    """따옴표 외부의 구분자로만 텍스트 분리 (개선된 버전)"""
    parts = []
    current = []
    in_quote = False
    quote_char = ''
    escape_next = False
    
    for char in text:
        if escape_next:
            current.append(char)
            escape_next = False
            continue
            
        if char == '\\':
            current.append(char)
            escape_next = True
            continue
            
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
            if current:
                parts.append(''.join(current).strip())
                current = []
        else:
            current.append(char)
    
    if current:
        parts.append(''.join(current).strip())
    
    return parts

def clean_ill_structured_json(text):
    """잘못 구조화된 JSON 형식의 텍스트 정리 (개선된 버전)"""
    # 중괄호 제거
    text = text.strip()
    if text.startswith('{'):
        text = text[1:]
    if text.endswith('}'):
        text = text[:-1]
    
    parts = split_outside_quotes(text.strip(), delimiter=',')
    cleaned_parts = []
    
    for part in parts:
        if not part.strip():
            continue
            
        key, value = split_key_value(part)
        
        if not value:
            continue
        
        # 키 정리 (따옴표 추가)
        key_clean = key.strip()
        if not (key_clean.startswith('"') and key_clean.endswith('"')):
            if key_clean.startswith('"') or key_clean.startswith("'"):
                key_clean = key_clean[1:]
            if key_clean.endswith('"') or key_clean.endswith("'"):
                key_clean = key_clean[:-1]
            key_clean = '"' + key_clean + '"'
        
        # 값 정리 (따옴표 처리)
        value_clean = escape_quotes_in_value(value)
        
        cleaned_parts.append(f"{key_clean}: {value_clean}")
    
    return '{' + ', '.join(cleaned_parts) + '}'

def repair_json(broken_json):
    """손상된 JSON 문자열 복구 (개선된 버전)"""
    json_str = broken_json.strip()
    
    # 후행 쉼표 제거
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    # 연속된 쉼표 제거
    json_str = re.sub(r',\s*,', ',', json_str)
    
    return json_str

def extract_json_objects(text):
    """텍스트에서 JSON 객체 추출 (개선된 버전)"""
    pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
    result = []
    
    for match in re.finditer(pattern, text):
        potential_json = match.group(0)
        
        # 여러 방법으로 시도
        attempts = [
            lambda: json.loads(potential_json),
            lambda: json.loads(repair_json(potential_json)),
            lambda: json.loads(clean_ill_structured_json(repair_json(potential_json))),
            lambda: ast.literal_eval(clean_ill_structured_json(repair_json(potential_json))),
        ]
        
        for attempt in attempts:
            try:
                json_obj = attempt()
                if isinstance(json_obj, dict):
                    result.append(json_obj)
                    break
            except (json.JSONDecodeError, SyntaxError, ValueError, TypeError):
                continue
    
    return result

def convert_df_to_json_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame을 특정 JSON 구조로 변환 (item_nm 기준 플랫 구조)"""
    result = []
    # item_nm으로 그룹화
    grouped = df.groupby('item_nm')
    
    for item_nm, group in grouped:
        # item_id 리스트 (중복 제거)
        item_ids = list(group['item_id'].unique())
        # item_name_in_msg 리스트 (중복 제거)
        item_names_in_msg = list(group['item_name_in_msg'].unique())
        
        item_dict = {
            'item_nm': item_nm,
            'item_id': item_ids,
            'item_name_in_msg': item_names_in_msg
        }
        result.append(item_dict)
    
    return result
