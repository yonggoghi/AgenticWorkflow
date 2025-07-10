from concurrent.futures import ThreadPoolExecutor
import time
import pandas as pd
pd.set_option('display.max_colwidth', 500)
def main():
    from openai import OpenAI
    # llm_api_key = config.CUSTOM_API_KEY"https://api.platform.a15t.com/v1"
    client = OpenAI(
        api_key = llm_api_key,
        base_url = llm_api_url
    )
    # from langchain.chat_models import ChatOpenAI
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain.schema import AIMessage, HumanMessage, SystemMessage
    import pandas as pd
    def ChatAnthropicSKT(model="skt/claude-3-5-sonnet-20241022", max_tokens=100):
        llm_api_key = config.CUSTOM_API_KEY"https://api.platform.a15t.com/v1"
        # llm_api_url = "https://43.203.77.11:443/v1"
        # model = "anthropic/claude-3-5-sonnet-20240620"
        model = ChatOpenAI(
            temperature=0,
            openai_api_key=llm_api_key,
            openai_api_base=llm_api_url,
            model=model,
            max_tokens=max_tokens
            )
        return model
    # Import configuration
    from config import config
    llm_cld35 = ChatAnthropicSKT()
    llm_cld37 = ChatAnthropic(
        api_key=config.ANTHROPIC_API_KEY,
        model="claude-3-7-sonnet-20250219",
        max_tokens=3000
    )
    llm_chat = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            openai_api_key=config.OPENAI_API_KEY,
            max_tokens=2000,
    )
    from typing import List, Tuple, Union, Dict, Any
    import ast
    import re
    import json
    import glob
    def dataframe_to_markdown_prompt(df, max_rows=None):
        # Limit rows if needed
        if max_rows is not None and len(df) > max_rows:
            display_df = df.head(max_rows)
            truncation_note = f"\n[Note: Only showing first {max_rows} of {len(df)} rows]"
        else:
            display_df = df
            truncation_note = ""
        # Convert to markdown
        df_markdown = display_df.to_markdown()
        prompt = f"""
        {df_markdown}
        {truncation_note}
        """
        return prompt
    def replace_strings(text, replacements):
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    def clean_segment(segment):
        """
        Given a segment that is expected to be quoted (i.e. begins and ends with
        the same single or double quote), remove any occurrences of that quote
        from the inner content.
        For example, if segment is:
            "에이닷 T 멤버십 쿠폰함에 "에이닷은통화요약된닷" 입력"
        then the outer quotes are preserved but the inner double quotes are removed.
        """
        segment = segment.strip()
        if len(segment) >= 2 and segment[0] in ['"', "'"] and segment[-1] == segment[0]:
            q = segment[0]
            # Remove inner occurrences of the quote character.
            inner = segment[1:-1].replace(q, '')
            return q + inner + q
        return segment
    def split_key_value(text):
        """
        Splits text into key and value based on the first colon that appears
        outside any quoted region.
        If no colon is found outside quotes, the value will be returned empty.
        """
        in_quote = False
        quote_char = ''
        for i, char in enumerate(text):
            if char in ['"', "'"]:
                # Toggle quote state (assumes well-formed starting/ending quotes for each token)
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
    def split_outside_quotes(text, delimiter=','):
        """
        Splits the input text on the given delimiter (default comma) but only
        if the delimiter occurs outside of quoted segments.
        Returns a list of parts.
        """
        parts = []
        current = []
        in_quote = False
        quote_char = ''
        for char in text:
            if char in ['"', "'"]:
                # When encountering a quote, toggle our state
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
    def clean_ill_structured_json(text):
        """
        Given a string that is intended to represent a JSON-like structure
        but may be ill-formed (for example, it might contain nested quotes that
        break standard JSON rules), attempt to “clean” it by processing each
        key–value pair.
        The function uses the following heuristics:
        1. Split the input text into comma-separated parts (only splitting
            when the comma is not inside a quoted string).
        2. For each part, split on the first colon (that is outside quotes) to separate key and value.
        3. For any segment that begins and ends with a quote, remove any inner occurrences
            of that same quote.
        4. Rejoin the cleaned key and value.
        Note: This approach does not build a fully robust JSON parser. For very complex
            or deeply nested ill-structured inputs further refinement would be needed.
        """
        # First, split the text by commas outside of quotes.
        parts = split_outside_quotes(text, delimiter=',')
        cleaned_parts = []
        for part in parts:
            # Try to split into key and value on the first colon not inside quotes.
            key, value = split_key_value(part)
            key_clean = clean_segment(key)
            value_clean = clean_segment(value) if value.strip() != "" else ""
            if value_clean:
                cleaned_parts.append(f"{key_clean}: {value_clean}")
            else:
                cleaned_parts.append(key_clean)
        # Rejoin the cleaned parts with commas (or you can use another format if desired)
        return ', '.join(cleaned_parts)
    def repair_json(broken_json):
        # json_str = broken_json.replace("'",'"')
        # Fix unquoted values (like NI00001863)
        json_str = re.sub(r':\s*([a-zA-Z0-9_]+)(\s*[,}])', r': "\1"\2', broken_json)
        # Fix unquoted keys
        json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+):', r'\1 "\2":', json_str)
        # Fix trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        return json_str
    def extract_json_objects(text):
        # More sophisticated pattern that tries to match proper JSON syntax
        pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
        result = []
        for match in re.finditer(pattern, text):
            potential_json = match.group(0)
            try:
                # Try to parse and validate
                # json_obj = json.loads(repair_json(potential_json))
                json_obj = ast.literal_eval(clean_ill_structured_json(repair_json(potential_json)))
                result.append(json_obj)
            except json.JSONDecodeError:
                # Not valid JSON, skip
                pass
        return result
    def extract_between(text, start_marker, end_marker):
        start_index = text.find(start_marker)
        if start_index == -1:
            return None
        start_index += len(start_marker)
        end_index = text.find(end_marker, start_index)
        if end_index == -1:
            return None
        return text[start_index:end_index]
    def extract_content(text: str, tag_name: str) -> List[str]:
        pattern = f'<{tag_name}>(.*?)</{tag_name}>'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches
    def clean_bad_text(text):
        import re
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
    def clean_text(text):
        """
        Cleans text by removing special characters that don't affect fine-tuning.
        Preserves important structural elements like quotes, brackets, and JSON syntax.
        Specifically handles Korean text (Hangul) properly.
        Args:
            text (str): The input text to clean
        Returns:
            str: Cleaned text ready for fine-tuning
        """
        import re
        # Preserve the basic structure by temporarily replacing important characters
        # with placeholder tokens that won't be affected by cleanup
        # Step 1: Temporarily replace JSON structural elements
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
        # Step 2: Remove problematic characters
        # Remove control characters (except newlines, carriage returns, and tabs which can be meaningful)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # Normalize all types of newlines to \n
        text = re.sub(r'\r\n|\r', '\n', text)
        # Remove zero-width characters and other invisible unicode
        text = re.sub(r'[\u200B-\u200D\uFEFF\u00A0]', '', text)
        # MODIFIED: Keep Korean characters (Hangul) along with other useful character sets
        # This regex keeps:
        # - ASCII (Basic Latin): \x00-\x7F
        # - Latin-1 Supplement: \u0080-\u00FF
        # - Latin Extended A & B: \u0100-\u017F\u0180-\u024F
        # - Greek and Coptic: \u0370-\u03FF
        # - Cyrillic: \u0400-\u04FF
        # - Korean Hangul Syllables: \uAC00-\uD7A3
        # - Hangul Jamo (Korean alphabet): \u1100-\u11FF
        # - Hangul Jamo Extended-A: \u3130-\u318F
        # - Hangul Jamo Extended-B: \uA960-\uA97F
        # - Hangul Compatibility Jamo: \u3130-\u318F
        # - CJK symbols and punctuation: \u3000-\u303F
        # - Full-width forms (often used with CJK): \uFF00-\uFFEF
        # - CJK Unified Ideographs (Basic common Chinese/Japanese characters): \u4E00-\u9FFF
        # Instead of removing characters, we'll define which ones to keep
        allowed_chars_pattern = r'[^\x00-\x7F\u0080-\u00FF\u0100-\u024F\u0370-\u03FF\u0400-\u04FF' + \
                            r'\u1100-\u11FF\u3130-\u318F\uA960-\uA97F\u3000-\u303F' + \
                            r'\uAC00-\uD7A3\uFF00-\uFFEF\u4E00-\u9FFF\n\r\t ]'
        text = re.sub(allowed_chars_pattern, '', text)
        # Step 3: Normalize whitespace (but preserve deliberate line breaks)
        text = re.sub(r'[ \t]+', ' ', text)  # Convert multiple spaces/tabs to single space
        # First ensure all newlines are standardized
        text = re.sub(r'\r\n|\r', '\n', text)  # Convert all newline variants to \n
        # Then normalize multiple blank lines to at most two
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Convert multiple newlines to at most two
        # Step 4: Restore original JSON structural elements
        for char, placeholder in placeholders.items():
            text = text.replace(placeholder, char)
        # Step 5: Fix common JSON syntax issues that might remain
        # Fix spaces between quotes and colons in JSON
        text = re.sub(r'"\s+:', r'":', text)
        # Fix trailing commas in arrays
        text = re.sub(r',\s*]', r']', text)
        # Fix trailing commas in objects
        text = re.sub(r',\s*}', r'}', text)
        return text
    def remove_control_characters(text):
        if isinstance(text, str):
            # Remove control characters except commonly used whitespace
            return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        return text
    import openai
    from langchain.chains import RetrievalQA
    from langchain.llms.openai import OpenAIChat  # For compatibility with newer setup
    # Create a custom LLM class that uses the OpenAI client directly
    class CustomOpenAI:
        def __init__(self, model="skt/a.x-3-lg"):
            self.model = model
        def __call__(self, prompt):
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content
    # Create a simple retrieval function
    def get_relevant_context(query, vectorstore, topk=5):
        docs = vectorstore.similarity_search(query, k=topk)
        context = "\n\n".join([doc.page_content for doc in docs])
        titles = ", ".join(set([doc.metadata['title'] for doc in docs if 'title' in doc.metadata.keys()]))
        return {'title':titles, 'context':context}
    # Create a function to combine everything
    def answer_question(query, vectorstore):
        # Get relevant context
        context = get_relevant_context(query, vectorstore)
        # Create combined prompt
        prompt = f"Answer the following question based on the provided context:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        # Use OpenAI directly
        custom_llm = CustomOpenAI()  # Or your preferred model
        response = custom_llm(prompt)
        return response
    def is_list_of_dicts(var):
        # Check if the variable is a list
        if not isinstance(var, list):
            return False
        # Check if the list is not empty and all elements are dictionaries
        if not var:  # Empty list
            return False
        # Check that all elements are dictionaries
        return all(isinstance(item, dict) for item in var)
    def remove_duplicate_dicts(dict_list):
        result = []
        seen = set()
        for d in dict_list:
            # Convert dictionary to a hashable tuple of items
            t = tuple(sorted(d.items()))
            if t not in seen:
                seen.add(t)
                result.append(d)
        return result
    def convert_to_custom_format(json_items):
        custom_format = []
        for item in json_items:
            item_name = item.get("item_name_in_message", "")
            item_id = item.get("item_id", "")
            category = item.get("category", "")
            # Create custom format for each item
            custom_line = f"[Item Name] {item_name} [Item ID] {item_id} [Item Category] {category}"
            custom_format.append(custom_line)
        return "\n".join(custom_format)
    def remove_urls(text):
        # Regular expression pattern to match URLs
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        # Replace URLs with an empty string
        return url_pattern.sub('', text)
    def remove_custom_pattern(text, keyword="바로가기"):
        # Create a pattern that matches any text followed by the specified keyword
        # We escape the keyword to handle any special regex characters it might contain
        escaped_keyword = re.escape(keyword)
        pattern = re.compile(r'.*? ' + escaped_keyword)
        # Replace the matched pattern with an empty string
        return pattern.sub('', text)
    from rapidfuzz import fuzz, process
    import re
    class KoreanEntityMatcher:
        def __init__(self, min_similarity=70, ngram_size=2, min_entity_length=2, token_similarity=True):
            self.min_similarity = min_similarity
            self.ngram_size = ngram_size
            self.min_entity_length = min_entity_length
            self.token_similarity = token_similarity  # 토큰 단위 유사도 비교 옵션 추가
            self.entities = []
            self.entity_data = {}
        def build_from_list(self, entities):
            """Build entity index from a list of entities"""
            self.entities = []
            self.entity_data = {}
            for i, entity in enumerate(entities):
                if isinstance(entity, tuple) and len(entity) == 2:
                    entity_name, data = entity
                    self.entities.append(entity_name)
                    self.entity_data[entity_name] = data
                else:
                    self.entities.append(entity)
                    self.entity_data[entity] = {'id': i, 'entity': entity}
            # 각 엔티티의 정규화된 형태를 저장 (검색 최적화)
            self.normalized_entities = {}
            for entity in self.entities:
                normalized = self._normalize_text(entity)
                self.normalized_entities[normalized] = entity
            # Create n-gram index for faster candidate selection
            self._build_ngram_index(n=self.ngram_size)
        def _normalize_text(self, text):
            """텍스트 정규화 - 소문자 변환, 공백 제거 등"""
            # 소문자로 변환
            text = text.lower()
            # 연속된 공백을 하나로 통일
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        def _tokenize(self, text):
            """텍스트를 토큰으로 분리 (한글, 영문, 숫자 분리)"""
            # 한글, 영문, 숫자 토큰 추출
            tokens = re.findall(r'[가-힣]+|[a-z0-9]+', self._normalize_text(text))
            return tokens
        def _build_ngram_index(self, n=2):
            """Build n-gram index optimized for Korean characters"""
            self.ngram_index = {}
            for entity in self.entities:
                # Skip entities shorter than min_entity_length
                if len(entity) < self.min_entity_length:
                    continue
                # 정규화된 엔티티 사용
                normalized_entity = self._normalize_text(entity)
                # Create n-grams for the entity
                entity_chars = list(normalized_entity)  # Split into characters for proper Korean handling
                ngrams = []
                # Create character-level n-grams (better for Korean)
                for i in range(len(entity_chars) - n + 1):
                    ngram = ''.join(entity_chars[i:i+n])
                    ngrams.append(ngram)
                # Add entity to the index for each n-gram
                for ngram in ngrams:
                    if ngram not in self.ngram_index:
                        self.ngram_index[ngram] = set()
                    self.ngram_index[ngram].add(entity)
                # 토큰 기반 n-gram도 추가 (실험적)
                tokens = self._tokenize(normalized_entity)
                for token in tokens:
                    if len(token) >= n:
                        token_key = f"TOKEN:{token}"
                        if token_key not in self.ngram_index:
                            self.ngram_index[token_key] = set()
                        self.ngram_index[token_key].add(entity)
        def _get_candidates(self, text, n=None):
            """Get candidate entities based on n-gram overlap (optimized for Korean)"""
            if n is None:
                n = self.ngram_size
            # 텍스트 정규화
            normalized_text = self._normalize_text(text)
            # 정규화된 텍스트가 정확히 일치하는지 확인 (빠른 경로)
            if normalized_text in self.normalized_entities:
                entity = self.normalized_entities[normalized_text]
                return [(entity, float('inf'))]  # 정확한 일치는 무한대 점수로 표시
            text_chars = list(normalized_text)  # Split into characters for proper Korean handling
            text_ngrams = set()
            # Create character-level n-grams
            for i in range(len(text_chars) - n + 1):
                ngram = ''.join(text_chars[i:i+n])
                text_ngrams.add(ngram)
            # 토큰 기반 n-gram 추가
            tokens = self._tokenize(normalized_text)
            for token in tokens:
                if len(token) >= n:
                    text_ngrams.add(f"TOKEN:{token}")
            candidates = set()
            for ngram in text_ngrams:
                if ngram in self.ngram_index:
                    candidates.update(self.ngram_index[ngram])
            # Prioritize candidates with multiple n-gram matches
            candidate_scores = {}
            for candidate in candidates:
                candidate_normalized = self._normalize_text(candidate)
                candidate_chars = list(candidate_normalized)
                candidate_ngrams = set()
                # 문자 n-gram
                for i in range(len(candidate_chars) - n + 1):
                    ngram = ''.join(candidate_chars[i:i+n])
                    candidate_ngrams.add(ngram)
                # 토큰 기반 n-gram
                candidate_tokens = self._tokenize(candidate_normalized)
                for token in candidate_tokens:
                    if len(token) >= n:
                        candidate_ngrams.add(f"TOKEN:{token}")
                # n-gram 교집합 크기로 초기 점수 계산
                overlap = len(candidate_ngrams.intersection(text_ngrams))
                # 토큰 수준 유사도 보너스 점수 추가
                token_bonus = 0
                if self.token_similarity:
                    query_tokens = set(tokens)
                    cand_tokens = set(candidate_tokens)
                    # 공통 토큰 비율 계산
                    if query_tokens and cand_tokens:
                        common = query_tokens.intersection(cand_tokens)
                        token_bonus = len(common) * 2  # 토큰 일치에 높은 가중치 부여
                candidate_scores[candidate] = overlap + token_bonus
            # Return candidates sorted by n-gram overlap score
            return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        def _calculate_similarity(self, text, entity):
            """다양한 유사도 측정 방법을 결합하여 더 정확한 유사도 계산"""
            normalized_text = self._normalize_text(text)
            normalized_entity = self._normalize_text(entity)
            # 정확히 일치하면 100점 반환
            if normalized_text == normalized_entity:
                return 100
            # 기본 문자열 유사도 (fuzz.ratio)
            ratio_score = fuzz.ratio(normalized_text, normalized_entity)
            # 부분 문자열 체크 (한 문자열이 다른 문자열의 부분 문자열인 경우)
            partial_score = 0
            if normalized_text in normalized_entity:
                text_len = len(normalized_text)
                entity_len = len(normalized_entity)
                partial_score = (text_len / entity_len) * 100 if entity_len > 0 else 0
            elif normalized_entity in normalized_text:
                text_len = len(normalized_text)
                entity_len = len(normalized_entity)
                partial_score = (entity_len / text_len) * 100 if text_len > 0 else 0
            # 토큰 유사도 (토큰 단위로 비교)
            token_score = 0
            if self.token_similarity:
                text_tokens = set(self._tokenize(normalized_text))
                entity_tokens = set(self._tokenize(normalized_entity))
                if text_tokens and entity_tokens:
                    common_tokens = text_tokens.intersection(entity_tokens)
                    all_tokens = text_tokens.union(entity_tokens)
                    if all_tokens:
                        token_score = (len(common_tokens) / len(all_tokens)) * 100
            # 토큰 순서 무시 유사도 (fuzz.token_sort_ratio)
            token_sort_score = fuzz.token_sort_ratio(normalized_text, normalized_entity)
            # 토큰 집합 유사도 (fuzz.token_set_ratio)
            token_set_score = fuzz.token_set_ratio(normalized_text, normalized_entity)
            # 최종 유사도는 여러 점수의 가중 평균
            # 토큰 유사도에 높은 가중치를 부여하여 "T우주"와 "T 우주패스"의 매칭 향상
            final_score = (
                ratio_score * 0.3 +  # 기본 유사도
                max(partial_score, 0) * 0.1 +  # 부분 문자열 유사도
                token_score * 0.2 +  # 토큰 유사도
                token_sort_score * 0.2 +  # 토큰 순서 무시 유사도
                token_set_score * 0.2  # 토큰 집합 유사도
            )
            return final_score
        def find_entities(self, text, max_candidates_per_span=10):
            """Find entity matches in Korean text using fuzzy matching"""
            # Extract spans that might contain entities
            potential_spans = self._extract_korean_spans(text)
            matches = []
            for span_text, start, end in potential_spans:
                if len(span_text.strip()) < self.min_entity_length:  # Skip spans shorter than min_entity_length
                    continue
                # Get candidate entities based on n-gram overlap
                candidates = self._get_candidates(span_text)
                # If no candidates found through n-gram filtering, skip
                if not candidates:
                    continue
                # Limit the number of candidates to check
                top_candidates = [c[0] for c in candidates[:max_candidates_per_span]]
                # 각 후보 엔티티에 대해 개선된 유사도 계산
                scored_matches = []
                for entity in top_candidates:
                    score = self._calculate_similarity(span_text, entity)
                    if score >= self.min_similarity:
                        scored_matches.append((entity, score, 0))  # 호환성을 위해 3번째 매개변수 추가
                # 기존 process.extract 대신 개선된 유사도 계산 사용
                best_matches = scored_matches
                for entity, score, _ in best_matches:
                    matches.append({
                        'text': span_text,
                        'matched_entity': entity,
                        'score': score,
                        'start': start,
                        'end': end,
                        'data': self.entity_data.get(entity, {})
                    })
            # Sort by position in text
            matches.sort(key=lambda x: (x['start'], -x['score']))
            # Handle overlapping matches by keeping the best match
            final_matches = self._resolve_overlapping_matches(matches)
            return final_matches
        def _extract_korean_spans(self, text):
            """한국어와 영어가 혼합된 텍스트에서 엔티티일 수 있는 잠재적 텍스트 범위 추출"""
            spans = []
            min_len = self.min_entity_length
            # 1. 영문+한글 혼합 패턴 (붙여쓰기) 예: "T우주"
            for match in re.finditer(r'[a-zA-Z]+[가-힣]+(?:\s+[가-힣가-힣a-zA-Z0-9]+)*', text):
                if len(match.group(0)) >= min_len:
                    spans.append((match.group(0), match.start(), match.end()))
            # 2. 영문+한글 혼합 패턴 (띄어쓰기) 예: "T 우주"
            for match in re.finditer(r'[a-zA-Z]+\s+[가-힣]+(?:\s+[가-힣가-힣a-zA-Z0-9]+)*', text):
                if len(match.group(0)) >= min_len:
                    spans.append((match.group(0), match.start(), match.end()))
            # 3. 특수한 혼합 패턴 - 영문+한글+숫자 (예: "T우주365", "SK텔레콤")
            for match in re.finditer(r'[a-zA-Z]+[가-힣]+(?:[0-9]+)?', text):
                if len(match.group(0)) >= min_len:
                    spans.append((match.group(0), match.start(), match.end()))
            # 4. 연속된 두 단어까지 확장 (예: "T우주 패스")
            # 영문+한글 후 공백 하나를 두고 다른 한글 단어가 나오는 패턴
            for match in re.finditer(r'[a-zA-Z]+[가-힣]+\s+[가-힣]+', text):
                if len(match.group(0)) >= min_len:
                    spans.append((match.group(0), match.start(), match.end()))
            # 5. 연속된 세 단어까지 확장 (예: "T우주 멤버십 패스")
            for match in re.finditer(r'[a-zA-Z]+[가-힣]+\s+[가-힣]+\s+[가-힣]+', text):
                if len(match.group(0)) >= min_len:
                    spans.append((match.group(0), match.start(), match.end()))
            # 6. 브랜드명 + 제품명 패턴 (예: "SK텔레콤 T우주")
            for match in re.finditer(r'[a-zA-Z가-힣]+(?:\s+[a-zA-Z가-힣]+){1,3}', text):
                if len(match.group(0)) >= min_len:
                    spans.append((match.group(0), match.start(), match.end()))
            # 7. 숫자와 영문 결합 패턴 (숫자 space 영문 패턴, e.g. "0 day")
            for match in re.finditer(r'\d+\s+[a-zA-Z]+', text):
                if len(match.group(0)) >= min_len:
                    spans.append((match.group(0), match.start(), match.end()))
            # 8. 더 일반적인 영한 혼합 패턴
            for match in re.finditer(r'[a-zA-Z가-힣0-9]+(?:\s+[a-zA-Z가-힣0-9]+)*', text):
                if len(match.group(0)) >= min_len:
                    spans.append((match.group(0), match.start(), match.end()))
            # 9. 일반적인 구분자로 분리된 텍스트 조각도 추출
            for span in re.split(r'[,\.!?;:"\'…\(\)\[\]\{\}\s_/]+', text):
                if span and len(span) >= min_len:
                    span_pos = text.find(span)
                    if span_pos != -1:
                        spans.append((span, span_pos, span_pos + len(span)))
            return spans
        def _remove_duplicate_entities(self, matches):
            """Keep only one instance of each unique entity"""
            if not matches:
                return []
            # Dictionary to track highest-scoring match for each entity
            best_matches = {}
            for match in matches:
                entity_key = match['matched_entity']
                # If we haven't seen this entity before, or if this match has a higher score
                # than the previously saved match for this entity, save this one
                if (entity_key not in best_matches or
                    match['score'] > best_matches[entity_key]['score']):
                    best_matches[entity_key] = match
            # Return the best matches sorted by start position
            return sorted(best_matches.values(), key=lambda x: x['start'])
        def _resolve_overlapping_matches(self, matches, high_score_threshold=50, overlap_tolerance=0.5):
            if not matches:
                return []
            # 점수 내림차순, 길이 오름차순으로 정렬 (높은 점수, 짧은 매치 우선)
            sorted_matches = sorted(matches, key=lambda x: (-x['score'], x['end'] - x['start']))
            final_matches = []
            for current_match in sorted_matches:
                current_score = current_match['score']
                current_start, current_end = current_match['start'], current_match['end']
                current_range = set(range(current_start, current_end))
                current_len = len(current_range)
                # 수정: overlap_ratio를 0으로 초기화
                current_match['overlap_ratio'] = 0.0
                # 높은 점수의 매치는 항상 포함
                if current_score >= high_score_threshold:
                    # 기존 매치들과 비교하여 너무 많은 중복이 있는지 확인
                    is_too_similar = False
                    for existing_match in final_matches:
                        if existing_match['score'] < high_score_threshold:
                            continue  # 낮은 점수의 기존 매치와는 비교하지 않음
                        existing_start, existing_end = existing_match['start'], existing_match['end']
                        existing_range = set(range(existing_start, existing_end))
                        # 교집합 비율 계산
                        intersection = current_range.intersection(existing_range)
                        # 현재 매치에 대한 중복 비율
                        current_overlap_ratio = len(intersection) / current_len if current_len > 0 else 0
                        # 수정: 중복 비율 저장 - 가장 높은 중복 비율 저장
                        current_match['overlap_ratio'] = max(current_match['overlap_ratio'], current_overlap_ratio)
                        # 중복 비율이 허용 범위를 초과하고, 동일한 엔티티면 추가하지 않음
                        if (current_overlap_ratio > overlap_tolerance
                            and current_match['matched_entity'] == existing_match['matched_entity']
                            ):
                            is_too_similar = True
                            break
                    if not is_too_similar:
                        final_matches.append(current_match)
                else:
                    # 낮은 점수의 매치는 기존 로직 적용 (중복 확인)
                    should_add = True
                    for existing_match in final_matches:
                        existing_start, existing_end = existing_match['start'], existing_match['end']
                        existing_range = set(range(existing_start, existing_end))
                        # 교집합 비율 계산
                        intersection = current_range.intersection(existing_range)
                        current_overlap_ratio = len(intersection) / current_len if current_len > 0 else 0
                        # 수정: 중복 비율 저장 - 가장 높은 중복 비율 저장
                        current_match['overlap_ratio'] = max(current_match['overlap_ratio'], current_overlap_ratio)
                        # 중복 비율이 허용 범위를 초과하면 추가하지 않음
                        if current_overlap_ratio > (1 - overlap_tolerance):
                            should_add = False
                            break
                    if should_add:
                        final_matches.append(current_match)
            # 시작 위치별로 정렬
            final_matches.sort(key=lambda x: x['start'])
            return final_matches
    def find_entities_in_text(text, entity_list, min_similarity=70, ngram_size=3, min_entity_length=2,
                            token_similarity=True, high_score_threshold=50, overlap_tolerance=0.5):
        """
        Find entity matches in text using fuzzy matching.
        Parameters:
        -----------
        text : str
            The text to search for entities
        entity_list : list
            List of entities to match against
        min_similarity : int, default=70
            Minimum similarity score (0-100) for fuzzy matching
        ngram_size : int, default=2
            Size of character n-grams to use for indexing (2 or 3 recommended for Korean)
        min_entity_length : int, default=2
            Minimum length of entities to consider (characters)
        token_similarity : bool, default=True
            Whether to use token-based similarity measures
        high_score_threshold : int, default=50
            Score threshold above which matches are always kept regardless of overlap
        overlap_tolerance : float, default=0.5
            Overlap tolerance ratio (0-1), higher values allow more overlapping matches
        Returns:
        --------
        list
            List of matched entities with position and metadata
        """
        matcher = KoreanEntityMatcher(
            min_similarity=min_similarity,
            ngram_size=ngram_size,
            min_entity_length=min_entity_length,
            token_similarity=token_similarity
        )
        matcher.build_from_list(entity_list)
        matches = matcher.find_entities(text)
        # 기존 _resolve_overlapping_matches 메서드 대신 직접 호출
        final_matches = matcher._resolve_overlapping_matches(
            matches,
            high_score_threshold=high_score_threshold,
            overlap_tolerance=overlap_tolerance
        )
        return final_matches
    # Function to highlight entities in text
    def highlight_entities(text, matches):
        marked_text = text
        offset = 0
        for match in sorted(matches, key=lambda x: x['start'], reverse=True):
            start = match['start'] + offset
            end = match['end'] + offset
            entity = match['matched_entity']
            score = match['score']
            marked_text = marked_text[:start] + f"[{marked_text[start:end]}→{entity} ({score:.1f}%)]" + marked_text[end:]
            offset += len(f"[→{entity} ({score:.1f}%)]") + 2
        return marked_text
    import kiwipiepy
    def reconstruct_text_from_tokens(tokens):
        """
        tokens: Kiwi.analyze()의 결과로 반환된 Token 객체 리스트
        filter_rules: 필터링 규칙 (예: 특정 품사만 선택, 특정 단어 제외 등)
        """
        if tokens:
            max_pos = max(token.start + token.len for token in tokens)
            text_array = [' '] * max_pos  # 공백으로 초기화
            for token in tokens:
                for i in range(token.start, token.start + token.len):
                    if i < max_pos:
                        text_array[i] = token.form[i - token.start]
            text = ''.join(text_array)
        else:
            text = ''
        return text
    def extract_by_tag_pattern(result, tag_patterns, original_text=None):
        """
        kiwi.analyze() 결과에서 특정 tag 패턴에 일치하는 form들을 추출합니다.
        Parameters:
        - result: kiwi.analyze()의 결과
        - tag_patterns: 찾고자 하는 tag 패턴 리스트
        - original_text: 원본 텍스트 (위치 기반 추출 시 필요)
        """
        # kiwi.analyze()의 결과에서 토큰 리스트 추출
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], tuple):
            tokens = result[0][0]  # 첫 번째 분석 결과에서 토큰 리스트 추출
        else:
            tokens = result  # 이미 토큰 리스트인 경우
        matches = {i: [] for i in range(len(tag_patterns))}  # 패턴별 매치 결과
        # 각 패턴에 대해 검색
        for pattern_idx, pattern in enumerate(tag_patterns):
            pattern_len = len(pattern)
            # 슬라이딩 윈도우 방식으로 패턴 검색
            for i in range(len(tokens) - pattern_len + 1):
                window = tokens[i:i+pattern_len]
                # 현재 윈도우가 패턴과 일치하는지 확인
                if all(token.tag == pattern[j] for j, token in enumerate(window)):
                    # 패턴과 일치하면 form들을 추출하여 저장
                    if original_text and window:
                        start_pos = window[0].start
                        end_pos = window[-1].start + window[-1].len
                        matched_forms = original_text[start_pos:end_pos]
                    else:
                        # 원본 텍스트가 없으면 토큰 form을 직접 합침 (공백 없이)
                        matched_forms = ''.join(token.form for token in window)
                    matches[pattern_idx].append(matched_forms)
        return matches
    def extract_by_flexible_tag_pattern(result, tag_patterns, max_gap=0, original_text=None):
        """
        kiwi.analyze() 결과에서 특정 tag 패턴에 일치하는 form들을 유연하게 추출합니다.
        중간에 다른 태그가 있어도 일정 개수 이내라면 매칭합니다.
        Parameters:
        - result: kiwi.analyze()의 결과
        - tag_patterns: 찾고자 하는 tag 패턴 리스트. 예: [['NNG', 'JKS', 'VV'], ['VA', 'NNG']]
        - max_gap: 패턴 사이에 허용되는 최대 간격(다른 태그의 개수)
        - original_text: 원본 텍스트 (위치 기반 추출 시 필요)
        Returns:
        - 패턴별 일치하는 form의 리스트 딕셔너리와 매칭된 토큰 인덱스
        """
        # kiwi.analyze()의 결과에서 토큰 리스트 추출
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], tuple):
            tokens = result[0][0]  # 첫 번째 분석 결과에서 토큰 리스트 추출
        else:
            tokens = result  # 이미 토큰 리스트인 경우
        matches = {i: [] for i in range(len(tag_patterns))}  # 패턴별 매치 결과
        match_indices = {i: [] for i in range(len(tag_patterns))}  # 매치된 토큰 인덱스
        # 각 패턴에 대해 검색
        for pattern_idx, pattern in enumerate(tag_patterns):
            i = 0
            while i < len(tokens):
                matched_indices = []
                pattern_pos = 0
                gaps = 0
                j = i
                # 패턴 매칭 시도
                match_found = False
                while j < len(tokens) and pattern_pos < len(pattern):
                    if tokens[j].tag == pattern[pattern_pos]:
                        matched_indices.append(j)
                        pattern_pos += 1
                        gaps = 0
                    else:
                        gaps += 1
                        if gaps > max_gap:
                            # 허용된 간격을 초과하면 매칭 실패
                            break
                    j += 1
                    # 패턴을 모두 매칭했는지 확인
                    if pattern_pos == len(pattern):
                        match_found = True
                        break
                # 매칭에 성공한 경우 결과 추가
                if match_found:
                    matched_tokens = [tokens[idx] for idx in matched_indices]
                    # 원본 텍스트에서 추출하거나 토큰 form 직접 연결
                    if original_text and matched_tokens:
                        start_pos = matched_tokens[0].start
                        end_pos = matched_tokens[-1].start + matched_tokens[-1].len
                        matched_forms = original_text[start_pos:end_pos]
                    else:
                        # 원본 텍스트가 없으면 토큰 form을 직접 합침 (공백 없이)
                        matched_forms = ''.join(token.form for token in matched_tokens)
                    matches[pattern_idx].append(matched_forms)
                    match_indices[pattern_idx].append(matched_indices)
                    # 매칭 후 겹치지 않도록 다음 위치로 이동
                    i = matched_indices[-1] + 1
                else:
                    i += 1  # 매칭 실패한 경우 다음 위치로 이동
        return matches, match_indices
    def reconstruct_with_tag_patterns(result, tag_patterns, include_unmatched=False, original_text=None):
        """
        kiwi.analyze() 결과에서 특정 tag 패턴에 일치하는 부분을 강조하여 텍스트 재구성
        Parameters:
        - result: kiwi.analyze()의 결과
        - tag_patterns: 찾고자 하는 tag 패턴 리스트
        - include_unmatched: 패턴에 일치하지 않는 토큰도 포함할지 여부
        - original_text: 원본 텍스트 (위치 기반 추출 시 필요)
        Returns:
        - 재구성된 텍스트 (패턴 일치 부분은 <match>로 강조)
        """
        # kiwi.analyze()의 결과에서 토큰 리스트 추출
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], tuple):
            tokens = result[0][0]  # 첫 번째 분석 결과에서 토큰 리스트 추출
        else:
            tokens = result  # 이미 토큰 리스트인 경우
        _, match_indices = extract_by_flexible_tag_pattern(result, tag_patterns, original_text=original_text)
        # 모든 매치 인덱스를 하나의 집합으로 합침
        all_matched_indices = set()
        for indices_list in match_indices.values():
            for indices in indices_list:
                all_matched_indices.update(indices)
        # 텍스트 재구성
        reconstructed_parts = []
        i = 0
        while i < len(tokens):
            if i in all_matched_indices:
                # 매치된 패턴의 시작 찾기
                for pattern_idx, indices_list in match_indices.items():
                    for indices in indices_list:
                        if i == indices[0]:  # 패턴의 시작
                            pattern_tokens = [tokens[idx] for idx in indices]
                            # 원본 텍스트에서 추출하거나 토큰 form 직접 연결
                            if original_text and pattern_tokens:
                                start_pos = pattern_tokens[0].start
                                end_pos = pattern_tokens[-1].start + pattern_tokens[-1].len
                                pattern_text = original_text[start_pos:end_pos]
                            else:
                                pattern_text = ''.join(token.form for token in pattern_tokens)
                            reconstructed_parts.append(f"<match>{pattern_text}</match>")
                            i = indices[-1] + 1  # 패턴 이후로 인덱스 이동
                            break
                    else:
                        continue
                    break
            else:
                # 매치되지 않은 토큰
                if include_unmatched:
                    reconstructed_parts.append(tokens[i].form)
                i += 1
        return ''.join(reconstructed_parts)
    import numpy as np
    import matplotlib.pyplot as plt
    from difflib import SequenceMatcher
    def advanced_sequential_similarity(str1, str2, metrics=None, visualize=False):
        """
        Calculate multiple character-level similarity metrics between two strings.
        Parameters:
        -----------
        str1 : str
            First string
        str2 : str
            Second string
        metrics : list
            List of metrics to compute. Options:
            ['ngram', 'lcs', 'subsequence', 'difflib']
            If None, all metrics will be computed
        visualize : bool
            If True, visualize the differences between strings
        Returns:
        --------
        dict
            Dictionary containing similarity scores for each metric
        """
        if metrics is None:
            metrics = ['ngram', 'lcs', 'subsequence', 'difflib']
        results = {}
        # Handle empty strings
        if not str1 or not str2:
            return {metric: 0.0 for metric in metrics}
        # Prepare strings
        s1, s2 = str1.lower(), str2.lower()
        # 1. N-gram similarity (with multiple window sizes)
        if 'ngram' in metrics:
            ngram_scores = {}
            for window in range(min([len(s1),len(s2),2]), min([5,max([len(s1),len(s2)])+1])):
                # Skip if strings are shorter than window
                if len(s1) < window or len(s2) < window:
                    ngram_scores[f'window_{window}'] = 0.0
                    continue
                # Generate character n-grams
                ngrams1 = [s1[i:i+window] for i in range(len(s1) - window + 1)]
                ngrams2 = [s2[i:i+window] for i in range(len(s2) - window + 1)]
                # Count matches
                matches = sum(1 for ng in ngrams1 if ng in ngrams2)
                max_possible = max(len(ngrams1), len(ngrams2))
                # Normalize
                score = matches / max_possible if max_possible > 0 else 0.0
                ngram_scores[f'window_{window}'] = score
            # Average of all n-gram scores
            results['ngram'] = max(ngram_scores.values())#sum(ngram_scores.values()) / len(ngram_scores)
            results['ngram_details'] = ngram_scores
        # 2. Longest Common Substring (LCS)
        if 'lcs' in metrics:
            def longest_common_substring(s1, s2):
                # Dynamic programming approach
                m, n = len(s1), len(s2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                max_length = 0
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if s1[i-1] == s2[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                            max_length = max(max_length, dp[i][j])
                return max_length
            lcs_length = longest_common_substring(s1, s2)
            max_length = max(len(s1), len(s2))
            results['lcs'] = lcs_length / max_length if max_length > 0 else 0.0
        # 3. Longest Common Subsequence
        if 'subsequence' in metrics:
            def longest_common_subsequence(s1, s2):
                # Dynamic programming approach for subsequence
                m, n = len(s1), len(s2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if s1[i-1] == s2[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                return dp[m][n]
            subseq_length = longest_common_subsequence(s1, s2)
            max_length = max(len(s1), len(s2))
            results['subsequence'] = subseq_length / max_length if max_length > 0 else 0.0
        # 4. SequenceMatcher from difflib
        if 'difflib' in metrics:
            sm = SequenceMatcher(None, s1, s2)
            results['difflib'] = sm.ratio()
        # Visualization of differences
        if visualize:
            try:
                # Only works in notebooks or environments that support plotting
                sm = SequenceMatcher(None, s1, s2)
                matches = sm.get_matching_blocks()
                # Prepare for visualization
                fig, ax = plt.subplots(figsize=(10, 3))
                # Draw strings as horizontal bars
                ax.barh(0, len(s1), height=0.4, left=0, color='lightgray', alpha=0.3)
                ax.barh(1, len(s2), height=0.4, left=0, color='lightgray', alpha=0.3)
                # Draw matching parts
                for match in matches:
                    i, j, size = match
                    if size > 0:  # Ignore zero-length matches
                        ax.barh(0, size, height=0.4, left=i, color='green', alpha=0.5)
                        ax.barh(1, size, height=0.4, left=j, color='green', alpha=0.5)
                        # Draw connection lines between matches
                        ax.plot([i + size/2, j + size/2], [0.2, 0.8], 'k-', alpha=0.3)
                # Add string texts
                for i, c in enumerate(s1):
                    ax.text(i + 0.5, 0, c, ha='center', va='center')
                for i, c in enumerate(s2):
                    ax.text(i + 0.5, 1, c, ha='center', va='center')
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['String 1', 'String 2'])
                ax.set_xlabel('Character Position')
                ax.set_title('Character-Level String Comparison')
                ax.grid(False)
                plt.tight_layout()
                # plt.show()  # Uncomment to display
            except Exception as e:
                print(f"Visualization error: {e}")
        # Calculate overall similarity score (average of all metrics)
        metrics_to_average = [m for m in results.keys() if not m.endswith('_details')]
        results['overall'] = sum(results[m] for m in metrics_to_average) / len(metrics_to_average)
        return results
    # advanced_sequential_similarity('시크릿', '시크릿', metrics='ngram')
    # advanced_sequential_similarity('에이닷_자사', '에이닷')
    # mms_pdf = pd.read_excel("./data/mms_data_250408.xlsx", engine="openpyxl")
    mms_pdf = pd.read_csv("./data/mms_data_250408.csv")
    mms_pdf['msg'] = mms_pdf['msg_nm']+"\n"+mms_pdf['mms_phrs']
    mms_pdf = mms_pdf.groupby(["msg_nm","mms_phrs","msg"])['offer_dt'].min().reset_index(name="offer_dt")
    mms_pdf = mms_pdf.reset_index()
    mms_pdf = mms_pdf.astype('str')
    # mms_pdf.sample(100)[['msg']].to_csv("./data/mms_sample.csv", index=False)
    schema_ext = {
        "title": {
            "type": "string",
            'description': '광고 제목. 광고의 핵심 주제와 가치 제안을 명확하게 설명할 수 있도록 생성'
        },
        'purpose': {
            'type': 'array',
            'description': '광고의 주요 목적을 다음 중에서 선택(복수 가능): [상품 가입 유도, 대리점 방문 유도, 웹/앱 접속 유도, 이벤트 응모 유도, 혜택 안내, 쿠폰 제공 안내, 경품 제공 안내, 기타 정보 제공]'
        },
        'product': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                'name': {'type': 'string', 'description': '광고하는 제품이나 서비스 이름'},
                'action': {'type': 'string', 'description': '고객에게 기대하는 행동: [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타] 중에서 선택'}
                }
            }
        },
        'channel': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'type': {'type': 'string', 'description': '채널 종류: [URL, 전화번호, 앱, 대리점] 중에서 선택'},
                    'value': {'type': 'string', 'description': '실제 URL, 전화번호, 앱 이름, 대리점 이름 등 구체적 정보'},
                    'action': {'type': 'string', 'description': '채널 목적: [가입, 추가 정보, 문의, 수신, 수신 거부] 중에서 선택'},
                    'benefit': {'type': 'string', 'description': '해당 채널 이용 시 특별 혜택'},
                    'store_code': {'type': 'string', 'description': "매장 코드 - tworldfriends.co.kr URL에서 D+숫자 9자리(D[0-9]{9}) 패턴의 코드 추출하여 대리점 채널에 설정"}
                }
            }
        },
        'pgm':{
            'type': 'array',
            'description': '아래 광고 분류 기준 정보에서 선택. 메세지 내용과 광고 분류 기준을 참고하여, 광고 메세지에 가장 부합하는 2개의 pgm_nm을 적합도 순서대로 제공'
        },
    'required': ['purpose', 'product', 'channel', 'pgm'],
    'objectType': 'object'}
    item_pdf_raw = pd.read_csv("./data/item_info_all_250527.csv")
    item_pdf_all = item_pdf_raw.drop_duplicates(['item_nm','item_id'])[['item_nm','item_id','item_desc','domain','start_dt','end_dt','rank']].copy()
    item_pdf_all.query("rank<1000 and item_nm.str.contains('넷플릭스', case=False)").head()
    # item_pdf_all.query("rank<1000")[['item_nm']].drop_duplicates().to_csv("./data/item_nm_1000.csv", index=False)
    alia_rule_set = list(zip(pd.read_csv("./data/alias_rules_ke.csv")['korean'], pd.read_csv("./data/alias_rules_ke.csv")['english']))
    def apply_alias_rule(item_nm):
        item_nm_list = [item_nm]
        for r in alia_rule_set:
            if r[0] in item_nm:
                item_nm_list.append(item_nm.replace(r[0], r[1]))
            if r[1] in item_nm:
                item_nm_list.append(item_nm.replace(r[1], r[0]))
        return item_nm_list
    item_pdf_all['item_nm_alias'] = item_pdf_all['item_nm'].apply(apply_alias_rule)
    item_pdf_all = item_pdf_all.explode('item_nm_alias')
    item_pdf_all.query("rank<1000 and item_nm.str.contains('넷플릭스', case=False) or item_nm.str.contains('웨이브', case=False)")[['item_nm','item_nm_alias','item_id']]
    user_defined_entity = ['AIA Vitality' , '부스트 파크 건대입구' , 'Boost Park 건대입구']
    item_pdf_ext = pd.DataFrame([{'item_nm':e,'item_id':e,'item_desc':e, 'domain':'user_defined', 'start_dt':20250101, 'end_dt':99991231, 'rank':1, 'item_nm_alias':e} for e in user_defined_entity])
    item_pdf_all = pd.concat([item_pdf_all,item_pdf_ext])
    entity_list_for_fuzzy = []
    for row in item_pdf_all.to_dict('records'):
        entity_list_for_fuzzy.append((row['item_nm'], {'item_id':row['item_id'], 'description':row['item_desc'], 'domain':row['domain'], 'start_dt':row['start_dt'], 'end_dt':row['end_dt'], 'rank':1, 'item_nm_alias':row['item_nm_alias']}))
    stop_item_names = pd.read_csv("./data/stop_words.csv")['stop_words'].to_list()
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer('jhgan/ko-sbert-nli')
    import re
    num_cand_pgms = 5
    pgm_pdf = pd.read_csv("./data/pgm_tag_ext_250516.csv")
    def preprocess_text(text):
        # 특수문자를 공백으로 변환
        text = re.sub(r'[^\w\s]', ' ', text)
        # 여러 공백을 하나로 통일
        text = re.sub(r'\s+', ' ', text)
        # 앞뒤 공백 제거
        return text.strip()
    clue_embeddings = model.encode(pgm_pdf[["pgm_nm","clue_tag"]].apply(lambda x: preprocess_text(x['pgm_nm'].lower())+" "+x['clue_tag'].lower(), axis=1).tolist(), convert_to_tensor=True)
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    stop_item_names = list(set(stop_item_names + [x.lower() for x in stop_item_names]))
    entity_list_for_kiwi = list(item_pdf_all['item_nm_alias'].unique())
    for w in entity_list_for_kiwi:
        kiwi.add_user_word(w, "NNP")
    for w in stop_item_names:
        kiwi.add_user_word(w, "NNG")
    kiwi_raw = Kiwi()
    kiwi_raw.space_tolerance = 2
    tags_to_exclude = ['W_SERIAL','W_URL','JKO','SSO','SSC','SW','SF','SP','SS','SE','SO','SB','SH']
    edf = item_pdf_all.copy()
    edf['token_entity'] = edf.apply(lambda x: kiwi_raw.tokenize(x['item_nm_alias'], normalize_coda=True, z_coda=False, split_complex=False), axis=1)
    edf['token_entity'] = edf.apply(lambda x: [d[0] for d in x['token_entity'] if d[1] not in tags_to_exclude], axis=1)
    edf['char_entity'] = edf.apply(lambda x: list(x['item_nm_alias'].lower().replace(' ', '')), axis=1)
    exc_tag_patterns = [
        ['SN', 'NNB'], ['W_SERIAL'], ['JKO'], ['W_URL'], ['W_EMAIL'],
        ['XSV', 'EC'], ['VV', 'EC'], ['VCP', 'ETM'], ['XSA', 'ETM'], ['VV', 'ETN']
    ]+[[t] for t in tags_to_exclude]
    class Token:
        def __init__(self, form, tag, start, len):
            self.form = form
            self.tag = tag
            self.start = start
            self.len = len
        def __repr__(self):
            return f"Token(form='{self.form}', tag='{self.tag}', start={self.start}, len={self.len})"
    # 제외할 품사 패턴
    exc_tag_patterns = [
        ['SN', 'NNB'], ['W_SERIAL'], ['JKO'], ['W_URL'], ['W_EMAIL'],
        ['XSV', 'EC'], ['VV', 'EC'], ['VCP', 'ETM'], ['XSA', 'ETM'], ['VV', 'ETN']
    ]+[[t] for t in tags_to_exclude]
    msg_text_list = ["""
    광고 제목:[SK텔레콤] 2월 0 day 혜택 안내
    광고 내용:(광고)[SKT] 2월 0 day 혜택 안내__[2월 10일(토) 혜택]_만 13~34세 고객이라면_베어유 모든 강의 14일 무료 수강 쿠폰 드립니다!_(선착순 3만 명 증정)_▶ 자세히 보기: http://t-mms.kr/t.do?m=#61&s=24589&a=&u=https://bit.ly/3SfBjjc__■ 에이닷 X T 멤버십 시크릿코드 이벤트_에이닷 T 멤버십 쿠폰함에 ‘에이닷이빵쏜닷’을 입력해보세요!_뚜레쥬르 데일리우유식빵 무료 쿠폰을 드립니다._▶ 시크릿코드 입력하러 가기: https://bit.ly/3HCUhLM__■ 문의: SKT 고객센터(1558, 무료)_무료 수신거부 1504
    """,
    """
    광고 제목:통화 부가서비스를 패키지로 저렴하게!
    광고 내용:(광고)[SKT] 콜링플러스 이용 안내  #04 고객님, 안녕하세요. <콜링플러스>에 가입하고 콜키퍼, 컬러링, 통화가능통보플러스까지 총 3가지의 부가서비스를 패키지로 저렴하게 이용해보세요.  ■ 콜링플러스 - 이용요금: 월 1,650원, 부가세 포함 - 콜키퍼(550원), 컬러링(990원), 통화가능통보플러스(770원)를 저렴하게 이용할 수 있는 상품  ■ 콜링플러스 가입 방법 - T월드 앱: 오른쪽 위에 있는 돋보기를 눌러 콜링플러스 검색 > 가입  ▶ 콜링플러스 가입하기: http://t-mms.kr/t.do?m=#61&u=https://skt.sh/17tNH  ■ 유의 사항 - 콜링플러스에 가입하면 기존에 이용 중인 콜키퍼, 컬러링, 통화가능통보플러스 서비스는 자동으로 해지됩니다. - 기존에 구매한 컬러링 음원은 콜링플러스 가입 후에도 계속 이용할 수 있습니다.(시간대, 발신자별 설정 정보는 다시 설정해야 합니다.)  * 최근 다운로드한 음원은 보관함에서 무료로 재설정 가능(다운로드한 날로부터 1년 이내)   ■ 문의: SKT 고객센터(114)  SKT와 함께해주셔서 감사합니다.  무료 수신거부 1504\n    ',
    """,
    """
    (광고)[SKT] 1월 0 day 혜택 안내_ _[1월 20일(토) 혜택]_만 13~34세 고객이라면 _CU에서 핫바 1,000원에 구매 하세요!_(선착순 1만 명 증정)_▶ 자세히 보기 : http://t-mms.kr/t.do?m=#61&s=24264&a=&u=https://bit.ly/3H2OHSs__■ 에이닷 X T 멤버십 구독캘린더 이벤트_0 day 일정을 에이닷 캘린더에 등록하고 혜택 날짜에 알림을 받아보세요! _알림 설정하면 추첨을 통해 [스타벅스 카페 라떼tall 모바일쿠폰]을 드립니다. _▶ 이벤트 참여하기 : https://bit.ly/3RVSojv_ _■ 문의: SKT 고객센터(1558, 무료)_무료 수신거부 1504
    """,
    """
    '[T 우주] 넷플릭스와 웨이브를 월 9,900원에! \n(광고)[SKT] 넷플릭스+웨이브 월 9,900원, 이게 되네! __#04 고객님,_넷플릭스와 웨이브 둘 다 보고 싶었지만, 가격 때문에 망설이셨다면 지금이 바로 기회! __오직 T 우주에서만, _2개월 동안 월 9,900원에 넷플릭스와 웨이브를 모두 즐기실 수 있습니다.__8월 31일까지만 드리는 혜택이니, 지금 바로 가입해 보세요! __■ 우주패스 Netflix 런칭 프로모션 _- 기간 : 2024년 8월 31일(토)까지_- 혜택 : 우주패스 Netflix(광고형 스탠다드)를 2개월 동안 월 9,900원에 이용 가능한 쿠폰 제공_▶ 프로모션 자세히 보기: http://t-mms.kr/jAs/#74__■ 우주패스 Netflix(월 12,000원)  _- 기본 혜택 : Netflix 광고형 스탠다드 멤버십_- 추가 혜택 : Wavve 콘텐츠 팩 _* 추가 요금을 내시면 Netflix 스탠다드와 프리미엄 멤버십 상품으로 가입 가능합니다.  __■ 유의 사항_-  프로모션 쿠폰은 1인당 1회 다운로드 가능합니다. _-  쿠폰 할인 기간이 끝나면 정상 이용금액으로 자동 결제 됩니다. __■ 문의: T 우주 고객센터 (1505, 무료)__나만의 구독 유니버스, T 우주 __무료 수신거부 1504'
    """,
    """
    광고 제목:[SK텔레콤] T건강습관 X AIA Vitality, 우리 가족의 든든한 보험!
    광고 내용:(광고)[SKT] 가족의 든든한 보험 (무배당)AIA Vitality 베스트핏 보장보험 안내  고객님, 안녕하세요. 4인 가족 표준생계비, 준비하고 계시나요? (무배당)AIA Vitality 베스트핏 보장보험(디지털 전용)으로 최대 20% 보험료 할인과 가족의 든든한 보험 보장까지 누려 보세요.   ▶ 자세히 보기: http://t-mms.kr/t.do?m=#61&u=https://bit.ly/36oWjgX  ■ AIA Vitality  혜택 - 매달 리워드 최대 12,000원 - 등급 업그레이드 시 특별 리워드 - T건강습관 제휴 할인 최대 40% ※ 제휴사별 할인 조건과 주간 미션 달성 혜택 등 자세한 내용은 AIA Vitality 사이트에서 확인하세요. ※ 이 광고는 AIA생명의 광고이며 SK텔레콤은 모집 행위를 하지 않습니다.  - 보험료 납입 기간 중 피보험자가 장해분류표 중 동일한 재해 또는 재해 이외의 동일한 원인으로 여러 신체 부위의 장해지급률을 더하여 50% 이상인 장해 상태가 된 경우 차회 이후의 보험료 납입 면제 - 사망보험금은 계약일(부활일/효력회복일)로부터 2년 안에 자살한 경우 보장하지 않음 - 일부 특약 갱신 시 보험료 인상 가능 - 기존 계약 해지 후 신계약 체결 시 보험인수 거절, 보험료 인상, 보장 내용 변경 가능 - 해약 환급금(또는 만기 시 보험금이나 사고보험금)에 기타 지급금을 합해 5천만 원까지(본 보험 회사 모든 상품 합산) 예금자 보호 - 계약 체결 전 상품 설명서 및 약관 참조 - 월 보험료 5,500원(부가세 포함)  * 생명보험협회 심의필 제2020-03026호(2020-09-22) COM-2020-09-32426  ■문의: 청약 관련(1600-0880)  무료 수신거부 1504
    """
    ]
    message_idx = 0
    mms_msg = msg_text_list[message_idx]
    for mms_msg in mms_pdf.sample(30)['mms_msg'].tolist():
        mms_embedding = model.encode([mms_msg.lower()], convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(
            mms_embedding,
            clue_embeddings,
            dim=1
        ).cpu().numpy()
        pgm_pdf_tmp = pgm_pdf.copy()
        pgm_pdf_tmp['sim'] = similarities
        pgm_pdf_tmp = pgm_pdf_tmp.sort_values('sim', ascending=False)
        def filter_specific_terms(strings: List[str]) -> List[str]:
            unique_strings = list(set(strings))  # 중복 제거
            unique_strings.sort(key=len, reverse=True)  # 길이 기준 내림차순 정렬
            filtered = []
            for s in unique_strings:
                if not any(s in other for other in filtered):
                    filtered.append(s)
            return filtered
        def sliding_window_with_step(data, window_size, step=1):
            """Sliding window with configurable step size."""
            return [data[i:i + window_size] for i in range(0, len(data) - window_size + 1, step)]
        # tdf = pd.DataFrame([{'form_text':d[0],'tag_text':d[1],'start_text':d[2],'end_text':d[2]+d[3]} for d in kiwi_raw.analyze(mms_msg)[0][0]])
        result_msg_raw = kiwi_raw.tokenize(mms_msg, normalize_coda=True, z_coda=False, split_complex=False)
        token_list_msg = [d for d in result_msg_raw
                        if d[1] not in tags_to_exclude
                        ]
        result_msg = kiwi.tokenize(mms_msg, normalize_coda=True, z_coda=False, split_complex=False)
        entities_from_kiwi = []
        for token in result_msg:  # 첫 번째 분석 결과의 토큰 리스트
            if token.tag == 'NNP' and token.form not in stop_item_names+['-'] and len(token.form)>=2 and not token.form.lower() in stop_item_names:  # 고유명사인 경우
            # if token.tag == 'NNG' and token.form in stop_item_names_ext:  # 고유명사인 경우
                entities_from_kiwi.append(token.form)
        from typing import List
        # 결과
        entities_from_kiwi = filter_specific_terms(entities_from_kiwi)
        # print("추출된 개체명:", list(set(entities_from_kiwi)))
        ngram_list_msg = []
        for w_size in range(1,5):
            windows = sliding_window_with_step(token_list_msg, w_size, step=1)
            windows_new = []
            for w in windows:
                tag_str = ','.join([t.tag for t in w])
                flag = True
                for et in exc_tag_patterns:
                    if ','.join(et) in tag_str:
                        flag = False
                        # print(w)
                        break
                if flag:
                    windows_new.append([[d.form for d in w], [d.tag for d in w]])
            ngram_list_msg.extend(windows_new)
        # 패턴에 해당하는 토큰 인덱스 찾기
        def find_pattern_indices(tokens, patterns):
            indices_to_exclude = set()
            # 단일 태그 패턴 먼저 체크
            for i in range(len(tokens)):
                for pattern in patterns:
                    if len(pattern) == 1 and tokens[i].tag == pattern[0]:
                        indices_to_exclude.add(i)
            # 연속된 패턴 검사
            i = 0
            while i < len(tokens):
                if i in indices_to_exclude:
                    i += 1
                    continue
                for pattern in patterns:
                    if len(pattern) > 1:  # 두 개 이상의 태그로 구성된 패턴
                        if i + len(pattern) <= len(tokens):  # 패턴 길이만큼 토큰이 남아있는지 확인
                            match = True
                            for j in range(len(pattern)):
                                if tokens[i+j].tag != pattern[j]:
                                    match = False
                                    break
                            if match:  # 패턴이 일치하면 해당 토큰들의 인덱스를 모두 추가
                                for j in range(len(pattern)):
                                    indices_to_exclude.add(i+j)
                i += 1
            return indices_to_exclude
        # 패턴에 해당하지 않는 토큰만 필터링
        def filter_tokens_by_patterns(tokens, patterns):
            indices_to_exclude = find_pattern_indices(tokens, patterns)
            return [tokens[i] for i in range(len(tokens)) if i not in indices_to_exclude]
        # 제외된 토큰 없이 텍스트 재구성 - 단순 연결 방식
        def reconstruct_text_without_spaces(tokens):
            # 토큰들을 원래 시작 위치에 따라 정렬
            sorted_tokens = sorted(tokens, key=lambda token: token.start)
            result = []
            for token in sorted_tokens:
                result.append(token.form)
            # 토큰들을 공백 하나로 구분하여 결합
            return ' '.join(result)
        # 더 자연스러운 텍스트 재구성 - 원본 위치 기반 보존, 제외된 토큰은 건너뜀
        def reconstruct_text_preserved_positions(original_tokens, filtered_tokens):
            # 원본 토큰의 위치와 형태를 기록할 사전 생성
            token_map = {}
            for i, token in enumerate(original_tokens):
                token_map[(token.start, token.len)] = (i, token.form)
            # 필터링된 토큰의 인덱스 찾기
            filtered_indices = set()
            for token in filtered_tokens:
                key = (token.start, token.len)
                if key in token_map:
                    filtered_indices.add(token_map[key][0])
            # 원본 순서대로 필터링된 토큰만 선택
            result = []
            for i, token in enumerate(original_tokens):
                if i in filtered_indices:
                    result.append(token.form)
            return ' '.join(result)
        # 결과 출력
        filtered_tokens = filter_tokens_by_patterns(result_msg_raw, exc_tag_patterns)
        msg_text_filtered = reconstruct_text_preserved_positions(result_msg_raw, filtered_tokens)
        msg_text_filtered
        ngram_list_msg_filtered = []
        for w_size in range(2,4):
            windows = sliding_window_with_step(list(msg_text_filtered.lower().replace(' ', '')), w_size, step=1)
            ngram_list_msg_filtered.extend(windows)
        col_for_form_tmp_ent = 'char_entity'
        col_for_form_tmp_msg = 'char_msg'
        edf['form_tmp'] = edf[col_for_form_tmp_ent].apply(lambda x: [' '.join(s) for s in sliding_window_with_step(x, 2, step=1)])
        tdf = pd.DataFrame(ngram_list_msg).rename(columns={0:'token_txt', 1:'token_tag'})
        tdf['token_key'] = tdf.apply(lambda x: ''.join(x['token_txt'])+''.join(x['token_tag']), axis=1)
        tdf = tdf.drop_duplicates(['token_key']).drop(['token_key'], axis=1)
        tdf['char_msg'] = tdf.apply(lambda x: list((" ".join(x['token_txt'])).lower().replace(' ', '')), axis=1)
        tdf['form_tmp'] = tdf[col_for_form_tmp_msg].apply(lambda x: [' '.join(s) for s in sliding_window_with_step(x, 2, step=1)])
        tdf['token_txt_str'] = tdf['token_txt'].str.join(',')
        tdf['token_tag_str'] = tdf['token_tag'].str.join(',')
        # tdf['txt'] = tdf.apply(lambda x: ' '.join(x['token_txt']), axis=1)
        fdf = edf.explode('form_tmp').merge(tdf.explode('form_tmp'), on='form_tmp').drop(['form_tmp'], axis=1)
        fdf = fdf.query("item_nm_alias.str.lower() not in @stop_item_names and token_txt_str.replace(',','').str.lower() not in @stop_item_names").drop_duplicates(['item_nm','item_nm_alias','item_id','token_txt_str','token_tag_str'])
        def ngram_jaccard_similarity(list1, list2, n=2):
            """Calculate similarity using Jaccard similarity of n-grams."""
            # Generate n-grams for both lists
            def get_ngrams(lst, n):
                return [tuple(lst[i:i+n]) for i in range(len(lst)-n+1)]
            # Handle edge cases
            if len(list1) < n or len(list2) < n:
                if list1 == list2:
                    return 1.0
                else:
                    return 0.0
            # Generate n-grams and calculate Jaccard similarity
            ngrams1 = set(get_ngrams(list1, n))
            ngrams2 = set(get_ngrams(list2, n))
            intersection = ngrams1.intersection(ngrams2)
            union = ngrams1.union(ngrams2)
            return len(intersection) / len(union) if union else 0
        def needleman_wunsch_similarity(list1, list2, match_score=1, mismatch_penalty=1, gap_penalty=1):
            """Global sequence alignment with Needleman-Wunsch algorithm."""
            m, n = len(list1), len(list2)
            # Initialize score matrix
            score = np.zeros((m+1, n+1))
            # Initialize first row and column with gap penalties
            for i in range(m+1):
                score[i][0] = -i * gap_penalty
            for j in range(n+1):
                score[0][j] = -j * gap_penalty
            # Fill the score matrix
            for i in range(1, m+1):
                for j in range(1, n+1):
                    match = score[i-1][j-1] + (match_score if list1[i-1] == list2[j-1] else -mismatch_penalty)
                    delete = score[i-1][j] - gap_penalty
                    insert = score[i][j-1] - gap_penalty
                    score[i][j] = max(match, delete, insert)
            # Calculate similarity score
            max_possible_score = min(m, n) * match_score
            alignment_score = score[m][n]
            # Normalize to 0-1 range
            min_possible_score = -max(m, n) * max(gap_penalty, mismatch_penalty)
            normalized_score = (alignment_score - min_possible_score) / (max_possible_score - min_possible_score)
            return normalized_score
        fdf['sim_score_token'] = fdf.apply(lambda row: needleman_wunsch_similarity(row['token_txt'], row['token_entity']), axis=1)
        fdf['sim_score_char'] = fdf.apply(lambda row: advanced_sequential_similarity((''.join(row['char_msg'])), (''.join(row['char_entity'])),metrics='difflib')['difflib'], axis=1)
        entity_list = [e.replace(' ', '').lower() for e in list(edf['item_nm_alias'].unique())]
        entities_from_kiwi_rev = [e.replace(' ', '').lower() for e in entities_from_kiwi]
        kdf = fdf.query("item_nm_alias in @entities_from_kiwi_rev or token_txt_str.str.replace(',',' ').str.lower() in @entities_from_kiwi_rev or token_txt_str.str.replace(',','').str.lower() in @entities_from_kiwi_rev").copy()
        kdf = kdf.query("(sim_score_token>=0.75 and sim_score_char>=0.75) or sim_score_char>=1").query("item_nm_alias.str.replace(',','').str.lower() in @entity_list or item_nm_alias.str.replace(' ','').str.lower() in @entity_list")
        kdf['rank'] = kdf.groupby(['token_txt_str'])['sim_score_char'].rank(ascending=False, method='dense')#.reset_index(name='rank')
        kdf = kdf.query("rank<=1")[['item_nm','item_nm_alias','item_id','token_txt_str','domain','sim_score_token','sim_score_char']].drop_duplicates()
        # kdf = kdf.query("rank<=1")
        # kdf = kdf.groupby('item_nm_alias', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 2), random_state=42), include_groups=False)
        tags_to_exclude_final = ['SN']
        filtering_condition = [
        """not token_tag_str in @tags_to_exclude_final"""
        ,"""and token_txt_str.str.len()>=2"""
        ,"""and not token_txt_str in @stop_item_names"""
        ,"""and not token_txt_str.str.replace(',','').str.lower() in @stop_item_names"""
        ,"""and not item_nm_alias in @stop_item_names"""
        ]
        sdf = (
            fdf
            .query("item_nm_alias.str.lower() not in @stop_item_names")
            .query("(sim_score_token>=0.7 and sim_score_char>=0.8) or (sim_score_token>=0.1 and sim_score_char>=0.9)")
            # .query("item_nm_alias.str.contains('에이닷', case=False)")
            .query(' '.join(filtering_condition))
            .sort_values('sim_score_char', ascending=False)
            [['item_nm_alias','item_id','token_txt','token_txt_str','sim_score_token','sim_score_char','domain']]
        ).copy()
        sdf['rank_e'] = sdf.groupby(['item_nm_alias'])['sim_score_char'].rank(ascending=False, method='dense')#.reset_index(name='rank')
        sdf['rank_t'] = sdf.groupby(['token_txt_str'])['sim_score_char'].rank(ascending=False, method='dense')#.reset_index(name='rank')
        sdf = sdf.query("rank_t<=1 and rank_e<=1")[['item_nm_alias','item_id','token_txt_str','domain']].drop_duplicates()
        # sdf = sdf.groupby('item_nm_alias', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 2), random_state=42), include_groups=False)
        product_df = pd.concat([kdf,sdf]).drop_duplicates(['item_id','item_nm','item_nm_alias','domain']).groupby(["item_nm","item_nm_alias","item_id","domain"])['token_txt_str'].apply(list).reset_index(name='item_name_in_message').rename(columns={'item_nm':'item_name_in_voca'}).sort_values('item_name_in_voca')
        product_df['item_name_in_message'] = product_df['item_name_in_message'].apply(lambda x: ",".join(list(set([w.replace(',',' ') for w in x]))))
        product_df[['item_name_in_message','item_name_in_voca','item_id','domain']]#.query("item_name_in_voca.str.contains('netflix', case=False)").drop_duplicates(['item_name_in_voca']).sort_values('item_id')
        from langchain_anthropic import ChatAnthropic
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        import json
        import re
        from pygments import highlight
        from pygments.lexers import JsonLexer
        from pygments.formatters import HtmlFormatter
        from IPython.display import HTML
        ### Entity-Assisted
        # product_info = (",\n".join(product_df.apply(lambda x: f'"item_name_in_msg":"{x['item_name_in_msg']}", "item_name_in_voca":"{x['item_name_in_voca']}", "item_id":"{x['item_id']}:, "action":고객에게 기대하는 행동. [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 없음, 기타] 중에서 선택', axis=1).tolist()))
        # product_info = ", ".join(product_df['item_name_in_voca'].unique().tolist())
        product_info = ", ".join(product_df[['item_name_in_voca','domain']].apply(lambda x: x['item_name_in_voca']+"("+x['domain']+")", axis=1))
        # product_df = product_df.drop_duplicates(['item_name_in_message','item_name_in_voca'])
        # product_df = product_df.merge(product_df.groupby('item_name_in_message')['item_id'].size().reset_index(name='count').sort_values('count', ascending=False), on='item_name_in_message', how='left').query('count<=3')
        product_df = product_df[['item_name_in_voca','item_id','domain']].drop_duplicates()
        product_df['action'] = '고객에게 기대하는 행동. [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 없음, 기타] 중에서 선택'
        product_element = product_df.to_dict(orient='records') if product_df.shape[0]>0 else schema_ext['product']
        pgm_cand_info = "\n\t".join(pgm_pdf_tmp.iloc[:num_cand_pgms][['pgm_nm','clue_tag']].apply(lambda x: re.sub(r'\[.*?\]', '', x['pgm_nm'])+" : "+x['clue_tag'], axis=1).to_list())
        rag_context = f"\n### 광고 분류 기준 정보 ###\n\t{pgm_cand_info}" if num_cand_pgms>0 else ""
        schema_prd_1 = {
            "title": {
                "type": "string",
                'description': '광고 제목. 광고의 핵심 주제와 가치 제안을 명확하게 설명할 수 있도록 생성'
            },
            "purpose": {
                "type": "array",
                'description': '광고의 주요 목적을 다음 중에서 선택(복수 가능): [상품 가입 유도, 대리점 방문 유도, 웹/앱 접속 유도, 이벤트 응모 유도, 혜택 안내, 쿠폰 제공 안내, 경품 제공 안내, 기타 정보 제공]'
            },
            "product":
                product_element
            ,
            'channel': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'type': {'type': 'string', 'description': '채널 종류: [URL, 전화번호, 앱, 대리점] 중에서 선택'},
                        'value': {'type': 'string', 'description': '실제 URL, 전화번호, 앱 이름, 대리점 이름 등 구체적 정보'},
                        'action': {'type': 'string', 'description': '채널 목적: [방문, 접속, 가입, 추가 정보, 문의, 수신, 수신 거부] 중에서 선택'},
                        # 'benefit': {'type': 'string', 'description': '해당 채널 이용 시 특별 혜택'},
                        'store_code': {'type': 'string', 'description': "매장 코드 - tworldfriends.co.kr URL에서 D+숫자 9자리(D[0-9]{9}) 패턴의 코드 추출하여 대리점 채널에 설정"}
                    }
                },
            },
            'pgm':{
                'type': 'array',
                'description': '아래 광고 분류 기준 정보에서 선택. 메세지 내용과 광고 분류 기준을 참고하여, 광고 메세지에 가장 부합하는 2개의 pgm_nm을 적합도 순서대로 제공'
            }
        }
        # Improved extraction guidance
        extraction_guide = """
        ### 분석 목표 ###
        * Schema의 Product 태그 내에 action을 추출하세요.
        * Schema내 action 항목 외 태그 정보는 원본 그대로 두세요.
        ### 고려사항 ###
        * 상품 정보에 있는 항목을 임의로 변형하거나 누락시키지 마세요.
        * 광고 분류 기준 정보는 pgm_nm : clue_tag 로 구성
        ### JSON 응답 형식 ###
        응답은 설명 없이 순수한 JSON 형식으로만 제공하세요. 응답의 시작과 끝은 '{'와 '}'여야 합니다. 어떠한 추가 텍스트나 설명도 포함하지 마세요.
        """
        # Create the system message with clear JSON output requirements
        user_message = f"""당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다. 아래 schema에 따라 광고 메시지를 분석하여 완전하고 정확한 JSON 객체를 생성해 주세요:
        ### 분석 대상 광고 메세지 ###
        {mms_msg}
        ### 결과 Schema ###
        {json.dumps(schema_prd_1, indent=2, ensure_ascii=False)}
        {extraction_guide}
        {rag_context}
        """
        try:
            # Use OpenAI's ChatCompletion with the current API format
            response = client.chat.completions.create(
                model="skt/a.x-3-lg",  # Or your preferred OpenAI model
            # model="skt/claude-3-5-sonnet-20241022",
                messages = [
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=4000,
                top_p=0.95,  # Reduces randomness
                frequency_penalty=0.0,  # Avoid repetition in JSON
                presence_penalty=0.0,
                response_format={"type": "json_object"}  # Explicitly request JSON format
            )
            # Extract the JSON from the response
            result_json_text = response.choices[0].message.content
            json_objects = extract_json_objects(result_json_text)[0]
            pgm_json = pgm_pdf[pgm_pdf['pgm_nm'].apply(lambda x: re.sub(r'\[.*?\]', '', x) in ' '.join(json_objects['pgm']))][['pgm_nm','pgm_id']].to_dict('records')
            final_json = json_objects.copy()
            final_json['pgm'] = pgm_json
        except Exception as e:
            print(f"Error with API call: {e}")
        print(json.dumps(final_json, indent=4, ensure_ascii=False))
        ### LLM-only
        schema_prd_2 = schema_ext
        extraction_guide = """
        ### 분석 시 고려사항 ###
        * 하나의 광고에 여러 상품이 포함될 수 있으며, 각 상품별로 별도 객체 생성
        * 재현율이 높도록 모든 상품을 선택
        * 상품 후보 정보는 상품 이름 (도메인) 형식으로 제공
        * 광고 분류 기준 정보는 pgm_nm : clue_tag 로 구성
        ### 분석 목표 ###
        * 텍스트 매칭 기법으로 만들어진 상품 후보 정보가 제공되면 이를 확인하여 참고하라.
        * 제공된 상품 이름이 적합하지 않으면 무시하고, 목록에 없어도 적합한 상품이 있으면 추출하세요.
        ### JSON 응답 형식 ###
        응답은 설명 없이 순수한 JSON 형식으로만 제공하세요. 응답의 시작과 끝은 '{'와 '}'여야 합니다. 어떠한 추가 텍스트나 설명도 포함하지 마세요.
        """
        # product_info = ", ".join(product_df['item_name_in_voca'].unique().tolist())
        product_info = ", ".join(product_df[['item_name_in_voca','domain']].apply(lambda x: x['item_name_in_voca']+"("+x['domain']+")", axis=1))
        rag_context = f"### 상품 후보 정보 ###\n\t{product_info}" if product_df.shape[0]>0 else ""
        pgm_cand_info = "\n\t".join(pgm_pdf_tmp.iloc[:num_cand_pgms][['pgm_nm','clue_tag']].apply(lambda x: re.sub(r'\[.*?\]', '', x['pgm_nm'])+" : "+x['clue_tag'], axis=1).to_list())
        rag_context += f"\n\n### 광고 분류 기준 정보 ###\n\t{pgm_cand_info}" if num_cand_pgms>0 else ""
        # Create the system message with clear JSON output requirements
        user_message = f"""당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다. 아래 schema에 따라 광고 메시지를 분석하여 완전하고 정확한 JSON 객체를 생성해 주세요:
        ### 분석 대상 광고 메세지 ###
        {mms_msg}
        ### 결과 Schema ###
        {json.dumps(schema_prd_2, indent=2, ensure_ascii=False)}
        {extraction_guide}
        {rag_context}
        """
        try:
            # Use OpenAI's ChatCompletion with the current API format
            response = client.chat.completions.create(
                model="skt/a.x-3-lg",  # Or your preferred OpenAI model
            #   model="skt/claude-3-5-sonnet-20241022",
                messages = [
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=4000,
                top_p=0.95,  # Reduces randomness
                frequency_penalty=0.0,  # Avoid repetition in JSON
                presence_penalty=0.0,
                response_format={"type": "json_object"}  # Explicitly request JSON format
            )
            # Extract the JSON from the response
            result_json_text = response.choices[0].message.content
            json_objects = extract_json_objects(result_json_text)[0]
        except Exception as e:
            print(f"Error with API call: {e}")
        matches = []
        for item_name_message in json_objects['product']:
            matches.extend(find_entities_in_text(
                item_name_message['name'],
                entity_list_for_fuzzy,
                min_similarity=50,
                high_score_threshold=50,
                overlap_tolerance=0.5
            ))
        mdf = pd.DataFrame(matches)
        if len(matches)>0:
            mdf = mdf.query("text.str.lower() not in @stop_item_names and matched_entity.str.lower() not in @stop_item_names")
        if mdf.shape[0]>0:
            mdf['item_id'] = mdf['data'].apply(lambda x: x['item_id'])
            mdf['domain'] = mdf['data'].apply(lambda x: x['domain'])
            mdf = mdf.query("not matched_entity.str.contains('test', case=False)").drop_duplicates(['item_id','domain'])
            mdf = mdf.merge(mdf.groupby(['text','start'])['end'].max().reset_index(name='end'), on=['text', 'start', 'end'])
            mdf['rank'] = mdf['data'].apply(lambda x: x['rank'])
            mdf['re_rank'] = mdf.groupby('text')['score'].rank(ascending=False)
            mdf = mdf.query("re_rank<=2")
            mdf = mdf.merge(pd.DataFrame(json_objects['product']).rename(columns={'name':'text'}), on='text', how='left')
            product_tag = mdf.rename(columns={'text':'item_name_in_message','matched_entity':'item_name_in_voca'})[['item_name_in_message','item_name_in_voca','item_id','domain']].drop_duplicates().to_dict(orient='records')
            final_result = {
                "title":json_objects['title'],
                "purpose":json_objects['purpose'],
                "product":product_tag,
                "channel":json_objects['channel'],
                "pgm":json_objects['pgm']
            }
        else:
            final_result = json_objects
            final_result['product'] = [{'item_name_in_message':d['name'], 'item_name_in_voca':d['name'], 'item_id': '#', 'domain': '#'} for d in final_result['product']]
        if num_cand_pgms>0:
            pgm_json = pgm_pdf[pgm_pdf['pgm_nm'].apply(lambda x: re.sub(r'\[.*?\]', '', x) in ' '.join(json_objects['pgm']))][['pgm_nm','pgm_id']].to_dict('records')
            final_result['pgm'] = pgm_json
        print(json.dumps(final_result, indent=4, ensure_ascii=False))
        ### LLM-COT
        schema_prd_3 = {
        "reasoning": {
            "type": "object",
            "description": "단계별 분석 과정 (최종 JSON에는 포함하지 않음)",
            "properties": {
            "step1_purpose_analysis": "광고 목적 분석 과정",
            "step2_product_identification": "상품 식별 및 도메인 매칭 과정",
            "step3_channel_extraction": "채널 정보 추출 과정",
            "step4_pgm_classification": "프로그램 분류 과정"
            }
        },
        "title": {
            "type": "string",
            "description": "광고 제목. 광고의 핵심 주제와 가치 제안을 명확하게 설명"
        },
        "purpose": {
            "type": "array",
            "description": "STEP 1에서 분석한 광고의 주요 목적 (복수 가능)"
        },
        "product": {
            "type": "array",
            "items": {
            "type": "object",
            "properties": {
                "name": {
                "type": "string",
                "description": "STEP 2에서 식별한 제품/서비스 이름"
                },
                "action": {
                "type": "string",
                "description": "STEP 2-3에서 결정한 고객 기대 행동"
                },
                "domain_match": {
                "type": "string",
                "description": "매칭된 상품 후보의 도메인 정보 (참고용, 최종 JSON에는 제외)"
                }
            }
            }
        },
        "channel": {
            "type": "array",
            "items": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "value": {"type": "string"},
                "action": {"type": "string"},
                "benefit": {"type": "string"},
                "store_code": {"type": "string"}
            }
            }
        },
        "pgm": {
            "type": "array",
            "description": "STEP 4에서 선택한 프로그램 분류 (적합도 순 2개)"
        }
        }
        extraction_guide = """
        ## 분석 지침
        1. **재현율 우선**: 광고에서 언급된 모든 상품을 누락 없이 추출
        2. **도메인 활용**: 상품 후보의 도메인 정보를 적극 활용하여 정확한 매칭 수행
        3. **목적 기반 추론**: 광고 목적을 명확히 파악한 후 다른 요소들을 일관성 있게 분석
        4. **컨텍스트 고려**: 제공된 상품 후보가 부적합하면 무시하고, 누락된 중요 상품이 있으면 추가
        ## JSON 응답 형식
        - reasoning 섹션은 분석 과정 설명용이며 최종 JSON에는 포함하지 않음
        - 순수한 JSON 형식으로만 응답
        - 시작과 끝은 '{'와 '}'
        - 추가 텍스트나 설명 없이 JSON만 제공
        """
        # product_info = ", ".join(product_df['item_name_in_voca'].unique().tolist())
        product_info = ", ".join(product_df[['item_name_in_voca','domain']].apply(lambda x: x['item_name_in_voca']+"("+x['domain']+")", axis=1))
        rag_context = f"### 상품 후보 정보 ###\n\t{product_info}" if product_df.shape[0]>0 else ""
        pgm_cand_info = "\n\t".join(pgm_pdf_tmp.iloc[:num_cand_pgms][['pgm_nm','clue_tag']].apply(lambda x: re.sub(r'\[.*?\]', '', x['pgm_nm'])+" : "+x['clue_tag'], axis=1).to_list())
        rag_context += f"\n\n### 광고 분류 기준 정보 ###\n\t{pgm_cand_info}" if num_cand_pgms>0 else ""
        # Create the system message with clear JSON output requirements
        user_message = f"""당당신은 SKT 캠페인 메시지에서 정확한 정보를 추출하는 전문가입니다. **단계별 사고 과정(Chain of Thought)**을 통해 광고 메시지를 분석하여 완전하고 정확한 JSON 객체를 생성해 주세요.
        ## 분석 단계 (Chain of Thought)
        ### STEP 1: 광고 목적(Purpose) 분석
        먼저 광고 메시지 전체를 읽고 다음 질문들에 답하여 광고의 주요 목적을 파악하세요:
        - 이 광고가 고객에게 무엇을 하라고 요구하는가?
        - 어떤 행동을 유도하려고 하는가? (가입, 방문, 다운로드, 참여 등)
        - 어떤 혜택이나 정보를 제공하고 있는가?
        **목적 후보**: [상품 가입 유도, 대리점 방문 유도, 웹/앱 접속 유도, 이벤트 응모 유도, 혜택 안내, 쿠폰 제공 안내, 경품 제공 안내, 기타 정보 제공]
        ### STEP 2: 상품(Product) 식별 및 도메인 매칭
        파악된 목적을 바탕으로 다음 과정을 거쳐 상품을 식별하세요:
        **2-1. 광고 메시지에서 언급된 모든 상품/서비스 추출**
        - 직접적으로 언급된 상품명을 모두 나열
        - 묵시적으로 언급된 서비스나 혜택도 포함
        **2-2. RAG Context의 상품 후보 정보와 도메인 매칭**
        - 각 추출된 상품을 상품 후보 정보와 비교
        - 도메인 정보(product, subscription_service 등)를 고려하여 가장 적합한 매칭 수행
        - 상품 후보에 없어도 광고에서 중요하게 다뤄지는 상품이 있다면 추가
        **2-3. 각 상품별 고객 행동(Action) 결정**
        - STEP 1에서 파악한 목적과 연결하여 각 상품에 대한 기대 행동 결정
        - 행동 후보: [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타]
        ### STEP 3: 채널(Channel) 및 기타 정보 추출
        - URL, 전화번호, 앱, 대리점 정보 추출
        - 각 채널의 목적과 혜택 파악
        - 대리점 URL에서 매장 코드(D[0-9]{9}) 패턴 확인
        ### STEP 4: 프로그램 분류(PGM) 결정
        - 광고 분류 기준 정보의 키워드와 메시지 내용 매칭
        - 적합도 순서대로 2개 선택
        ### 분석 대상 광고 메세지 ###
        {mms_msg}
        {rag_context}
        ### 결과 Schema ###
        {json.dumps(schema_prd_3, indent=2, ensure_ascii=False)}
        {extraction_guide}
        """
        try:
            # Use OpenAI's ChatCompletion with the current API format
            response = client.chat.completions.create(
                model="skt/a.x-3-lg",  # Or your preferred OpenAI model
            #   model="skt/claude-3-5-sonnet-20241022",
                messages = [
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=4000,
                top_p=0.95,  # Reduces randomness
                frequency_penalty=0.0,  # Avoid repetition in JSON
                presence_penalty=0.0,
                response_format={"type": "json_object"}  # Explicitly request JSON format
            )
            # Extract the JSON from the response
            result_json_text = response.choices[0].message.content
            json_objects = extract_json_objects(result_json_text)[0]
        except Exception as e:
            print(f"Error with API call: {e}")
        matches = []
        for item_name_message in json_objects['product']:
            matches.extend(find_entities_in_text(
                item_name_message['name'],
                entity_list_for_fuzzy,
                min_similarity=50,
                high_score_threshold=50,
                overlap_tolerance=0.5
            ))
        mdf = pd.DataFrame(matches)
        if len(matches)>0:
            mdf = mdf.query("text.str.lower() not in @stop_item_names and matched_entity.str.lower() not in @stop_item_names")
        if mdf.shape[0]>0:
            mdf['item_id'] = mdf['data'].apply(lambda x: x['item_id'])
            mdf['domain'] = mdf['data'].apply(lambda x: x['domain'])
            mdf = mdf.query("not matched_entity.str.contains('test', case=False)").drop_duplicates(['item_id','domain'])
            mdf = mdf.merge(mdf.groupby(['text','start'])['end'].max().reset_index(name='end'), on=['text', 'start', 'end'])
            mdf['rank'] = mdf['data'].apply(lambda x: x['rank'])
            mdf['re_rank'] = mdf.groupby('text')['score'].rank(ascending=False)
            mdf = mdf.query("re_rank<=2")
            mdf = mdf.merge(pd.DataFrame(json_objects['product']).rename(columns={'name':'text'}), on='text', how='left')
            product_tag = mdf.rename(columns={'text':'item_name_in_message','matched_entity':'item_name_in_voca'})[['item_name_in_message','item_name_in_voca','item_id','domain']].drop_duplicates().to_dict(orient='records')
            final_result = {
                "title":json_objects['title'],
                "purpose":json_objects['purpose'],
                "product":product_tag,
                "channel":json_objects['channel'],
                "pgm":json_objects['pgm']
            }
        else:
            final_result = json_objects
            final_result['product'] = [{'item_name_in_message':d['name'], 'item_name_in_voca':d['name'], 'item_id': '#', 'domain': '#'} for d in final_result['product']]
        if num_cand_pgms>0:
            pgm_json = pgm_pdf[pgm_pdf['pgm_nm'].apply(lambda x: re.sub(r'\[.*?\]', '', x) in ' '.join(json_objects['pgm']))][['pgm_nm','pgm_id']].to_dict('records')
            final_result['pgm'] = pgm_json
        print(json.dumps(final_result, indent=4, ensure_ascii=False))
if __name__ == "__main__":
    main()