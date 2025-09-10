# MMS 광고 텍스트 추출기 시스템

---

## 1: 프로젝트 개요
**MMS 광고 텍스트 추출기란?**
- **목적**: 통신사 MMS 광고 메시지에서 구조화된 정보를 자동 추출
- **핵심 기능**: 
  - 제품/서비스명 추출
  - 광고 목적 분류
  - 채널 정보 파싱
  - 프로그램 분류
  - 엔티티 관계 그래프 생성 (DAG)
- **기술 스택**: Python, LLM (GPT/Claude/Gemini), NetworkX, Kiwi 형태소분석기
- **처리 방식**: 단일/배치 처리 모두 지원

---

## 2: 시스템 초기화 과정
**MMSExtractor 초기화 단계별 진행**

**Step 1: 기본 설정 적용**
```python
def _set_default_config(self, ...):
    self.data_dir = data_dir or './data/'
    self.model_path = model_path or 'jhgan/ko-sroberta-multitask'
    self.product_info_extraction_mode = 'nlp'  # nlp/llm/rag
    self.entity_extraction_mode = 'llm'        # logic/llm
```

**Step 2: 디바이스 및 모델 초기화**
- GPU/MPS/CPU 디바이스 자동 감지
- LLM 모델 로드 (temperature=0, seed=42로 일관성 보장)
- 임베딩 모델 로드 (SentenceTransformer)
- Kiwi 형태소분석기 초기화

**Step 3: 데이터 로딩**
- 상품 정보, 별칭 규칙, 불용어, 프로그램 정보 순차 로드

---

## 3: 데이터 로딩 과정 (Step 3 상세)
**데이터 소스별 로딩 방식**

**로컬 모드 (CSV 파일)**
```python
def _load_item_data(self):
    item_pdf_raw = pd.read_csv('./data/items.csv')
    self.item_pdf_all = item_pdf_raw.drop_duplicates(['item_nm','item_id'])
    # 추가 컬럼 생성 및 소문자 변환
```

**데이터베이스 모드 (Oracle)**
```python
def _load_item_from_database(self):
    sql = "SELECT * FROM TCAM_RC_OFER_MST"
    self.item_pdf_all = pd.read_sql(sql, conn)
    # LOB 데이터 강제 로드 처리
```

**별칭 규칙 적용**
- 상품명에 대한 다양한 표기법 매핑
- `explode()` 함수로 별칭 확장: 병렬 처리 (one alias per row)

**Kiwi에 상품명 등록**
- 추출된 모든 상품명을 고유명사(NNP)로 등록

---

## 4: 메시지 처리 워크플로우
**process_message() 메서드의 8단계 처리 과정**

```python
def process_message(self, mms_msg: str):
    # 1단계: 엔티티 추출
    cand_item_list, extra_item_pdf = self._extract_entities(msg)
    
    # 2단계: 프로그램 분류  
    pgm_info = self._classify_programs(msg)
    
    # 3단계: RAG 컨텍스트 구성
    rag_context = self._build_rag_context(pgm_info, cand_item_list)
    
    # 4단계: 제품 정보 준비
    product_element = self._prepare_product_info(extra_item_pdf)
    
    # 5단계: LLM 프롬프트 구성 및 실행
    result_json_text = self._safe_llm_invoke(prompt)
    
    # 6단계: JSON 파싱
    json_objects = extract_json_objects(result_json_text)
    
    # 7단계: 최종 결과 구성
    final_result = self._build_final_result(json_objects, msg, pgm_info)
    
    # 8단계: 결과 검증
    return self._validate_extraction_result(final_result)
```

---

## 5: 1단계 - 엔티티 추출 상세
**extract_entities_from_kiwi() 메서드**

**형태소 분석 및 문장 분할**
```python
# 1. 문장 분할 (하위 문장 포함. 병력 처리 및 길이 정규화 이슈)
sentences = self.kiwi.split_into_sents(re.split(r"_+", mms_msg))

# 2. 제외 패턴 적용하여 문장 필터링
sentence_list = [filter_text_by_exc_patterns(sent, self.exc_tag_patterns) 
                for sent in sentences_all]

# 3. 고유명사(NNP) 추출
entities_from_kiwi = [token.form for token in result_msg 
                     if token.tag == 'NNP' and len(token.form) >= 2]
```

**병렬 유사도 매칭**
- Fuzzy 매칭으로 1차 후보 추출
- 시퀀스 유사도로 2차 정밀 매칭
- 임계값 기반 필터링 (fuzzy: 0.5, sequence: 0.2)

---

## 6: Fuzzy 매칭 알고리즘 상세
**RapidFuzz 라이브러리의 4가지 매칭 방식**

**1. Ratio (전체 유사도)**
```python
# 예시: "갤럭시S24" vs "갤럭시 S24"
fuzz.ratio("갤럭시S24", "갤럭시 S24")  # → 86점
# 전체 문자열 대비 편집거리 기반 계산
```

**2. Partial Ratio (부분 매칭)**
```python
# 예시: "아이폰15프로" vs "아이폰15프로맥스"  
fuzz.partial_ratio("아이폰15프로", "아이폰15프로맥스")  # → 100점
# 짧은 문자열이 긴 문자열에 완전 포함될 때 높은 점수
```

**3. Token Sort Ratio (토큰 정렬 후 비교)**
```python
# 예시: "갤럭시 S24 울트라" vs "울트라 갤럭시 S24"
fuzz.token_sort_ratio("갤럭시 S24 울트라", "울트라 갤럭시 S24")  # → 100점
# 단어 순서가 달라도 동일한 토큰 구성이면 매칭
```

**4. Token Set Ratio (토큰 집합 비교)**
```python
# 예시: "아이폰15 프로 256GB" vs "아이폰15 프로"
fuzz.token_set_ratio("아이폰15 프로 256GB", "아이폰15 프로")  # → 86점
# 공통 토큰과 차이점을 분석하여 유사도 계산
```

---

## 7: 시퀀스 유사도 알고리즘
**3가지 알고리즘을 조합한 종합 유사도**

**1. Substring Aware Similarity (부분문자열 인식)**
```python
def substring_aware_similarity(s1, s2):
    if s1 in s2 or s2 in s1:  # 포함 관계가 있으면
        base_score = len(shorter) / len(longer)
        return min(0.95 + base_score * 0.05, 1.0)  # 높은 점수 부여
    return longest_common_subsequence_ratio(s1, s2)
```

**2. Sequence Matcher (Python 표준 라이브러리)**
- `difflib.SequenceMatcher`를 사용한 매칭 블록 분석
- 문자 단위 정확한 일치 부분 계산

**3. Token Sequence (토큰 기반 LCS)**
```python
# "갤럭시 S24 플러스" → ["갤럭시", "S24", "플러스"]
# LCS 알고리즘으로 공통 토큰 순서 찾기
token_lcs_length = longest_common_subsequence(tokens1, tokens2)
```

**가중 평균 계산**
- Substring: 40%, Sequence Matcher: 40%, Token Sequence: 20%

---

## 8: 2단계 - 프로그램 분류
**_classify_programs() 메서드**

**임베딩 기반 의미적 유사도 계산**
```python
def _classify_programs(self, mms_msg: str):
    # 1. 메시지 임베딩
    mms_embedding = self.emb_model.encode([mms_msg.lower()])
    
    # 2. 프로그램 임베딩과 코사인 유사도 계산
    similarities = torch.nn.functional.cosine_similarity(
        mms_embedding, self.clue_embeddings, dim=1
    )
    
    # 3. 상위 후보 선별 (기본 5개)
    pgm_pdf_tmp = self.pgm_pdf.copy()
    pgm_pdf_tmp['sim'] = similarities
    top_candidates = pgm_pdf_tmp.sort_values('sim', ascending=False)
    
    return {
        "pgm_cand_info": candidates_text,
        "similarities": similarities,
        "pgm_pdf_tmp": pgm_pdf_tmp
    }
```

**프로그램 DB 구조**
- `pgm_nm`: 프로그램명
- `clue_tag`: 분류 키워드
- 임베딩: 프로그램명 + 분류태그 결합

---

## 9: 3-4단계 - RAG 컨텍스트 및 제품 정보 구성
**모드별 RAG 컨텍스트 구성**

**RAG 모드**: 후보 상품을 강제 참조
```python
if self.product_info_extraction_mode == 'rag':
    rag_context += f"\n\n### 후보 상품 이름 목록 ###\n\t{cand_item_list}"
```

**LLM 모드**: 참고용 후보 목록 제공
```python
elif self.product_info_extraction_mode == 'llm':
    rag_context += f"\n\n### 참고용 후보 상품 이름 목록 ###\n\t{cand_item_list}"
```

**NLP 모드**: 구조화된 제품 요소 생성
```python
elif self.product_info_extraction_mode == 'nlp':
    product_df = extra_item_pdf.rename(columns={'item_nm': 'name'})
    product_df['action'] = '기타'
    product_element = product_df.to_dict(orient='records')
```

**프로그램 분류 정보 추가**
```python
rag_context = f"\n### 광고 분류 기준 정보 ###\n\t{pgm_info['pgm_cand_info']}"
```

---

## 10: 5단계 - LLM 프롬프트 구성 및 실행
**구조화된 JSON 스키마 기반 프롬프트**

**핵심 출력 스키마**
```json
{
  "title": "광고 제목",
  "purpose": ["상품 가입 유도", "혜택 안내", "정보 제공"],
  "product": [{"name": "제품명", "action": "가입"}],
  "channel": [{"type": "URL", "value": "링크", "action": "가입"}],
  "pgm": ["프로그램 분류"]
}
```

**안전한 LLM 호출**
```python
def _safe_llm_invoke(self, prompt: str, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = self.llm_model.invoke(prompt)
            result_text = response.content
            
            # 스키마 응답 감지 및 재시도
            if self._detect_schema_response(result_text):
                enhanced_prompt = self._enhance_prompt_for_retry(prompt)
                response = self.llm_model.invoke(enhanced_prompt)
                
            return result_text
        except Exception as e:
            if attempt == max_retries - 1:
                return self._fallback_extraction(prompt)
            time.sleep(2 ** attempt)  # 지수 백오프
```

---

## 11: 6-7단계 - JSON 파싱 및 결과 구성
**JSON 객체 추출 및 검증**

```python
def extract_json_objects(text):
    # 정규표현식으로 JSON 패턴 찾기
    pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
    
    for match in re.finditer(pattern, text):
        potential_json = match.group(0)
        try:
            # JSON 복구 및 파싱
            cleaned_json = repair_json(potential_json)
            json_obj = ast.literal_eval(cleaned_json)
            result.append(json_obj)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
    return result
```

**최종 결과 구성**
```python
def _build_final_result(self, json_objects, msg, pgm_info):
    # 1. 엔티티 매칭 (logic vs llm 모드)
    if self.entity_extraction_mode == 'logic':
        similarities = self.extract_entities_by_logic(cand_entities)
    else:
        similarities = self.extract_entities_by_llm(msg)
    
    # 2. 상품 정보 매핑
    final_result['product'] = self._map_products_with_similarity(similarities)
    
    # 3. 프로그램 분류 매핑
    final_result['pgm'] = self._map_program_classification(json_objects, pgm_info)
    
    # 4. 채널 정보 처리
    final_result['channel'] = self._extract_channels(json_objects, msg)
```

---

## 12: 8단계 - 결과 검증 및 DAG 추출
**결과 검증 과정**
```python
def _validate_extraction_result(self, result):
    # 1. 필수 필드 확인
    required_fields = ['title', 'purpose', 'product', 'channel']
    
    # 2. 상품명 길이 검증 (2자 이상)
    validated_products = [p for p in result.get('product', []) 
                         if len(p.get('name', '')) >= 2]
    
    # 3. 불용어 필터링
    validated_products = [p for p in validated_products 
                         if p.get('name') not in self.stop_item_names]
    
    return result
```

**선택적 DAG 추출**
```python
if self.extract_entity_dag:
    # 1. LLM을 통한 엔티티 관계 추출
    dag_result = extract_dag(DAGParser(), msg, self.llm_model)
    
    # 2. NetworkX 그래프 생성
    dag = dag_result['dag']
    
    # 3. 시각화 다이어그램 생성
    create_dag_diagram(dag, filename=f'dag_{sha256_hash(msg)}')
    
    # 4. 결과에 DAG 정보 추가
    final_result['entity_dag'] = dag_result['dag_section'].split('\n')
```

---

## 13: Entity Linking - 제품명 ID 연결
**LLM 추출 제품명을 상품 DB와 연결하는 후처리 과정**

**Entity Linking 프로세스**
```python
def _map_products_with_similarity(self, similarities_fuzzy, json_objects):
    # 1. LLM이 추출한 제품명 목록 수집
    product_items = json_objects.get('product', [])
    extracted_names = [item.get('name', '') for item in product_items]
    
    # 2. 높은 유사도 아이템 필터링 (임계값: 1.5)
    high_sim_items = similarities_fuzzy.query('sim >= 1.5')['item_nm_alias'].unique()
    
    # 3. 상품 DB와 매칭하여 ID 연결
    matched_products = self.item_pdf_all.merge(similarities_fuzzy, on=['item_nm_alias'])
    
    # 4. JSON 구조로 변환
    product_tag = convert_df_to_json_list(matched_products)
    
    return product_tag
```

**매칭 결과 JSON 구조**
```json
{
  "item_name_in_msg": "갤럭시S24",           # LLM이 추출한 원본 이름
  "expected_action": "가입",                 # LLM이 추출한 기대 행동
  "item_in_voca": [                          # DB 매칭 결과
    {
      "item_nm": "Galaxy S24",               # DB의 정식 상품명
      "item_id": ["GXYS24_001", "GXYS24_002"] # 연결된 상품 ID 목록
    }
  ]
}
```

**Entity Linking의 핵심 도전과제**
- **변형 표기**: "갤럭시S24" ↔ "Galaxy S24" ↔ "갤S24"
- **부분 매칭**: "아이폰15프로" ↔ "아이폰15프로맥스"  
- **다중 매핑**: 하나의 제품명이 여러 상품 ID에 대응
- **임계값 조정**: 너무 낮으면 오매칭, 너무 높으면 누락

**품질 보장 메커니즘**
```python
# 1. 불용어 제거
filtered_products = [p for p in products 
                    if p['item_name_in_msg'] not in self.stop_item_names]

# 2. 최소 길이 검증 (2자 이상)
valid_products = [p for p in filtered_products 
                 if len(p['item_name_in_msg']) >= 2]

# 3. 순위 기반 필터링 (상위 5개 후보)
ranked_products = similarities.query("rank <= 5")
```

---

## 14: 사용법 및 실행 예제
**기본 사용법**
```python
# 추출기 초기화 (실제 작업 순서 반영)
extractor = MMSExtractor(
    offer_info_data_src='db',           # 데이터 소스
    product_info_extraction_mode='llm', # 제품 추출 모드
    entity_extraction_mode='logic',     # 엔티티 매칭 방식
    llm_model='gpt',                   # LLM 모델
    extract_entity_dag=True            # DAG 추출 여부
)

# 단일 메시지 처리 (8단계 순차 실행)
result = extractor.process_message(mms_message)

# 배치 처리 (멀티프로세싱)
results = process_messages_batch(extractor, messages, max_workers=4)
```

**명령줄 실행**
```bash
# 단일 메시지 처리
python mms_extractor.py --message "광고 메시지" --extract-entity-dag

# 배치 파일 처리  
python mms_extractor.py --batch-file messages.txt --max-workers 8

# DB 모드로 DAG 추출
python mms_extractor.py --batch-file messages.txt --offer-data-source db --extract-entity-dag
```