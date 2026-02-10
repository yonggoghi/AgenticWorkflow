# Entity Extraction Prompt Improvement Guide

## Key Insight: Why High Recall → Better Final Results

**Key Finding**: High recall in the extraction stage, combined with effective heuristic methods between extraction and linking, leads to better final linked results. Both precision and recall matter in the extraction stage.

**Lesson Learned**: Aggressive exclusion rules in prompts hurt recall too much. The two-stage architecture relies on high recall in Stage 1 for optimal final results.

### Two-Stage Filtering Architecture

The pipeline employs **intentional over-extraction followed by LLM-based refinement**:

```
High Recall Extraction (CLD: 90%)
         ↓
    More candidates in pool
         ↓
    Higher chance correct product is present
         ↓
    Stage 2 LLM selects best match with context
         ↓
    Better Linked F1 (CLD: 95.9%)
```

**Stage 1 (Extraction)**: Intentionally over-extracts with LOW thresholds
- Fuzzy threshold: 0.6
- Keeps top 50-100 candidates per entity
- Goal: Capture ALL possible product mentions (high recall)

**Stage 2 (Linking/Re-validation)**: LLM-based filtering with context
- Uses DAG/PAIRING context from Stage 1
- Disambiguates false positives using message context
- Goal: Precision recovery from abundant options

### Heuristic Methods Between Extraction and Linking

| Method | Location | Purpose |
|--------|----------|---------|
| **N-gram Expansion** | entity_recognizer.py:756 | Captures compound names ("아이폰 17 Pro Max" → 2/3/4-grams) |
| **Triple Similarity Stack** | entity_recognizer.py:859-932 | Fuzzy + Sequence(s1) + Sequence(s2) combined |
| **Progressive Threshold** | entity_recognizer.py:904-920 | Filters: combined_sim → high_sim → rank_limit |
| **Alias Expansion** | result_builder.py:279-303 | Merges extracted entities with alias rules |

### Filtering Logic Summary

| Stage | Component | Threshold | Purpose | Recall Impact |
|-------|-----------|-----------|---------|---------------|
| **1: Extraction** | Fuzzy similarity | 0.5-0.6 | Initial candidate pool | HIGH |
| **1: Pre-Link** | Sequence s1+s2 | >= 0.2 each | Broad similarity filtering | HIGH |
| **1: Pre-Link** | High similarity | >= 1.0 (s1+s2) | Baseline quality threshold | MEDIUM |
| **1: Pre-Link** | Rank limit | Top 50-100 per entity | Candidate diversity | HIGH |
| **2: LLM Re-validation** | Context matching | Custom (LLM) | Remove contradictions | MEDIUM |
| **2: Post-Linking** | Alias expansion rules | type != 'expansion' | DB reconciliation | HIGH |

### Why This Architecture Works

1. **Recall Preservation**: Stage 1 ensures no true products are missed
2. **Precision Recovery**: Stage 2's context-aware filtering removes false positives
3. **Relationship Awareness**: DAG/PAIRING context enables LLM to understand product relationships
4. **Abundant Options**: Stage 2 LLM filtering is highly effective when given diverse candidates

---

## Prompt Improvement Guidelines

Based on evaluation findings:

1. **DO NOT add aggressive exclusion rules** - They hurt recall too much
2. **Focus on Stage 2 context utilization** - This is where precision is recovered
3. **Maintain low thresholds in Stage 1** - High recall is intentional
4. **Test any changes with full evaluation** - Use generate_entity_extraction_eval.py

---

## Prompt Template for New Sessions

아래 프롬프트를 새 세션에서 사용하여 엔티티 추출 로직을 개선합니다.

---

## 프롬프트 (새 세션에 복사하여 사용)

```
# Task: Improve Entity Extraction Prompts Based on Human Evaluation Feedback

## Objective
어노테이터가 작성한 정답(correct_extracted_entities, correct_linked_entities)을 분석하여
엔티티 추출 프롬프트와 관련 로직을 개선합니다.

## Evaluation Data
평가 데이터 파일: outputs/entity_extraction_eval_20260205_180307.csv

파일 구조:
- mms: 원본 MMS 메시지
- extracted_entities_ax: A.X(SKT) LLM 1차 추출 결과
- linked_entities_ax: A.X(SKT) LLM 추출 후 DB 매칭 결과
- extracted_entities_cld: Claude LLM 1차 추출 결과
- linked_entities_cld: Claude LLM 추출 후 DB 매칭 결과
- correct_extracted_entities: 어노테이터가 작성한 정답 (추출)
- correct_linked_entities: 어노테이터가 작성한 정답 (링크) - 옵션

엔티티는 " | " (파이프)로 구분됩니다.

## Target Files to Improve

### 1. prompts/entity_extraction_prompt.py

#### HYBRID_DAG_EXTRACTION_PROMPT (1차 추출 프롬프트)
- MMS 메시지에서 상품/서비스 엔티티를 추출하는 1차 프롬프트
- DAG(Directed Acyclic Graph) 구조로 사용자 행동 경로 분석
- Root Node 우선순위: Store → Service → Event → App → Product

주요 섹션:
- Root Node Selection Hierarchy
- DAG Construction Rules
- Strict Exclusions
- Output Format (ENTITY, DAG)

#### build_context_based_entity_extraction_prompt() (2차 필터링 프롬프트)
- 1차 추출 결과를 vocabulary와 매칭하여 필터링
- context_keyword에 따라 다른 프롬프트 생성 ('DAG', 'PAIRING', 'ONT', None)
- 핵심 지침: vocabulary에 있는 엔티티만 정확히 일치하는 문자열로 반환

### 2. services/entity_recognizer.py

#### extract_entities_with_llm() 메서드
- 2단계 LLM 추출 파이프라인 구현
- Stage 1: HYBRID_DAG_EXTRACTION_PROMPT로 초기 추출
- Stage 2: build_context_based_entity_extraction_prompt()로 vocabulary 필터링

#### _match_entities_with_products() 메서드
- 추출된 엔티티를 상품 DB와 매칭
- Fuzzy similarity (threshold 0.6) + Sequence similarity
- 최종 linked entities 생성

## Analysis Steps

### Step 1: Load and Analyze Evaluation Data
1. outputs/entity_extraction_eval_20260205_180307.csv 파일 읽기
2. 각 row에 대해 분석:
   - extracted_entities_ax vs correct_extracted_entities 비교
   - extracted_entities_cld vs correct_extracted_entities 비교
   - linked_entities_ax vs correct_linked_entities 비교
   - linked_entities_cld vs correct_linked_entities 비교 (옵션. 없을 수도 있음)

### Step 2: Error Pattern Analysis
에러 유형 분류:
1. False Positive (과잉 추출): LLM이 추출했지만 정답에 없음
   - 예: SKT 고객센터, 전용 앱, 안심번호 등 일반 명사
2. False Negative (누락): 정답에 있지만 LLM이 추출 못함
   - 예: 특정 요금제명, 서비스명
3. Linking Error: 추출은 맞지만 DB 매칭 실패
   - 예: "050 넘버플러스" → "넘버플러스" 매칭 실패

### Step 3: Identify Improvement Opportunities
분석 결과를 바탕으로:
1. HYBRID_DAG_EXTRACTION_PROMPT 개선점 도출
   - Exclusion 규칙 강화/완화
   - Root Node 우선순위 조정
   - 예시 추가/수정
2. build_context_based_entity_extraction_prompt() 개선점 도출
   - 핵심 지침 명확화
   - Guidelines 추가/수정
3. entity_recognizer.py 로직 개선점 도출
   - Fuzzy threshold 조정
   - 매칭 알고리즘 개선

### Step 4: Propose and Implement Changes
1. 프롬프트 수정안 제시 (before/after)
2. 사용자 승인 후 수정 적용
3. 테스트 실행하여 개선 효과 확인

## Key Metrics to Improve

### Extraction Metrics
- Precision: 추출한 것 중 정답 비율 (False Positive 감소)
- Recall: 정답 중 추출한 비율 (False Negative 감소)
- F1 Score: Precision과 Recall의 조화평균

### Linking Metrics
- Link Accuracy: 추출된 엔티티 중 올바르게 링크된 비율

## Constraints
1. 프롬프트 수정 시 기존 구조 유지 (Output Format 등)
2. 한국어/영어 혼용 텍스트 처리 고려
3. "DO NOT TRANSLATE" 규칙 유지

## Expected Deliverables
1. 에러 분석 리포트 (에러 유형별 통계)
2. 프롬프트 수정안 (HYBRID_DAG_EXTRACTION_PROMPT, build_context_based_entity_extraction_prompt)
3. entity_recognizer.py 로직 수정안 (필요시)
4. 개선 전/후 비교 결과

---

분석을 시작하기 전에, 먼저 평가 데이터 파일을 읽고
어노테이터가 작성한 correct_extracted_entities와 correct_linked_entities를 확인해 주세요.
그 다음, 현재 프롬프트 파일(prompts/entity_extraction_prompt.py)과
entity_recognizer.py를 읽어서 현재 로직을 파악해 주세요.
```

---

## 참고: 현재 프롬프트 구조

### HYBRID_DAG_EXTRACTION_PROMPT 핵심 섹션
1. Language Rule: DO NOT TRANSLATE (원문 그대로 추출)
2. Root Node Selection Hierarchy (5단계 우선순위)
3. DAG Construction Rules (Node/Edge 정의)
4. Strict Exclusions (제외 항목)
5. Output Format: ENTITY + DAG

### build_context_based_entity_extraction_prompt() 동작
1. context_keyword에 따라 다른 프롬프트 생성
2. 핵심 지침: vocabulary 엔티티만 정확히 반환
3. Guidelines: 핵심 혜택/프로모션 관련 엔티티만 포함
4. Return Format: REASON + ENTITY

### 2단계 추출 파이프라인
```
Stage 1: HYBRID_DAG_EXTRACTION_PROMPT
  Input: MMS 메시지
  Output: ENTITY (초기 추출), DAG Context

Stage 2: build_context_based_entity_extraction_prompt()
  Input: 메시지, Stage 1 엔티티, DAG Context, vocabulary 후보
  Output: REASON, ENTITY (vocabulary 필터링됨)
```

---

## Key Configuration Thresholds

From `config/settings.py`:

```python
# Extraction phase (Stage 1) - Keep LOW for high recall
entity_llm_fuzzy_threshold = 0.6          # RapidFuzz threshold
entity_similarity_threshold = 0.2          # Sequence baseline
entity_combined_similarity_threshold = 0.2 # Both s1 AND s2 must exceed this
entity_high_similarity_threshold = 1.0     # Final filter: sim_s1 + sim_s2 >= 1.0

# Linking phase (Stage 2)
fuzzy_threshold = 0.5                      # Initial Kiwi extraction
similarity_threshold = 0.2                 # Broad filtering
high_similarity_threshold = 1.0            # Final gate
```

**Important**: Do not increase these thresholds without thorough evaluation. Lower thresholds = higher recall = better final linked results.

---

## Evaluation Commands

```bash
# Generate evaluation with specific model
python tests/generate_entity_extraction_eval.py \
    --input data/reg_test.txt \
    --output-dir outputs/ \
    --llm-model cld  # or 'ax'

# Re-evaluate existing file with different model
python tests/generate_entity_extraction_eval.py \
    --re-evaluate outputs/entity_extraction_eval_YYYYMMDD_HHMMSS.csv \
    --output-dir outputs/ \
    --llm-model ax
```
