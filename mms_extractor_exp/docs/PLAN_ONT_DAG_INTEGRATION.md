# Plan: ONT 모드 결과로 Step 9 (DAG Extraction) 대체

> **상태: ✅ 구현 완료** (2026-02-03) → **⚠️ ONT 최적화 이후 제거됨** (commit a4e1ef0)
>
> DAG 추출은 모든 context mode에서 fresh LLM call을 사용합니다. ONT 모드에서의 DAG 재사용 최적화는 결과 품질 문제로 제거되었습니다.

## 목표
`entity_extraction_context_mode='ont'` 사용 시, **별도의 LLM 호출 없이** 이미 추출된 온톨로지 결과를 활용하여 Step 9 (DAGExtractionStep)의 기능을 수행한다.

---

## 현재 상황 분석

### Step 9 (DAGExtractionStep) 현재 동작
```
1. extract_dag() 호출
2. build_dag_extraction_prompt() → LLM 호출
3. DAGParser.extract_dag_section() → DAG 텍스트 추출
4. DAGParser.parse_dag() → NetworkX DiGraph 생성
5. create_dag_diagram() → PNG 이미지 생성
6. 결과: entity_dag 리스트
```

**출력 형식:**
```python
entity_dag = [
    "(상품A:구매) -[획득]-> (혜택B:제공)",
    "(이벤트C:참여) -[응모]-> (혜택B:제공)"
]
```

### ONT 모드 현재 결과 (entity_recognizer.py)
```python
{
    "entities": ["아이폰 17", "캐시백"],
    "entity_types": {"아이폰 17": "Product", "캐시백": "Benefit"},
    "relationships": [
        {"source": "아이폰 17", "target": "캐시백", "type": "OFFERS"}
    ],
    "dag_text": "(아이폰 17:구매) -[획득]-> (캐시백:제공)",
    "raw_json": { ... }
}
```

### 비교 분석

| 항목 | Step 9 (현재) | ONT 모드 결과 |
|------|--------------|---------------|
| **DAG 형식** | `(Entity:Action) -[Rel]-> (Entity:Action)` | 동일 형식 (`dag_text`) |
| **LLM 호출** | 별도 호출 필요 | 이미 완료 |
| **추가 정보** | 없음 | entity_types, relationships |
| **NetworkX 그래프** | DAGParser 생성 | 직접 생성 필요 |
| **이미지 생성** | create_dag_diagram() | 동일 함수 활용 가능 |

---

## 구현 계획

### Phase 1: ONT 결과 저장 및 전달

**문제:** 현재 ONT 결과(`dag_text`, `relationships`, `entity_types`)가 `extract_entities_with_llm()` 내부에서만 사용되고, Step 9까지 전달되지 않음.

**해결:**
1. `EntityExtractionStep`에서 ONT 결과를 `WorkflowState`에 저장
2. `DAGExtractionStep`에서 해당 결과 활용

**수정 파일:**
- `services/entity_recognizer.py` - ONT 결과 반환 구조 확장
- `core/mms_workflow_steps.py` - EntityExtractionStep에서 ONT 메타데이터 저장

### Phase 2: NetworkX 그래프 생성 함수

**새 함수:** `build_dag_from_ontology()`

```python
def build_dag_from_ontology(ont_result: dict) -> nx.DiGraph:
    """
    ONT 결과에서 NetworkX DiGraph 생성

    Args:
        ont_result: {
            'dag_text': str,
            'entity_types': dict,
            'relationships': list
        }

    Returns:
        nx.DiGraph: DAG 그래프
    """
    # 방법 1: dag_text 파싱 (기존 DAGParser 활용)
    # 방법 2: relationships에서 직접 생성
```

**수정 파일:**
- `core/entity_dag_extractor.py` - 새 함수 추가

### Phase 3: DAGExtractionStep 분기 처리

**수정:** `entity_extraction_context_mode='ont'`일 때 LLM 재호출 건너뛰기

```python
class DAGExtractionStep(WorkflowStep):
    def execute(self, state: WorkflowState) -> WorkflowState:
        extractor = state.get("extractor")

        # ONT 모드일 경우 이미 추출된 결과 사용
        if extractor.entity_extraction_context_mode == 'ont':
            ont_result = state.get("ont_extraction_result")
            if ont_result:
                dag = self._build_dag_from_ont(ont_result)
                dag_list = self._format_dag_list(ont_result['dag_text'])
                # 이미지 생성...
                return state

        # 기존 로직 (LLM 호출)
        dag_result = extract_dag(...)
```

**수정 파일:**
- `core/mms_workflow_steps.py` - DAGExtractionStep 수정

### Phase 4: 출력 형식 일치

**확인 사항:**
- `entity_dag` 리스트 형식 동일
- DAG 이미지 파일 생성

**dag_text → entity_dag 변환:**
```python
# ONT dag_text 예시:
"(아이폰 17:구매) -[획득]-> (캐시백:제공)"

# entity_dag 리스트 (Step 9 출력과 동일):
["(아이폰 17:구매) -[획득]-> (캐시백:제공)"]
```

---

## 상세 구현 단계

### Step 1: ONT 결과 저장 구조 확장

**`services/entity_recognizer.py`:**
```python
# extract_entities_with_llm() 반환값에 ont_metadata 추가
return {
    'similarities_df': cand_entities_sim,
    'ont_metadata': {  # ONT 모드일 때만 포함
        'dag_text': combined_context,  # 이미 DAG + Entity Types 포함
        'entity_types': all_entity_types,
        'relationships': all_relationships,
        'raw_json': raw_json
    }
}
```

### Step 2: EntityExtractionStep에서 ONT 메타데이터 저장

**`core/mms_workflow_steps.py` - EntityExtractionStep:**
```python
# LLM 모드일 때
similarities = entity_recognizer.extract_entities_with_llm(...)

# ONT 메타데이터 저장 (있으면)
if 'ont_metadata' in similarities:
    state.set("ont_extraction_result", similarities['ont_metadata'])
```

### Step 3: NetworkX 그래프 생성

**`core/entity_dag_extractor.py` - 새 함수:**
```python
def build_dag_from_ontology(ont_result: dict) -> nx.DiGraph:
    """
    ONT 결과에서 NetworkX DiGraph 생성

    두 가지 방법 지원:
    1. dag_text 파싱 (기존 DAGParser 활용)
    2. relationships에서 직접 생성 (더 정확한 타입 정보 보존)
    """
    G = nx.DiGraph()

    # relationships에서 그래프 생성
    entity_types = ont_result.get('entity_types', {})
    relationships = ont_result.get('relationships', [])

    for rel in relationships:
        src = rel.get('source', '')
        tgt = rel.get('target', '')
        rel_type = rel.get('type', '')

        if src and tgt:
            # 노드 추가 (타입 정보 포함)
            G.add_node(src, entity_type=entity_types.get(src, 'Unknown'))
            G.add_node(tgt, entity_type=entity_types.get(tgt, 'Unknown'))

            # 엣지 추가
            G.add_edge(src, tgt, relation=rel_type)

    return G
```

### Step 4: DAGExtractionStep 수정

**`core/mms_workflow_steps.py` - DAGExtractionStep:**
```python
def execute(self, state: WorkflowState) -> WorkflowState:
    extractor = state.get("extractor")

    if not extractor.extract_entity_dag:
        # 비활성화 처리...
        return state

    msg = state.get("msg")
    message_id = state.get("message_id", "#")

    # ONT 모드 확인
    if extractor.entity_extraction_context_mode == 'ont':
        ont_result = state.get("ont_extraction_result")
        if ont_result and ont_result.get('dag_text'):
            return self._execute_from_ont(state, ont_result, msg, message_id)

    # 기존 로직 (LLM 호출)
    return self._execute_with_llm(state, msg, message_id)

def _execute_from_ont(self, state, ont_result, msg, message_id):
    """ONT 결과에서 DAG 생성 (LLM 호출 없음)"""
    from .entity_dag_extractor import build_dag_from_ontology

    # 1. DAG 텍스트를 리스트로 변환
    dag_text = ont_result.get('dag_text', '')
    # "DAG: ..." 부분만 추출
    dag_lines = []
    for line in dag_text.split('\n'):
        if line.startswith('DAG:'):
            dag_lines.append(line.replace('DAG:', '').strip())
        elif '->->' in line or '-[' in line:
            dag_lines.append(line.strip())

    dag_list = sorted([d for d in dag_lines if d])

    # 2. NetworkX 그래프 생성
    dag = build_dag_from_ontology(ont_result)

    # 3. 결과 저장
    final_result = state.get("final_result", {})
    final_result['entity_dag'] = dag_list
    state.set("final_result", final_result)

    raw_result = state.get("raw_result", {})
    raw_result['entity_dag'] = dag_list
    state.set("raw_result", raw_result)

    # 4. 이미지 생성
    if dag.number_of_nodes() > 0:
        from utils import create_dag_diagram, sha256_hash
        dag_filename = f'dag_{message_id}_{sha256_hash(msg)}'
        create_dag_diagram(dag, filename=dag_filename)
        logger.info(f"📊 DAG 다이어그램 저장 (ONT): {dag_filename}.png")

    return state
```

---

## 파일 변경 요약

| 파일 | 변경 내용 |
|------|----------|
| `services/entity_recognizer.py` | `extract_entities_with_llm()` 반환값에 `ont_metadata` 추가 |
| `services/result_builder.py` | ONT 메타데이터 전달 (필요시) |
| `core/mms_workflow_steps.py` | EntityExtractionStep: ONT 메타데이터 저장<br>DAGExtractionStep: ONT 분기 처리 |
| `core/entity_dag_extractor.py` | `build_dag_from_ontology()` 함수 추가 |

---

## 결과 일치 검증

### 검증 항목

1. **entity_dag 형식**
   - 기존: `["(Entity:Action) -[Rel]-> (Entity:Action)", ...]`
   - ONT: 동일 형식

2. **DAG 이미지**
   - 기존: `dag_{message_id}_{hash}.png`
   - ONT: 동일 파일명 규칙

3. **NetworkX 그래프 구조**
   - 노드: entity 정보 포함
   - 엣지: relation 정보 포함
   - ONT 추가: entity_type 속성

---

## 장점

1. **LLM 호출 절감**: ONT 모드 사용 시 Step 9에서 별도 LLM 호출 불필요
2. **일관성**: 동일 메시지에 대해 엔티티 추출과 DAG가 동일한 LLM 응답 기반
3. **풍부한 메타데이터**: entity_types, relationships 정보 활용 가능
4. **호환성**: 기존 출력 형식 완전 호환

---

## 구현 순서

1. `entity_dag_extractor.py`에 `build_dag_from_ontology()` 추가
2. `entity_recognizer.py`에서 ONT 메타데이터 반환 구조 확장
3. `mms_workflow_steps.py`의 EntityExtractionStep 수정
4. `mms_workflow_steps.py`의 DAGExtractionStep에 ONT 분기 추가
5. 테스트 작성 및 검증
