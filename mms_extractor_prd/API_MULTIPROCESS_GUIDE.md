# MMS 추출기 API - 멀티프로세스 처리 가이드

## 개요

MMS 추출기 API가 멀티프로세스 처리를 지원하도록 개선되었습니다. 이제 단일 메시지 처리에서는 메인 처리와 DAG 추출을 병렬로 수행하고, 배치 처리에서는 여러 메시지를 동시에 처리할 수 있습니다.

## 주요 개선사항

### 1. 병렬 처리 지원
- **단일 메시지**: 메인 처리와 DAG 추출을 멀티스레드로 병렬 실행
- **배치 처리**: 여러 메시지를 멀티프로세스로 동시 처리

### 2. DAG 추출 기능
- `extract_entity_dag=true` 옵션으로 엔티티 간 관계를 DAG 형태로 추출
- 시각적 다이어그램 생성 (./dag_images/ 디렉토리에 저장)
- 병렬 처리로 성능 최적화

## API 엔드포인트

### 1. 단일 메시지 처리 - `/extract` (POST)

#### 요청 예시 (DAG 추출 포함)
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "message": "[SKT] ZEM폰 포켓몬에디션3 안내 - 우리 아이 첫 번째 스마트폰",
    "llm_model": "ax",
    "extract_entity_dag": true,
    "product_info_extraction_mode": "llm",
    "entity_matching_mode": "llm"
  }'
```

#### 응답 예시
```json
{
  "success": true,
  "result": {
    "title": "ZEM폰 포켓몬에디션3 안내",
    "purpose": ["상품 가입 유도"],
    "product": [
      {
        "item_name_in_msg": "ZEM폰",
        "expected_action": "가입",
        "item_in_voca": [{"item_name_in_voca": "ZEM폰", "item_id": ["ZEM001"]}]
      }
    ],
    "channel": [],
    "pgm": [],
    "entity_dag": [
      "(고객:가입) -[하면]-> (혜택:수령)",
      "(ZEM폰:구매) -[통해]-> (안전:보장)"
    ]
  },
  "metadata": {
    "llm_model": "ax",
    "extract_entity_dag": true,
    "processing_time_seconds": 3.456,
    "timestamp": 1703123456.789
  }
}
```

### 2. 배치 처리 - `/extract/batch` (POST)

#### 요청 예시 (멀티프로세스)
```bash
curl -X POST http://localhost:8000/extract/batch \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      "[SKT] ZEM폰 포켓몬에디션3 안내",
      "[KT] 5G 요금제 혜택 안내",
      "[LGU+] 무제한 데이터 프로모션"
    ],
    "extract_entity_dag": true,
    "max_workers": 4,
    "llm_model": "ax"
  }'
```

#### 응답 예시
```json
{
  "success": true,
  "results": [
    {
      "index": 0,
      "success": true,
      "result": {
        "title": "ZEM폰 포켓몬에디션3 안내",
        "entity_dag": ["(고객:가입) -[하면]-> (혜택:수령)"]
      }
    },
    {
      "index": 1,
      "success": true,
      "result": {
        "title": "5G 요금제 혜택 안내",
        "entity_dag": ["(요금제:가입) -[하면]-> (할인:적용)"]
      }
    }
  ],
  "summary": {
    "total_messages": 3,
    "successful": 2,
    "failed": 1
  },
  "metadata": {
    "extract_entity_dag": true,
    "max_workers": 4,
    "processing_time_seconds": 2.134
  }
}
```

## 서버 실행

### 기본 실행
```bash
python api.py --host 0.0.0.0 --port 8000
```

### 데이터베이스 모드로 실행
```bash
python api.py --offer-data-source db --port 8000
```

### 테스트 모드 (DAG 추출 포함)
```bash
python api.py --test --extract-entity-dag --message "테스트 메시지"
```

## 새로운 파라미터

### 단일 메시지 처리
- `extract_entity_dag` (boolean): DAG 추출 여부 (기본값: false)

### 배치 처리
- `extract_entity_dag` (boolean): DAG 추출 여부 (기본값: false)
- `max_workers` (integer): 병렬 처리 워커 수 (기본값: CPU 코어 수)

## 성능 개선 효과

### 단일 메시지 처리
- **기존**: 순차 처리 (메인 처리 → DAG 추출)
- **개선**: 병렬 처리 (메인 처리 || DAG 추출)
- **성능 향상**: 약 30-50% 처리 시간 단축

### 배치 처리
- **기존**: 순차 처리 (메시지1 → 메시지2 → 메시지3)
- **개선**: 병렬 처리 (메시지1 || 메시지2 || 메시지3)
- **성능 향상**: 메시지 수에 비례한 처리 시간 단축

## DAG 추출 기능

### 기능 설명
- 메시지에서 엔티티 간의 관계를 방향성 있는 그래프로 추출
- NetworkX를 사용한 그래프 구조 생성
- Graphviz를 통한 시각적 다이어그램 생성

### 출력 형태
```
(고객:가입) -[하면]-> (혜택:수령) -[통해]-> (만족도:향상)
(상품:구매) -[시]-> (할인:적용)
```

### 시각적 다이어그램
- 파일 위치: `./dag_images/dag_[해시값].png`
- 자동 생성되며 메시지별로 고유한 파일명 사용

## 모니터링 및 로깅

### 로그 파일
- API 서버 로그: `./logs/api_server.log`
- MMS 추출기 로그: `./logs/mms_extractor.log`

### 주요 로그 메시지
```
🎯 DAG 추출 요청됨 - LLM: ax, 메시지 길이: 150자
✅ DAG 추출 성공 - 길이: 45자
🎯 배치 DAG 추출 요청됨 - 3개 메시지, 워커: 4
```

## 에러 처리

### 일반적인 오류
- 400: 잘못된 요청 파라미터
- 500: 서버 내부 오류 (LLM 호출 실패 등)

### 배치 처리 오류
- 개별 메시지 실패는 전체 배치를 중단시키지 않음
- 각 메시지별로 성공/실패 상태 반환
- 전체 통계 정보 제공

## 사용 권장사항

### 단일 메시지
- DAG 추출이 필요한 경우에만 `extract_entity_dag=true` 사용
- 처리 시간이 중요한 경우 DAG 추출 비활성화

### 배치 처리
- `max_workers`는 서버 리소스에 맞게 조정 (CPU 코어 수의 50-80%)
- 대량 처리 시 배치 크기를 50-100개로 제한
- DAG 추출 시 처리 시간이 증가함을 고려

### 성능 최적화
- 로컬 데이터 소스가 DB보다 빠름
- LLM 모델별 처리 속도 차이 고려
- 네트워크 지연을 고려한 타임아웃 설정
