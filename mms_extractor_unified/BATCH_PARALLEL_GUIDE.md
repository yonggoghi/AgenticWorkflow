# 배치 처리 병렬 처리 가이드

## 개요

`batch.py`가 병렬 처리를 지원하도록 개선되어 대량의 MMS 메시지를 효율적으로 처리할 수 있습니다.

## 주요 기능

### 🚀 병렬 처리 지원
- **멀티프로세스**: 여러 메시지를 동시에 처리
- **멀티스레드**: DAG 추출 시 메인 처리와 DAG 생성을 병렬 실행
- **자동 fallback**: 병렬 처리 실패 시 순차 처리로 자동 전환

### 📊 성능 모니터링
- 처리 시간 측정 (총 시간, 메시지 처리 시간)
- 메시지당 평균 처리 시간
- 처리량 (메시지/초)
- 성공/실패 통계

### ⚙️ 설정 옵션
- 워커 수 조정 (`--max-workers`)
- 병렬 처리 비활성화 (`--disable-multiprocessing`)
- DAG 추출 활성화 (`--extract-entity-dag`)

## 사용법

### 기본 사용법

```bash
# 기본 병렬 처리 (CPU 코어 수만큼 워커 사용)
python batch.py --batch-size 20

# 워커 수 지정
python batch.py --batch-size 50 --max-workers 8

# DAG 추출과 함께 병렬 처리
python batch.py --batch-size 30 --extract-entity-dag --max-workers 4
```

### 성능 최적화

```bash
# 고성능 서버에서 대량 처리
python batch.py --batch-size 100 --max-workers 16 --extract-entity-dag

# 리소스 제한 환경에서 안정적 처리
python batch.py --batch-size 20 --max-workers 2 --disable-multiprocessing
```

### 완전한 설정 예시

```bash
python batch.py \
  --batch-size 50 \
  --max-workers 8 \
  --extract-entity-dag \
  --llm-model ax \
  --entity-extraction-mode llm \
  --product-info-extraction-mode llm \
  --output-file ./data/parallel_batch_results.csv
```

## 성능 비교

### 예상 성능 향상

| 처리 모드 | 메시지 수 | 예상 시간 | 워커 수 |
|-----------|-----------|-----------|---------|
| 순차 처리 | 10개 | 60초 | 1개 |
| 병렬 처리 | 10개 | 20초 | 4개 |
| 병렬 처리 | 10개 | 15초 | 8개 |

### DAG 추출 포함 시

| 처리 모드 | DAG 추출 | 예상 개선 |
|-----------|----------|-----------|
| 순차 처리 | ON | 기준 시간 |
| 병렬 처리 (단일) | ON | 15-30% 단축 |
| 병렬 처리 (배치) | ON | 50-70% 단축 |

## 로그 분석

### 성능 로그 예시

```
🚀 병렬 처리 모드로 20개 메시지 처리 시작 (워커: 8개)
🎯 병렬 처리 완료: 20개 메시지, 45.23초 소요

🎯 배치 처리 성능 요약
==================================================
처리 모드: 병렬 처리
워커 수: 8
DAG 추출: ON
총 처리 시간: 47.85초
메시지 처리 시간: 45.23초
메시지당 평균 시간: 2.26초
처리량: 0.44 메시지/초
==================================================
```

### 주요 메트릭 해석

- **처리 모드**: 병렬/순차 처리 여부
- **워커 수**: 동시 실행 프로세스 수
- **DAG 추출**: 엔티티 관계 그래프 생성 여부
- **처리량**: 초당 처리 가능한 메시지 수

## 권장 설정

### 환경별 권장 워커 수

| CPU 코어 | 메모리 | 권장 워커 수 | 배치 크기 |
|----------|--------|--------------|-----------|
| 4코어 | 8GB | 2-4개 | 10-20개 |
| 8코어 | 16GB | 4-8개 | 20-50개 |
| 16코어 | 32GB | 8-16개 | 50-100개 |

### 사용 시나리오별 권장 설정

**개발/테스트 환경:**
```bash
python batch.py --batch-size 5 --max-workers 2
```

**운영 환경 (일반):**
```bash
python batch.py --batch-size 30 --max-workers 6 --extract-entity-dag
```

**고성능 처리:**
```bash
python batch.py --batch-size 100 --max-workers 12 --extract-entity-dag
```

## 문제 해결

### 메모리 부족 시
- 워커 수를 줄이세요: `--max-workers 2`
- 배치 크기를 줄이세요: `--batch-size 10`

### 처리 속도가 느린 경우
- 병렬 처리가 활성화되어 있는지 확인
- 워커 수를 CPU 코어 수에 맞게 조정
- DAG 추출을 비활성화해보세요

### 오류 발생 시
- 순차 처리로 전환: `--disable-multiprocessing`
- 로그에서 상세 오류 내용 확인
- 배치 크기를 줄여서 재시도

## 출력 파일

처리 결과는 CSV 파일로 저장되며 다음 정보를 포함합니다:

- `msg_id`: 메시지 ID
- `msg`: 원본 메시지
- `extraction_result`: JSON 형태의 추출 결과
- `processed_at`: 처리 시각
- `title`: 추출된 제목
- `purpose`: 추출된 목적
- `product_count`: 상품 개수
- `channel_count`: 채널 개수
- `pgm`: 프로그램 정보

병렬 처리를 통해 대량의 MMS 메시지를 효율적으로 처리하고 분석할 수 있습니다!
