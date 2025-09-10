# 성능 검증 시스템

이 디렉토리는 5개 모델 (gemma, gemini, claude, ax, gpt)의 MMS 추출 성능을 비교하는 시스템입니다.

## 📋 실험 개요

### 목적
- 5개 LLM 모델의 MMS 메시지 추출 성능 비교
- 각 모델이 생성하는 "7단계: 엔티티 매칭 및 최종 결과 구성" 전의 json_objects 분석
- 정답 데이터셋 기반 객관적 성능 평가

### 비교 모델
1. **gemma**: `skt/gemma3-12b-it`
2. **gemini**: `gcp/gemini-2.5-flash` 
3. **claude**: `amazon/anthropic/claude-sonnet-4-20250514`
4. **ax**: `skt/ax4`
5. **gpt**: `azure/openai/gpt-4o-2024-08-06`

### 실험 설정
- `extract-entity-dag=false`
- `product-info-extraction-mode='rag'`
- `entity-matching-mode='llm'`

## 🚀 실행 방법

### 1. 전체 실험 자동 실행 (권장)

**⚠️ 중요: mms_extractor_unified 디렉토리에서 실행하세요**

```bash
# mms_extractor_unified 디렉토리로 이동
cd /path/to/mms_extractor_unified

# 기본 설정으로 실행 (배치 크기 100, 최소 메시지 길이 300자)
python performance_validation/run_validation.py

# 사용자 설정으로 실행
python performance_validation/run_validation.py --batch-size 50 --output-dir my_results --similarity-threshold 0.85 --min-message-length 400
```

### 2. 개별 단계 실행

#### 단계 1: 모델 추출 실험
```bash
# mms_extractor_unified 디렉토리에서
python performance_validation/model_comparison_experiment.py --batch-size 100 --output-dir results --min-message-length 300
```

#### 단계 2: 성능 평가
```bash
# mms_extractor_unified 디렉토리에서
python performance_validation/model_performance_evaluator.py --results-dir results --similarity-threshold 0.9 --min-message-length 300
```

### 3. 기존 결과 활용

이미 추출 결과가 있는 경우 평가만 실행:
```bash
# mms_extractor_unified 디렉토리에서
python performance_validation/run_validation.py --skip-extraction
```

## 📁 결과 구조

실험 실행 후 다음과 같은 디렉토리 구조가 생성됩니다:

```
results/
├── combined_extraction_results.json    # 전체 모델 결과 (JSON)
├── combined_extraction_results.pkl     # 전체 모델 결과 (피클)
├── experiment_metadata.json            # 실험 메타데이터
├── gemma_extraction_results.json       # gemma 모델 개별 결과
├── gemini_extraction_results.json      # gemini 모델 개별 결과
├── claude_extraction_results.json      # claude 모델 개별 결과
├── ax_extraction_results.json          # ax 모델 개별 결과
├── gpt_extraction_results.json         # gpt 모델 개별 결과
└── evaluation/
    ├── ground_truth_dataset.json       # 정답 데이터셋
    ├── model_evaluation_results.json   # 상세 평가 결과
    ├── performance_summary.json        # 성능 요약
    └── performance_report.txt          # 텍스트 리포트
```

## 📊 평가 방법

### 정답 데이터셋 생성
1. gemini, gpt, claude 3개 모델이 모두 성공한 메시지 선별
2. **추가 필터링 조건 적용**:
   - 메시지 길이가 최소 300자 이상
   - 1st depth 태그들(title, purpose, product, channel, pgm)의 값이 모두 유효하게 채워져 있어야 함
3. 3개 모델 결과 간 유사도 계산 (기본 임계값: 90%)
4. 임계값 이상의 메시지에서 claude 결과를 정답으로 설정

### 성능 평가 지표
- **전체 유사도** (Overall Similarity): 가중평균 점수
  - title: 20%
  - purpose: 15% 
  - product: 35%
  - channel: 15%
  - pgm: 15%

### 유사도 계산 세부사항
- **텍스트 유사도**: SequenceMatcher 사용
- **리스트 유사도**: Jaccard 유사도
- **제품 유사도**: 다중 필드 가중평균
- **채널/프로그램 유사도**: 필드별 평균

## 📈 결과 해석

### 성능 등급
- **A+ (Excellent)**: 0.9 이상
- **A (Very Good)**: 0.8 이상
- **B+ (Good)**: 0.7 이상  
- **B (Fair)**: 0.6 이상
- **C+ (Below Average)**: 0.5 이상
- **C (Poor)**: 0.5 미만

### 주요 출력 파일

#### 1. performance_report.txt
텍스트 형태의 종합 리포트
- 모델별 성능 요약
- 필드별 상세 분석
- 성능 순위

#### 2. performance_summary.json  
프로그래밍적 접근을 위한 요약 데이터
- 모델별 통계
- 성능 순위
- 메타데이터

#### 3. model_evaluation_results.json
상세한 평가 결과
- 메시지별 유사도 점수
- 필드별 분석 결과
- 통계 정보

## ⚙️ 설정 옵션

### 명령행 옵션

#### run_validation.py
- `--batch-size`: 실험할 메시지 수 (기본값: 100)
- `--output-dir`: 결과 저장 디렉토리 (기본값: results)
- `--similarity-threshold`: 정답 생성 유사도 임계값 (기본값: 0.9)
- `--min-message-length`: 정답용 메시지 최소 길이 (기본값: 300)
- `--skip-extraction`: 추출 단계 건너뛰기
- `--skip-evaluation`: 평가 단계 건너뛰기

#### model_comparison_experiment.py
- `--batch-size`: 배치 크기
- `--output-dir`: 결과 저장 디렉토리
- `--min-message-length`: 최소 메시지 길이 (추출 단계에서 사전 필터링)

#### model_performance_evaluator.py  
- `--results-dir`: 결과 디렉토리
- `--similarity-threshold`: 유사도 임계값
- `--min-message-length`: 정답용 메시지 최소 길이

## 🔧 문제 해결

### 일반적인 문제

1. **API 키 오류**
   - `.env` 파일에 필요한 API 키들이 설정되어 있는지 확인
   - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `CUSTOM_API_KEY` 등

2. **메모리 부족**
   - 배치 크기를 줄여서 실행: `--batch-size 50`

3. **모델 초기화 실패**
   - 네트워크 연결 확인
   - API 키 권한 확인

4. **결과 파일 없음**
   - 추출 단계가 성공했는지 확인
   - `combined_extraction_results.json` 또는 `.pkl` 파일 존재 확인

### 로그 확인
각 실행 시 로그 파일이 생성됩니다:
- `validation_run_YYYYMMDD_HHMMSS.log`
- `model_comparison_YYYYMMDD_HHMMSS.log`  
- `model_evaluation_YYYYMMDD_HHMMSS.log`

## 📝 예제 실행

### 소규모 테스트
```bash
# 10개 메시지로 빠른 테스트
python run_validation.py --batch-size 10 --output-dir test_results
```

### 대규모 실험
```bash
# 500개 메시지로 본격적인 실험
python run_validation.py --batch-size 500 --output-dir large_experiment
```

### 기존 결과 재평가
```bash
# 다른 유사도 임계값과 메시지 길이 조건으로 재평가
python model_performance_evaluator.py --results-dir results --similarity-threshold 0.85 --min-message-length 500
```

## 🏗️ 시스템 구조

### 핵심 특징
- **최소 원본 수정**: 기존 MMSExtractor 코드를 최대한 그대로 사용
- **독립적 실행**: mms_extractor_unified 디렉토리 내에서 완전히 실행
- **모듈화 설계**: 각 단계별로 독립 실행 가능
- **확장 가능성**: 새로운 모델 추가 용이

### 주요 컴포넌트
1. **model_comparison_experiment.py**: 모델별 추출 실험
2. **model_performance_evaluator.py**: 성능 평가 및 분석
3. **run_validation.py**: 통합 실행 스크립트
4. **README.md**: 사용 가이드 (이 문서)

## 📚 참고 정보

### 원본 코드 의존성
- `mms_extractor.py`: 메인 추출기 (기존 메소드 활용)
- `config/settings.py`: 설정 정보
- 데이터 파일들: MMS 메시지, 상품 정보 등

### 실험 재현성
- 메시지 샘플링에 `random_state=42` 사용
- 모든 설정과 메타데이터가 저장됨
- 동일한 설정으로 재실행 가능

### 확장 가능성
- 새로운 모델 추가: `models` 딕셔너리 수정
- 평가 지표 추가: `calculate_dictionary_similarity` 함수 수정
- 새로운 유사도 함수: 평가기 클래스 확장

## 📅 결과 디렉토리 자동 타임스탬프

모든 실험 결과는 실행 시간에 따라 자동으로 구분됩니다:

```bash
# 입력: --output-dir results
# 실제 생성: results_202501031430

# 입력: --output-dir my_experiment  
# 실제 생성: my_experiment_202501031430
```

**타임스탬프 형식**: `YYYYMMDDHHmm` (년월일시분)

이를 통해 여러 실험을 동시에 실행하거나 기록을 보관할 때 충돌 없이 관리할 수 있습니다.
