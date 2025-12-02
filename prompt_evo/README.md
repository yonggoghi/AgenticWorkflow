# Prompt Evolution System

자동으로 Student LLM의 System Prompt를 Teacher LLM 출력과 유사하게 진화시키는 시스템입니다.

## 설치

```bash
# 가상환경 활성화
source /Users/yongwook/workspace/AgenticWorkflow/venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 환경 설정

`.env` 파일을 생성하고 API 키를 설정하세요:

```bash
cp .env.example .env
# .env 파일을 편집하여 OPENAI_API_KEY 설정
```

## 사용법

### 기본 실행

```bash
python prompt_evolution.py
```

### 커스텀 설정

```bash
python prompt_evolution.py \
    --prompt_file my_prompt.txt \
    --student_model gpt-4o-mini \
    --teacher_model gpt-4o \
    --batch_size 3 \
    --anchor_count 3 \
    --verbose
```

### 소규모 데이터셋 최적화 (권장)

```bash
python prompt_evolution.py \
    --batch_size 3 \
    --anchor_count 3 \
    --anchor_threshold 0.90 \
    --train_ratio 0.7 \
    --verbose
```

## CLI 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--prompt_file` | `prompt.txt` | 초기 프롬프트 파일 |
| `--student_model` | `skt/ax4` | Student 모델명 |
| `--teacher_model` | `gcp/gemini-2.5-flash` | Teacher 모델명 |
| `--evaluator_model` | `amazon/anthropic/claude-sonnet-4-20250514` | Evaluator 모델명 |
| `--data_file` | `reg_test.txt` | 데이터 파일 |
| `--train_ratio` | `0.7` | 학습 데이터 비율 |
| `--output_dir` | `./outputs` | 출력 디렉토리 |
| `--batch_size` | `3` | 배치 크기 |
| `--anchor_count` | `3` | 앵커 샘플 수 |
| `--anchor_threshold` | `0.90` | 앵커 점수 임계값 |
| `--max_iterations` | `None` | 최대 반복 횟수 |
| `--checkpoint_every` | `2` | 체크포인트 저장 간격 |
| `--seed` | `42` | 랜덤 시드 |
| `--verbose` | `False` | 상세 로그 |

## 출력 구조

```
outputs/
├── final_prompt_YYYYMMDD_HHMMSS.txt    # 최종 프롬프트
├── evolution_log.jsonl                  # 진화 로그
├── validation_results.json              # 검증 결과
├── anchor_samples.json                  # 앵커 샘플
└── checkpoints/
    ├── batch_0.json
    └── ...
```

## 주요 기능

### 과적합 방지 메커니즘

1. **배치 기반 업데이트**: 단일 메시지가 아닌 N개 메시지를 묶어서 평가
2. **앵커 샘플 회귀 테스트**: 대표 샘플에 대한 성능 유지 검증
3. **롤백 메커니즘**: 성능 저하 시 이전 프롬프트로 복원
4. **보수적 수정**: 규칙 추가만 허용, 삭제/변경 금지

### 재현 가능성

- `--seed` 인자로 데이터 분할 및 셔플 제어
- 모든 설정이 `validation_results.json`에 저장됨

### 복구 기능

- 주기적 체크포인트 저장
- Ctrl+C 중단 시 현재 최고 프롬프트 자동 저장 (`interrupted_prompt.txt`)

## 프로젝트 구조

```
prompt_evo/
├── prompt_evolution.py          # 메인 스크립트
├── evaluator_prompts/           # Evaluator 프롬프트
│   ├── evolution_prompt.txt
│   └── similarity_prompt.txt
├── prompt.txt                   # 초기 프롬프트
├── reg_test.txt                 # MMS 메시지 데이터
├── .env                         # API 키
├── requirements.txt             # 의존성
└── outputs/                     # 실행 결과
```
