# Quick Extractor 사용 가이드

메시지에서 제목과 수신 거부 전화번호를 빠르게 추출하는 도구입니다.

## 특징

✅ **다양한 입력 형식 지원**:
- CSV 파일 (컬럼 구조 필요)
- 텍스트 파일 (각 줄이 하나의 메시지, `.txt`)
- 단일 메시지 직접 입력 (`--message`)

✅ **유연한 제목 추출 방법**:
- NLP 기반 (TextRank, TF-IDF) - 빠르고 의존성 없음
- LLM 기반 - 고품질 제목 생성

## 설치

```bash
# 기본 기능 (NLP 기반)
pip install pandas numpy

# LLM 기능 추가 (선택사항)
pip install langchain langchain-openai python-dotenv
```

## 사용 방법 (mms_extractor.py와 동일한 방식)

### 1. 단일 메시지 처리 (`--message`)

```bash
# TextRank 방법으로 단일 메시지 처리
python quick_extractor.py --message "광고 메시지 내용"

# LLM(GPT)으로 단일 메시지 처리
python quick_extractor.py --message "광고 메시지" --method llm --llm-model gpt

# LLM(Claude)으로 단일 메시지 처리
python quick_extractor.py --message "광고 메시지" --method llm --llm-model claude

# TF-IDF 방법으로 단일 메시지 처리
python quick_extractor.py --message "광고 메시지" --method tfidf
```

**출력 예시** (단일 메시지):
```json
{
  "title": "신시가지 지행농협 건너편, 에스알대리점 지행역점에서 9월 혜택 안내",
  "unsubscribe_phone": "1504",
  "original_message": "광고 메시지 내용..."
}
```

### 2. 파일 전체 처리 (`--batch-file`, mms_extractor.py와 동일)

#### CSV 파일 처리

```bash
# 기본 실행 (기본 CSV 파일, TextRank 방법)
python quick_extractor.py

# 커스텀 CSV 파일 처리
python quick_extractor.py --batch-file ./data/messages.csv

# LLM 기반 추출 (고품질)
python quick_extractor.py --batch-file ./data/messages.csv --method llm --llm-model gpt

# 출력 파일 지정
python quick_extractor.py --batch-file ./data/messages.csv --output ./custom_output.json
```

#### 텍스트 파일 처리 (NEW! 🎉)

```bash
# 텍스트 파일 처리 (각 줄이 하나의 메시지)
python quick_extractor.py --batch-file ./data/messages.txt

# LLM 기반 고품질 제목 추출 (권장)
python quick_extractor.py --batch-file ./data/messages.txt --method llm --llm-model ax

# TextRank 방법 (빠름)
python quick_extractor.py --batch-file ./data/messages.txt --method textrank
```

**출력 예시**:
- 파일: `quick_extracted_info.json`
- 형식: 각 메시지별 결과 배열
- CSV와 텍스트 파일 출력 형식이 약간 다름 (컬럼 구조 차이)

## 추출 방법 비교

| 방법 | 속도 | 품질 | 의존성 | 설명 |
|------|------|------|--------|------|
| **llm** | ⚡ | ⭐⭐⭐⭐⭐ | LangChain, OpenAI API | LLM 기반 지능형 제목 생성 (가장 고품질) |
| **textrank** | ⚡⚡⚡ | ⭐⭐⭐ | 없음 | 문장 중요도 기반 (기본값, 빠름) |
| **tfidf** | ⚡⚡⚡ | ⭐⭐ | 없음 | 단어 빈도 기반 |
| **first_bracket** | ⚡⚡⚡⚡ | ⭐ | 없음 | 단순 패턴 매칭 |

## 출력 형식

출력 파일: `quick_extracted_info.json`

```json
{
  "index": 0,
  "offer_date": "20250918",
  "title": "신시가지 지행농협 건너편, 에스알대리점 지행역점에서 9월 혜택을 안내드립니다.",
  "unsubscribe_phone": "1504",
  "original_message_name": "**이미지 검토 의견 확인 후 수정 부탁드립니다.**"
}
```

## LLM 사용을 위한 환경 설정

### 설정 방식

**`quick_extractor.py`는 `mms_extractor.py`와 완전히 동일한 설정 시스템을 사용합니다.**

1. **`config/settings.py` 사용 (권장)**: `mms_extractor.py`와 동일한 설정 공유
2. **환경변수 직접 사용 (fallback)**: `config/settings.py`가 없는 경우

### 설정 예시

LLM 기능을 사용하려면 `.env` 파일에 API 키를 설정해야 합니다:

```bash
# .env 파일 예시
# mms_extractor.py와 동일한 설정 사용
CUSTOM_API_KEY=your-api-key-here
CUSTOM_BASE_URL=https://api.platform.a15t.com/v1

# OpenAI 직접 사용 시
OPENAI_API_KEY=your-openai-key

# 모델별 설정 (선택사항, config/settings.py에서 자동 관리)
GPT_MODEL=azure/openai/gpt-4o-2024-08-06
CLAUDE_MODEL=amazon/anthropic/claude-sonnet-4-20250514
GEMINI_MODEL=gcp/gemini-2.5-flash
AX_MODEL=skt/ax4
```

**참고**: `config/settings.py`가 있으면 자동으로 해당 설정을 사용하므로, `mms_extractor.py`와 완전히 동일한 LLM 환경에서 작동합니다.

## 명령줄 옵션 (mms_extractor.py와 완전히 동일)

```
usage: quick_extractor.py [-h] [--message MESSAGE] [--batch-file BATCH_FILE] 
                          [--output OUTPUT]
                          [--method {textrank,tfidf,first_bracket,llm}]
                          [--llm-model {gpt,claude,gemini,ax,gem,gen,cld}]

옵션:
  -h, --help            도움말 표시
  
  입력 옵션 (mms_extractor.py와 동일):
  --message MESSAGE     단일 메시지 텍스트 입력 (mms_extractor.py와 동일)
  --batch-file FILE     배치 파일 경로 (CSV 또는 텍스트, 기본값: ./data/mms_data_251023.csv)
  
  추출 옵션:
  --method METHOD       제목 추출 방법 (기본값: textrank)
                        - textrank: NLP 기반 문장 중요도
                        - tfidf: 단어 빈도 기반
                        - first_bracket: 패턴 매칭
                        - llm: LLM 기반 지능형 추출
  
  출력 옵션:
  --output OUTPUT       출력 JSON 파일 경로 (배치 파일 모드만, 기본값: ./quick_extracted_info.json)
  
  LLM 옵션:
  --llm-model MODEL     LLM 모델 선택 (기본값: gpt)
                        - gpt: GPT-4
                        - cld/claude: Claude
                        - gen/gemini: Gemini
                        - ax: AX4
```

## 성능

### NLP 기반 (textrank, tfidf)
- **처리 속도**: 830개 메시지 약 3-5초 ⚡
- **추출 정확도**: 수신거부 번호 90.8% (754/830)
- **의존성**: 최소 (pandas, numpy만)

### LLM 기반
- **처리 속도**: 830개 메시지 약 5-10분 (API 호출)
- **제목 품질**: 매우 높음 (문맥 이해 기반)
- **의존성**: LangChain, OpenAI API 키 필요
- **비용**: API 사용량에 따라 과금

## API로 사용하기 (api.py 통합용)

Quick Extractor는 API로 사용할 수 있도록 JSON 반환 메서드를 제공합니다:

### 단일 메시지 처리 API

```python
from quick_extractor import MessageInfoExtractor

# 추출기 초기화
extractor = MessageInfoExtractor(csv_path=None, use_llm=True, llm_model='ax')

# 메시지 처리 (method를 불문하고 JSON 반환)
message = "광고 메시지 내용..."
result = extractor.process_single_message(message, method='llm')

# 결과 구조
# {
#   "success": true,
#   "data": {
#     "title": "추출된 제목",
#     "unsubscribe_phone": "1504",
#     "message_preview": "메시지 미리보기..."
#   },
#   "metadata": {
#     "method": "llm",
#     "message_length": 188
#   }
# }
```

### 배치 파일 처리 API

```python
from quick_extractor import MessageInfoExtractor

# 추출기 초기화
extractor = MessageInfoExtractor(csv_path='./data/messages.txt', use_llm=False)

# 배치 파일 처리
result = extractor.process_batch_file('./data/messages.txt', method='textrank')

# 결과 구조
# {
#   "success": true,
#   "data": {
#     "messages": [...],  # 추출 결과 배열
#     "statistics": {
#       "total_messages": 11,
#       "with_unsubscribe_phone": 11,
#       "extraction_rate": 100.0
#     }
#   },
#   "metadata": {
#     "method": "textrank",
#     "file_path": "./data/messages.txt",
#     "file_type": "text"
#   }
# }
```

### API 테스트 실행

```bash
python test_quick_extractor_api.py
```

## 참고

- 원본 데이터: `./data/mms_data_251023.csv`
- 기본은 NLP 방법 (빠르고 의존성 없음)
- LLM은 고품질이 필요할 때만 사용 권장
- `mms_extractor.py`와 동일한 LLM 설정 사용
- **API 통합**: `process_single_message()`, `process_batch_file()` 메서드로 JSON 반환

