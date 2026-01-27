# Experiments 디렉토리

이 디렉토리는 **새로운 기능 개발 및 검증**을 위한 실험용 공간입니다.

## 목적

- 프롬프트 개선 테스트
- 새로운 기능 프로토타이핑
- 빠른 검증 및 디버깅
- 실제 코드를 수정하지 않고 안전하게 실험

## 주요 특징

✅ **Git에서 제외**: 이 디렉토리의 모든 파일은 자동으로 Git에서 제외됩니다
✅ **API 키 안전**: 민감한 정보를 포함해도 커밋 걱정 없음
✅ **빠른 반복**: 실제 코드 수정 없이 빠르게 테스트 가능

## 사용 방법

### 1. 빠른 테스트
```bash
python experiments/quick_test.py
```

### 2. 프롬프트 테스트
```bash
python experiments/prompt_test.py
```

### 3. 커스텀 실험
원하는 실험 스크립트를 만들어서 사용:
```bash
python experiments/my_experiment.py
```

## 예시: 프롬프트 개선 워크플로우

1. **`prompt_test.py` 복사**
   ```bash
   cp experiments/prompt_test.py experiments/my_prompt_experiment.py
   ```

2. **새 프롬프트 작성**
   - 파일 내에서 `test_improved_prompt()` 함수 수정
   - 새로운 프롬프트 로직 구현

3. **비교 테스트**
   ```python
   # 기존 프롬프트
   result_old = test_original_prompt()
   
   # 새 프롬프트
   result_new = test_improved_prompt()
   
   # 비교
   compare_results(result_old, result_new)
   ```

4. **만족스러우면 실제 코드에 적용**
   - `prompts/` 디렉토리의 실제 파일 수정
   - 테스트 실행
   - 커밋

## 권장 패턴

### A/B 테스트
```python
def test_version_a():
    # 버전 A 테스트
    pass

def test_version_b():
    # 버전 B 테스트
    pass

def compare():
    # 두 버전 비교
    pass
```

### 성능 측정
```python
import time

start = time.time()
result = extractor.process_message(message)
elapsed = time.time() - start

print(f"처리 시간: {elapsed:.2f}초")
```

### 배치 테스트
```python
test_cases = [
    "테스트 메시지 1",
    "테스트 메시지 2",
    "테스트 메시지 3"
]

for msg in test_cases:
    result = extractor.process_message(msg)
    # 결과 검증
```

## 주의사항

⚠️ **이 디렉토리의 파일은 Git에 커밋되지 않습니다**
- 중요한 코드는 반드시 실제 프로젝트 파일로 옮기세요
- 백업이 필요한 실험은 별도로 저장하세요

## 추가 팁

- **Jupyter Notebook 대안**: Python 스크립트가 더 가볍고 빠릅니다
- **결과 저장**: JSON 파일로 결과를 저장하여 나중에 비교 가능
- **로깅**: 상세한 로그를 남겨서 디버깅에 활용
