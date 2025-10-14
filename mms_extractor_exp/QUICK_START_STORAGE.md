# 🚀 빠른 시작 가이드 - 저장 모드 선택

## ⚡ 즉시 사용 (로컬 저장)

**방화벽 문제가 있거나 즉시 사용하고 싶으면** → **로컬 저장 모드 (권장)**

```bash
cd /path/to/mms_extractor_exp

# API 서버 시작 (기본값 = 로컬 저장)
python api.py

# 또는 명시적으로
python api.py --storage local
```

✅ **완료!** 바로 사용 가능합니다.

---

## 📊 두 가지 저장 모드

### **1. 로컬 저장 (local) - 기본값** ⭐ 권장

```bash
python api.py --storage local
```

| 항목 | 설명 |
|------|------|
| **저장 위치** | `./dag_images_local/` |
| **장점** | • 즉시 사용 가능<br>• 빠른 속도<br>• 네트워크/방화벽 문제 없음 |
| **단점** | • 로컬 디스크 사용 |
| **권장** | 개발, 테스트, 방화벽 제한 환경 |

---

### **2. NAS 저장 (nas)**

```bash
# 사전 작업 (한 번만)
sudo bash scripts/setup_nas_mount.sh
bash scripts/setup_symlink.sh

# API 시작
python api.py --storage nas
```

| 항목 | 설명 |
|------|------|
| **저장 위치** | `./dag_images/` → NAS 서버 |
| **장점** | • 중앙 저장<br>• 자동 백업<br>• 대용량 저장 |
| **단점** | • NAS 마운트 필요<br>• 방화벽 설정 필요할 수 있음 |
| **권장** | 프로덕션, 팀 공유 |

---

## 🧪 테스트

### **저장 모드 확인**

```bash
bash test_storage_modes.sh
```

### **API 테스트**

```bash
# 1. API 시작 (로컬 저장)
python api.py --storage local

# 2. 다른 터미널에서 테스트
python api_test.py

# 3. 저장 확인
ls -la ./dag_images_local/
```

---

## 📝 API 사용 예시

저장 모드와 **상관없이** API 사용법은 동일:

```python
import requests

response = requests.post('http://127.0.0.1:8000/dag', json={
    "message": "고객이 가입하면 혜택을 받고 만족도가 향상된다",
    "save_dag_image": True
})

result = response.json()
image_url = result['result']['dag_image_url']
# http://127.0.0.1:8000/dag_images/dag_abc123.png
```

---

## 🔄 모드 전환

### **현재 사용 중인 모드는?**

API 로그에서 확인:
```
📁 DAG 저장 모드: local - Local disk storage (no NAS required)
📂 DAG 저장 경로: dag_images_local
```

### **모드 변경 방법**

API를 재시작하면서 `--storage` 옵션 변경:

```bash
# 로컬 → NAS
python api.py --storage nas

# NAS → 로컬
python api.py --storage local
```

---

## 🎯 상황별 권장 사항

| 상황 | 권장 모드 | 명령어 |
|------|----------|--------|
| 개발 중 | `local` | `python api.py` |
| 테스트 중 | `local` | `python api.py --storage local` |
| 방화벽 제한 | `local` | `python api.py --storage local` |
| NAS 접근 가능 | `nas` | `python api.py --storage nas` |
| 프로덕션 | `nas` | `python api.py --storage nas` |

---

## 📚 상세 가이드

- **저장 모드 전체 가이드**: `STORAGE_MODE_GUIDE.md`
- **NAS 설정 가이드**: `NAS_SETUP_README.md`
- **방화벽 문제 해결**: `NAS_FIREWALL_SOLUTIONS.md`
- **DAG API 사용법**: `DAG_IMAGE_API_GUIDE.md`

---

## ⚡ TL;DR (가장 빠른 방법)

```bash
cd /Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp
python api.py
# 끝!
```

기본값(로컬 저장)으로 바로 사용 가능합니다. 🎉

---

**작성일**: 2024-10-14  
**상태**: ✅ 즉시 사용 가능

