# 📊 DAG 이미지 API 가이드

## 🎯 개요

Entity DAG API는 이제 **외부 시스템에서 접근 가능한 HTTP URL**을 반환합니다.
NAS 서버에 이미지를 저장하고, HTTP를 통해 제공합니다.

---

## 🔄 변경 사항

### **이전 (Before)**
로컬 파일 경로만 반환 (외부 시스템 접근 불가):
```json
{
  "dag_image_path": "/Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp/dag_images/dag_abc123.png"
}
```

### **현재 (After)**
HTTP URL과 로컬 경로를 모두 반환:
```json
{
  "dag_image_url": "http://127.0.0.1:8000/dag_images/dag_abc123.png",
  "dag_image_path": "/Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp/dag_images/dag_abc123.png"
}
```

---

## 🚀 API 사용법

### **1. DAG 생성 요청**

```python
import requests

response = requests.post('http://127.0.0.1:8000/dag', json={
    "message": "고객이 가입하면 혜택을 받고 만족도가 향상된다",
    "llm_model": "ax",
    "save_dag_image": True
})

result = response.json()
```

### **2. 응답 형식**

```json
{
  "success": true,
  "result": {
    "dag_section": "...",
    "dag_raw": "...",
    "dag_json": { ... },
    "analysis": {
      "num_nodes": 3,
      "num_edges": 2,
      ...
    },
    "dag_image_url": "http://127.0.0.1:8000/dag_images/dag_abc123.png",
    "dag_image_path": "/Users/yongwook/.../dag_images/dag_abc123.png"
  },
  "metadata": {
    "llm_model": "ax",
    "processing_time_seconds": 2.345,
    ...
  }
}
```

### **3. 이미지 접근**

#### **방법 1: 웹 브라우저**
```
http://127.0.0.1:8000/dag_images/dag_abc123.png
```

#### **방법 2: Python**
```python
import requests
from PIL import Image
from io import BytesIO

# 이미지 URL 가져오기
image_url = result['result']['dag_image_url']

# 이미지 다운로드
img_response = requests.get(image_url)
img = Image.open(BytesIO(img_response.content))
img.show()
```

#### **방법 3: curl**
```bash
curl -O http://127.0.0.1:8000/dag_images/dag_abc123.png
```

---

## 🌐 외부 시스템 통합

### **시나리오 1: 웹 애플리케이션**

```javascript
// React/Vue/Angular 등
async function fetchDAG(message) {
  const response = await fetch('http://api-server:8000/dag', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: message,
      save_dag_image: true
    })
  });
  
  const result = await response.json();
  
  // 이미지 URL을 직접 사용
  const imageUrl = result.result.dag_image_url;
  document.getElementById('dag-image').src = imageUrl;
}
```

### **시나리오 2: 모바일 앱**

```swift
// iOS Swift
let imageUrl = URL(string: result.dag_image_url)
imageView.load(url: imageUrl)
```

```kotlin
// Android Kotlin
Glide.with(context)
    .load(result.dag_image_url)
    .into(imageView)
```

### **시나리오 3: 다른 백엔드 서비스**

```python
# Flask 예시
@app.route('/process')
def process():
    # DAG API 호출
    dag_response = requests.post('http://api-server:8000/dag', json={
        'message': message,
        'save_dag_image': True
    })
    
    dag_result = dag_response.json()
    
    # 이미지 URL을 클라이언트에 전달
    return jsonify({
        'dag_image': dag_result['result']['dag_image_url'],
        'analysis': dag_result['result']['analysis']
    })
```

---

## 🗂️ 파일 저장 위치

### **실제 저장 위치**
이미지는 NAS 서버에 저장됩니다:
```
로컬 심볼릭 링크:    ./dag_images/
실제 마운트 위치:    /mnt/nas_dag_images/dag_images/  (Linux)
NAS 서버 경로:       172.27.7.58:/aos_ext/dag_images/
```

### **HTTP 접근 경로**
```
http://127.0.0.1:8000/dag_images/{filename}
또는
http://api-server-ip:8000/dag_images/{filename}
```

---

## 🔧 API 서버 설정

### **1. 포트 변경**

```bash
# 기본 (8000)
python api.py

# 다른 포트 사용
python api.py --port 8080
```

이 경우 URL도 변경됩니다:
```
http://127.0.0.1:8080/dag_images/dag_abc123.png
```

### **2. 외부 접근 허용**

```bash
# 모든 IP에서 접근 가능
python api.py --host 0.0.0.0 --port 8000
```

이 경우 외부에서 접근:
```
http://<서버-IP>:8000/dag_images/dag_abc123.png
```

---

## ⚠️ 주의사항

### **1. 네트워크 접근**
- API 서버가 **0.0.0.0**으로 실행 중인지 확인
- 방화벽에서 **8000번 포트** 개방 확인
- NAS 마운트가 정상적으로 되어 있는지 확인

### **2. CORS (크로스 오리진)**
웹 브라우저에서 접근 시 CORS가 이미 활성화되어 있음:
```python
# api.py에 이미 설정됨
CORS(app)
```

### **3. 이미지 파일 크기**
- PNG 형식, 일반적으로 50-500KB
- 대역폭 고려하여 필요 시 압축 또는 썸네일 제공

---

## 🧪 테스트

### **테스트 스크립트 실행**

```bash
cd /Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp
python api_test.py
```

### **예상 출력**

```json
{
  "success": true,
  "result": {
    ...
    "dag_image_url": "http://127.0.0.1:8000/dag_images/dag_7624a7d9dc579604383d572d683a433d2c942896ac4eaab92562f8ebd1814b0d.png",
    "dag_image_path": "/Users/yongwook/.../dag_images/dag_7624a7d9dc579604383d572d683a433d2c942896ac4eaab92562f8ebd1814b0d.png"
  }
}
================================================================================
📊 DAG 이미지 URL (외부 시스템 접근 가능):
http://127.0.0.1:8000/dag_images/dag_7624a7d9dc579604383d572d683a433d2c942896ac4eaab92562f8ebd1814b0d.png
================================================================================
```

### **브라우저에서 확인**

1. API 서버 실행:
   ```bash
   python api.py
   ```

2. 브라우저에서 열기:
   ```
   http://127.0.0.1:8000/dag_images/dag_abc123.png
   ```

---

## 📖 추가 엔드포인트

### **GET /dag_images/<filename>**

DAG 이미지 파일을 HTTP로 제공합니다.

**요청 예시:**
```bash
curl http://127.0.0.1:8000/dag_images/dag_abc123.png --output dag.png
```

**응답:**
- **200 OK**: 이미지 파일 (image/png)
- **404 Not Found**: 파일 없음

**에러 응답 예시:**
```json
{
  "success": false,
  "error": "Image not found"
}
```

---

## 🎯 마이그레이션 가이드

기존 코드에서 `dag_image_path`를 사용하고 있다면:

### **변경 전:**
```python
# 로컬 경로만 사용
local_path = result['result']['dag_image_path']
# 외부 시스템에서는 접근 불가 ❌
```

### **변경 후:**
```python
# HTTP URL 사용 (권장)
image_url = result['result']['dag_image_url']
# 외부 시스템에서 접근 가능 ✅

# 로컬 경로도 여전히 사용 가능
local_path = result['result']['dag_image_path']
```

---

## 🔗 관련 문서

- [NAS 서버 설정 가이드](./NAS_SETUP_README.md)
- [API 사용 가이드](./USAGE_GUIDE.md)
- [API 멀티프로세스 가이드](./API_MULTIPROCESS_GUIDE.md)

---

**작성일**: 2024-10-14  
**버전**: 1.0.0

