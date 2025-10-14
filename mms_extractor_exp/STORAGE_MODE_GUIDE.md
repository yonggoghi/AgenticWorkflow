# 📁 DAG 이미지 저장 모드 가이드

## 🎯 개요

Entity DAG API는 이제 **로컬 저장**과 **NAS 저장** 두 가지 모드를 지원합니다.
API 실행 시 `--storage` 옵션으로 저장 위치를 선택할 수 있습니다.

---

## 🚀 사용 방법

### **방법 1: 로컬 저장 (기본값, 권장)**

방화벽 문제가 있거나 NAS 접근이 불가능할 때 사용:

```bash
cd /Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp

# 로컬 저장 모드 (기본값)
python api.py --storage local

# 또는 간단히
python api.py
```

**저장 위치:**
```
./dag_images_local/dag_abc123.png
```

**장점:**
- ✅ 방화벽/네트워크 문제 없음
- ✅ 빠른 읽기/쓰기 속도
- ✅ 즉시 사용 가능
- ✅ NAS 마운트 불필요

**단점:**
- ❌ 로컬 디스크 용량 사용
- ❌ 서버 재시작 시 백업 필요

---

### **방법 2: NAS 저장**

NAS 서버 마운트가 완료되었을 때 사용:

```bash
# 1. NAS 마운트 (사전 작업)
sudo bash scripts/setup_nas_mount.sh
bash scripts/setup_symlink.sh

# 2. NAS 저장 모드로 API 실행
python api.py --storage nas
```

**저장 위치:**
```
./dag_images/dag_abc123.png  (심볼릭 링크)
 ↓
/mnt/nas_dag_images/dag_images/dag_abc123.png
 ↓
NAS: 172.27.7.58:/aos_ext/dag_images/dag_abc123.png
```

**장점:**
- ✅ 중앙 저장소 (여러 서버 공유 가능)
- ✅ 로컬 디스크 용량 절약
- ✅ 자동 백업 (NAS 기능)
- ✅ 대용량 저장 가능

**단점:**
- ❌ NAS 마운트 필요
- ❌ 네트워크 의존성
- ❌ 방화벽 설정 필요할 수 있음

---

## 📊 저장 모드 비교

| 항목 | 로컬 저장 (`local`) | NAS 저장 (`nas`) |
|------|-------------------|-----------------|
| **설정 난이도** | ⭐ 매우 쉬움 | ⭐⭐⭐ 중간 |
| **속도** | ⚡⚡⚡ 빠름 | ⚡⚡ 보통 |
| **용량** | 로컬 디스크 제한 | NAS 용량 제한 |
| **백업** | 수동 필요 | 자동 (NAS 기능) |
| **공유** | 불가 | 가능 |
| **네트워크 의존** | 없음 | 있음 |
| **권장 상황** | 개발/테스트 | 프로덕션 |

---

## 🔧 상세 사용 가이드

### **로컬 저장 모드 (local)**

#### **1. API 서버 시작**

```bash
cd /path/to/mms_extractor_exp

# 기본 (로컬 저장)
python api.py

# 명시적으로 로컬 저장 지정
python api.py --storage local --port 8000
```

#### **2. 로그 확인**

API 시작 시 다음과 같은 로그가 표시됩니다:

```
📁 DAG 저장 모드: local - Local disk storage (no NAS required)
📂 DAG 저장 경로: dag_images_local
```

#### **3. DAG 생성 테스트**

```bash
# 다른 터미널에서
python api_test.py
```

#### **4. 저장 위치 확인**

```bash
ls -la ./dag_images_local/
# dag_abc123.png 파일들이 표시됨
```

#### **5. HTTP 접근**

```bash
# 브라우저 또는 curl
curl http://127.0.0.1:8000/dag_images/dag_abc123.png --output test.png
```

---

### **NAS 저장 모드 (nas)**

#### **사전 조건**

NAS 마운트가 완료되어 있어야 합니다:

```bash
# 1. NAS 마운트 확인
mount | grep nas_dag_images

# 2. 마운트 안 되어 있으면
sudo bash scripts/setup_nas_mount.sh
bash scripts/setup_symlink.sh
bash scripts/verify_nas_setup.sh
```

#### **1. API 서버 시작**

```bash
python api.py --storage nas --port 8000
```

#### **2. 로그 확인**

```
📁 DAG 저장 모드: nas - NAS server storage (requires NAS mount)
📂 DAG 저장 경로: dag_images
```

#### **3. DAG 생성 및 확인**

```bash
# 테스트
python api_test.py

# NAS에 저장되었는지 확인
ls -la /mnt/nas_dag_images/dag_images/
```

---

## 🔄 모드 전환

### **로컬에서 NAS로 전환**

```bash
# 1. API 중지 (Ctrl+C)

# 2. NAS 마운트 (아직 안 했으면)
sudo bash scripts/setup_nas_mount.sh
bash scripts/setup_symlink.sh

# 3. 기존 로컬 이미지를 NAS로 복사 (선택사항)
cp -r ./dag_images_local/* /mnt/nas_dag_images/dag_images/

# 4. NAS 모드로 재시작
python api.py --storage nas
```

### **NAS에서 로컬로 전환**

```bash
# 1. API 중지

# 2. NAS 이미지를 로컬로 복사 (선택사항)
cp -r /mnt/nas_dag_images/dag_images/* ./dag_images_local/

# 3. 로컬 모드로 재시작
python api.py --storage local
```

---

## 🌐 환경변수 사용

코드나 스크립트에서 환경변수로 제어 가능:

### **방법 1: 환경변수 설정**

```bash
# 로컬 저장
export DAG_STORAGE_MODE=local
python api.py

# NAS 저장
export DAG_STORAGE_MODE=nas
python api.py
```

### **방법 2: .env 파일**

```bash
# .env 파일 생성
cat > .env << EOF
DAG_STORAGE_MODE=local
EOF

python api.py
```

### **방법 3: Docker/Kubernetes**

```yaml
# docker-compose.yml
services:
  api:
    environment:
      - DAG_STORAGE_MODE=local
```

---

## 🧪 테스트 방법

### **로컬 저장 테스트**

```bash
# 1. 로컬 모드로 시작
python api.py --storage local &
API_PID=$!

# 2. 테스트
sleep 5
python api_test.py

# 3. 파일 확인
ls -lh ./dag_images_local/

# 4. 종료
kill $API_PID
```

### **NAS 저장 테스트**

```bash
# 1. NAS 마운트 확인
bash scripts/verify_nas_setup.sh

# 2. NAS 모드로 시작
python api.py --storage nas &
API_PID=$!

# 3. 테스트
sleep 5
python api_test.py

# 4. NAS에 저장되었는지 확인
ls -lh /mnt/nas_dag_images/dag_images/

# 5. 종료
kill $API_PID
```

---

## 📝 API 응답 형식

저장 모드와 관계없이 API 응답은 동일합니다:

```json
{
  "success": true,
  "result": {
    "dag_image_url": "http://127.0.0.1:8000/dag_images/dag_abc123.png",
    "dag_image_path": "/Users/yongwook/.../dag_images_local/dag_abc123.png"
  }
}
```

**참고**: 
- `dag_image_url`: HTTP URL (항상 `/dag_images/` 경로)
- `dag_image_path`: 실제 파일 시스템 경로 (모드에 따라 변경됨)

---

## 🔍 트러블슈팅

### **문제 1: 로컬 모드에서 이미지가 저장 안 됨**

```bash
# 디렉토리 존재 확인
ls -la ./dag_images_local/

# 없으면 생성
mkdir -p ./dag_images_local

# 권한 확인
chmod 755 ./dag_images_local
```

### **문제 2: NAS 모드에서 마운트 오류**

```bash
# 마운트 상태 확인
mount | grep nas

# 재마운트
sudo bash scripts/setup_nas_mount.sh
```

### **문제 3: HTTP URL로 이미지 접근 안 됨**

```bash
# 파일 존재 확인
ls -la ./dag_images_local/dag_abc123.png  # 로컬 모드
ls -la ./dag_images/dag_abc123.png        # NAS 모드

# API 로그 확인
# 📊 DAG 이미지 요청: dag_abc123.png (from dag_images_local)
```

### **문제 4: 저장 모드가 변경 안 됨**

```bash
# 환경변수 확인
echo $DAG_STORAGE_MODE

# 재시작 (환경변수 재로드)
python api.py --storage local  # 명령줄 옵션이 우선
```

---

## 📚 관련 파일

### **설정 파일**
- `config/settings.py` - StorageConfig 클래스
- `.env` - 환경변수 설정 (선택사항)

### **코드 파일**
- `api.py` - API 서버 (--storage 옵션 처리)
- `utils.py` - create_dag_diagram 함수 (동적 경로 지원)

### **저장 디렉토리**
- `dag_images_local/` - 로컬 저장 디렉토리
- `dag_images/` - NAS 저장 디렉토리 (심볼릭 링크)

### **관련 가이드**
- `NAS_SETUP_README.md` - NAS 서버 설정 가이드
- `NAS_FIREWALL_SOLUTIONS.md` - 방화벽 이슈 해결
- `DAG_IMAGE_API_GUIDE.md` - DAG 이미지 API 사용법

---

## 🎯 권장 사용 시나리오

### **개발 환경**
```bash
# 로컬 저장 사용 (빠르고 간단)
python api.py --storage local
```

### **테스트 환경**
```bash
# 로컬 저장 또는 NAS (요구사항에 따라)
python api.py --storage local
```

### **프로덕션 환경**
```bash
# NAS 저장 권장 (중앙 관리, 백업)
python api.py --storage nas
```

### **방화벽 제한 환경**
```bash
# 로컬 저장 필수
python api.py --storage local
```

---

## 🔐 보안 고려사항

### **로컬 저장**
- 파일 권한: `chmod 755 dag_images_local`
- 정기 백업 권장
- 디스크 용량 모니터링

### **NAS 저장**
- NFS 보안 설정 확인
- 방화벽 규칙 검토
- 접근 권한 제한 (IP 기반)

---

## 📞 지원

문제가 발생하면:

1. 로그 확인: API 시작 시 저장 모드 확인
2. 디렉토리 확인: 저장 위치가 올바른지 확인
3. 권한 확인: 파일 생성 권한 확인
4. 네트워크 확인: NAS 모드 시 마운트 상태 확인

---

**작성일**: 2024-10-14  
**버전**: 1.0.0  
**상태**: ✅ 로컬/NAS 이중 모드 지원

