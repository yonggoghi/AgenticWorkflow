# 🔥 NAS 방화벽 이슈 해결 가이드

## 📊 상황 분석

### **확인된 사실**
- ✅ NAS 서버(172.27.7.58)는 **정상 작동 중**
- ✅ 사내 서버에서는 **접속 가능**
- ❌ 현재 macOS 시스템(172.23.25.171)에서는 **접속 불가**

### **원인**
**네트워크 방화벽이 두 서브넷 간 NFS 트래픽을 차단**
```
현재 시스템:  172.23.25.171 (172.23.24.0/22 네트워크)
              ↓ [방화벽 차단]
NAS 서버:     172.27.7.58   (172.27.0.0/16 네트워크)
```

---

## ✅ 해결 방법 (3가지)

---

## 🔧 방법 1: 네트워크 관리자에게 방화벽 규칙 요청 (권장)

### **요청 내용**

네트워크 관리자에게 다음 내용으로 요청하세요:

```
제목: NFS 접속을 위한 방화벽 규칙 추가 요청

안녕하세요,

MMS API 서버에서 NAS 스토리지 접근을 위해 방화벽 규칙 추가를 요청드립니다.

[출발지(Source)]
- IP 주소: 172.23.25.171
- 또는 네트워크: 172.23.24.0/22
- 호스트명: M226G6XVXM

[목적지(Destination)]
- IP 주소: 172.27.7.58
- 프로토콜: TCP, UDP
- 포트: 111, 2049, 4045

[목적]
- DAG 이미지 파일을 NAS에 저장하기 위한 NFS 마운트

[기간]
- 영구 허용 (또는 프로젝트 종료 시까지)

감사합니다.
```

### **필요한 방화벽 규칙**

| 프로토콜 | 포트 | 출발지 | 목적지 | 용도 |
|---------|------|--------|--------|------|
| TCP/UDP | 111 | 172.23.25.171 | 172.27.7.58 | RPC (portmapper) |
| TCP | 2049 | 172.23.25.171 | 172.27.7.58 | NFS |
| TCP/UDP | 4045 | 172.23.25.171 | 172.27.7.58 | NFS Lock Manager |

---

## 🌐 방법 2: VPN 사용 (가장 빠른 해결)

### **2-1. VPN 연결 확인**

```bash
# 현재 VPN 연결 상태 확인
scutil --nc list

# 또는
ifconfig | grep -A 5 utun
```

### **2-2. 회사 VPN 연결**

만약 회사 VPN이 있다면:
```
1. VPN 클라이언트 실행
2. 회사 네트워크에 연결
3. 다시 마운트 시도
```

VPN 연결 후 자동으로 172.27.x.x 네트워크에 접근 가능할 수 있습니다.

### **2-3. VPN 연결 후 테스트**

```bash
# VPN 연결 후
ping -c 3 172.27.7.58
showmount -e 172.27.7.58

# 성공하면
cd /Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp
sudo bash scripts/setup_nas_mount.sh
```

---

## 💻 방법 3: 사내 서버에서 API 실행 (임시 해결책)

NAS 접근이 가능한 사내 서버에서 API를 실행하는 방법:

### **3-1. 코드 배포**

```bash
# 사내 서버 접속
ssh user@internal-server

# 프로젝트 클론 또는 복사
cd /path/to/project
git clone <repository> 
# 또는 scp로 복사

cd mms_extractor_exp
```

### **3-2. NAS 마운트 (사내 서버에서)**

```bash
# 사내 서버에서 실행 (ping 성공 확인됨)
sudo bash scripts/setup_nas_mount.sh
bash scripts/setup_symlink.sh
bash scripts/verify_nas_setup.sh
```

### **3-3. API 서버 실행**

```bash
# 외부 접근 허용
python api.py --host 0.0.0.0 --port 8000
```

### **3-4. 현재 macOS에서 원격 API 사용**

```python
# api_test.py 수정
response = requests.post('http://internal-server:8000/dag', json={
    "message": "...",
    "save_dag_image": True
})

# 이미지 URL
# http://internal-server:8000/dag_images/dag_abc123.png
```

---

## 🏠 방법 4: 로컬 저장 사용 (방화벽 해결 전까지)

NAS 접근 문제를 해결하는 동안 로컬에서 사용:

### **4-1. 현재 상태 유지**

```bash
cd /Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp

# dag_images 디렉토리는 이미 존재 (77개 파일)
ls -la dag_images/

# API 서버 실행
python api.py
```

### **4-2. API 정상 작동**

```python
# 로컬 테스트
python api_test.py

# 응답:
{
  "dag_image_url": "http://127.0.0.1:8000/dag_images/dag_abc123.png",
  "dag_image_path": "/Users/yongwook/.../dag_images/dag_abc123.png"
}
```

### **4-3. 이미지 접근**

```bash
# 브라우저에서
http://127.0.0.1:8000/dag_images/dag_abc123.png

# 또는 로컬 파일로
open ./dag_images/dag_abc123.png
```

**장점**: 코드 변경 없이 즉시 사용 가능  
**단점**: 로컬 디스크 용량 사용, 백업 필요

---

## 🔍 방법 5: SSH 터널링 (고급)

사내 서버를 통해 NAS 접근:

### **5-1. SSH 터널 생성**

```bash
# 로컬에서 실행
ssh -L 2049:172.27.7.58:2049 \
    -L 111:172.27.7.58:111 \
    user@internal-server -N -f
```

### **5-2. localhost로 마운트**

```bash
# 스크립트 수정
nano scripts/setup_nas_mount.sh

# NAS_IP 변경
NAS_IP="localhost"  # 또는 127.0.0.1
```

### **5-3. 마운트 실행**

```bash
sudo bash scripts/setup_nas_mount.sh
```

**주의**: 이 방법은 SSH 연결이 끊어지면 마운트도 해제됩니다.

---

## 📊 방법 비교

| 방법 | 난이도 | 속도 | 안정성 | 권장도 |
|------|--------|------|--------|--------|
| **1. 방화벽 규칙 요청** | 쉬움 | 느림 (승인 대기) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **2. VPN 사용** | 쉬움 | 빠름 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **3. 사내 서버에서 실행** | 중간 | 빠름 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **4. 로컬 저장** | 매우 쉬움 | 즉시 | ⭐⭐⭐ | ⭐⭐⭐ |
| **5. SSH 터널링** | 어려움 | 중간 | ⭐⭐ | ⭐⭐ |

---

## 🎯 추천 순서

### **즉시 사용 필요 시**
```
1. 방법 4 (로컬 저장) - 지금 바로 사용
2. 병행하여 방법 1 (방화벽 규칙 요청) 진행
```

### **VPN 사용 가능 시**
```
1. 방법 2 (VPN) - 가장 간단
2. VPN 연결 후 NAS 마운트
```

### **사내 서버 접근 가능 시**
```
1. 방법 3 (사내 서버에서 실행)
2. 원격 API로 사용
```

---

## 🔄 방화벽 규칙 승인 후 적용

방화벽 규칙이 적용되면:

```bash
# 1. 네트워크 연결 확인
ping -c 3 172.27.7.58
showmount -e 172.27.7.58

# 2. NAS 마운트
cd /Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp
sudo bash scripts/setup_nas_mount.sh

# 3. 심볼릭 링크 생성
bash scripts/setup_symlink.sh

# 4. 검증
bash scripts/verify_nas_setup.sh

# 5. API 테스트
python api.py &
python api_test.py
```

---

## 📞 문의 정보

### **네트워크 관리자에게 전달할 정보**

```
[시스템 정보]
호스트명: M226G6XVXM
현재 IP: 172.23.25.171
네트워크: 172.23.24.0/22

[NAS 정보]
NAS IP: 172.27.7.58
공유 폴더: /aos_ext
프로토콜: NFS

[필요 포트]
- 111 (RPC)
- 2049 (NFS)
- 4045 (NFS Lock Manager)

[테스트 명령어]
ping 172.27.7.58
showmount -e 172.27.7.58
nc -zv 172.27.7.58 2049
```

---

## 🎯 결론

**현재 상황**: 방화벽으로 인한 NAS 접근 차단

**즉시 조치**: 
- ✅ 방법 4 (로컬 저장) 사용하여 개발/테스트 계속 진행
- ✅ 방법 1 (방화벽 규칙 요청)을 네트워크 팀에 제출

**장기 해결**:
- ✅ 방화벽 규칙 승인 후 NAS 마운트
- ✅ 또는 VPN 사용

어떤 방법을 선호하시나요? 도와드리겠습니다! 🚀

---

**작성일**: 2024-10-14  
**상태**: 방화벽 이슈 확인됨

