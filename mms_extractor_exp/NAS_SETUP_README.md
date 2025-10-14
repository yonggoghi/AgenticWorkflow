# 📚 Entity DAG API - NAS 서버 설정 가이드

## 🎯 개요

Entity DAG API의 이미지를 로컬이 아닌 NAS 서버(172.27.7.58:/aos_ext)에 저장하도록 설정합니다.

**코드 수정 없이** NFS 마운트와 심볼릭 링크만으로 구현됩니다.

---

## 🚀 빠른 시작 (3단계)

### **1단계: NAS 마운트** (sudo 필요)

```bash
cd /Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp
sudo bash scripts/setup_nas_mount.sh
```

이 스크립트는:
- `/Volumes/nas_dag_images` 디렉토리 생성 (macOS 표준 위치)
- NAS(172.27.7.58:/aos_ext) 마운트
- `dag_images` 디렉토리 생성 및 권한 설정
- 쓰기 권한 테스트

---

### **2단계: 심볼릭 링크 설정** (sudo 불필요)

```bash
bash scripts/setup_symlink.sh
```

이 스크립트는:
- 기존 `dag_images` 백업 (날짜/시간 포함)
- 기존 이미지를 NAS로 복사
- 심볼릭 링크 생성: `./dag_images` → `/Volumes/nas_dag_images/dag_images`
- 설정 검증

---

### **3단계: 검증**

```bash
bash scripts/verify_nas_setup.sh
```

모든 설정이 정상인지 확인합니다.

---

## 🔄 선택사항: 영구 마운트 (재부팅 후에도 유지)

```bash
sudo bash scripts/setup_fstab.sh
```

`/etc/fstab`에 NAS 마운트를 추가하여 시스템 재부팅 후에도 자동으로 마운트됩니다.

---

## ✅ 완료 후 상태

### API 코드 변경: 없음 ✅
### DAG 이미지 저장 위치: NAS ✅

**API 응답 (변경 없음):**
```json
{
  "dag_image_path": "/Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp/dag_images/dag_abc123.png"
}
```

**실제 파일 위치:**
```
/Volumes/nas_dag_images/dag_images/dag_abc123.png
↓
NAS: 172.27.7.58:/aos_ext/dag_images/dag_abc123.png
```

---

## 🔧 문제 해결

### macOS 특정 이슈: /mnt 사용 불가

macOS Catalina (10.15) 이후로 루트 파일시스템이 읽기 전용이므로 `/mnt` 사용 불가합니다.
이 가이드는 **`/Volumes`** 를 사용하도록 설정되어 있습니다 (macOS 표준).

```bash
# ❌ macOS에서 작동하지 않음
/mnt/nas_dag_images

# ✅ macOS에서 작동함
/Volumes/nas_dag_images
```

---

### NAS 마운트가 안 될 때

```bash
# NFS 서버 확인
showmount -e 172.27.7.58

# 네트워크 연결 확인
ping 172.27.7.58

# NFS 포트 확인
nc -zv 172.27.7.58 2049
```

### 권한 문제

```bash
# NAS 디렉토리 권한 재설정
sudo chown yongwook:staff /Volumes/nas_dag_images/dag_images
sudo chmod 755 /Volumes/nas_dag_images/dag_images
```

### 마운트 해제

```bash
# 강제 언마운트
sudo umount -f /Volumes/nas_dag_images

# 심볼릭 링크 제거
cd /Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp
rm dag_images

# 백업 복원 (필요 시)
mv dag_images_backup_YYYYMMDD_HHMMSS dag_images
```

---

## 📂 파일 구조

```
mms_extractor_exp/
├── scripts/
│   ├── setup_nas_mount.sh      (NAS 마운트 - sudo 필요)
│   ├── setup_symlink.sh        (심볼릭 링크 생성)
│   ├── setup_fstab.sh          (영구 마운트 설정 - sudo 필요)
│   └── verify_nas_setup.sh     (설정 검증)
├── NAS_SETUP_README.md         (이 파일)
└── dag_images -> /Volumes/nas_dag_images/dag_images  (설정 후)
```

---

## 🎯 적용 범위

현재 설정: `mms_extractor_exp`

다른 디렉토리에도 적용하려면:
```bash
# mms_extractor_dev
cd /Users/yongwook/workspace/AgenticWorkflow/mms_extractor_dev
rm -rf ./dag_images
ln -s /Volumes/nas_dag_images/dag_images ./dag_images

# mms_extractor_prd
cd /Users/yongwook/workspace/AgenticWorkflow/mms_extractor_prd
rm -rf ./dag_images
ln -s /Volumes/nas_dag_images/dag_images ./dag_images
```

---

## 📞 지원

문제가 발생하면 다음 명령어로 상태를 확인하세요:
```bash
cd /Users/yongwook/workspace/AgenticWorkflow/mms_extractor_exp
bash scripts/verify_nas_setup.sh
```

