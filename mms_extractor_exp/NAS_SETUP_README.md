# 📚 Entity DAG API - NAS 서버 설정 가이드

## 🎯 개요

Entity DAG API의 이미지를 로컬이 아닌 NAS 서버(172.27.7.58:/aos_ext)에 저장하도록 설정합니다.

**코드 수정 없이** NFS 마운트와 심볼릭 링크만으로 구현됩니다.

**환경**: Red Hat Enterprise Linux 8.10

---

## 🚀 빠른 시작 (3단계)

### **1단계: NAS 마운트** (sudo 필요)

```bash
cd $(pwd)/mms_extractor_exp
sudo bash scripts/setup_nas_mount.sh
```

이 스크립트는:
- `/mnt/nas_dag_images` 디렉토리 생성 (Linux 표준 위치)
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
- 심볼릭 링크 생성: `./dag_images` → `/mnt/nas_dag_images/dag_images`
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
  "dag_image_path": "/path/to/mms_extractor_exp/dag_images/dag_abc123.png"
}
```

**실제 파일 위치:**
```
/mnt/nas_dag_images/dag_images/dag_abc123.png
↓
NAS: 172.27.7.58:/aos_ext/dag_images/dag_abc123.png
```

---

## 🔧 문제 해결

### **Red Hat Linux 특정 설정**

#### **NFS 클라이언트 패키지 설치**

```bash
# nfs-utils 설치 확인
rpm -qa | grep nfs-utils

# 설치되어 있지 않으면
sudo yum install nfs-utils -y

# NFS 서비스 활성화
sudo systemctl enable --now nfs-client.target
sudo systemctl enable --now rpcbind
```

#### **방화벽 설정 (필요 시)**

```bash
# 방화벽 상태 확인
sudo firewall-cmd --state

# NFS 클라이언트 허용
sudo firewall-cmd --permanent --add-service=nfs
sudo firewall-cmd --permanent --add-service=rpc-bind
sudo firewall-cmd --permanent --add-service=mountd
sudo firewall-cmd --reload
```

---

### NAS 마운트가 안 될 때

```bash
# NFS 서버 확인
showmount -e 172.27.7.58

# 네트워크 연결 확인
ping -c 3 172.27.7.58

# NFS 포트 확인
nc -zv 172.27.7.58 2049
nc -zv 172.27.7.58 111
```

### 권한 문제

```bash
# NAS 디렉토리 권한 재설정
sudo chown $(whoami):$(id -gn) /mnt/nas_dag_images/dag_images
sudo chmod 755 /mnt/nas_dag_images/dag_images
```

### 마운트 해제

```bash
# 강제 언마운트
sudo umount -f /mnt/nas_dag_images

# 심볼릭 링크 제거
cd $(pwd)/mms_extractor_exp
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
└── dag_images -> /mnt/nas_dag_images/dag_images  (설정 후)
```

---

## 🎯 적용 범위

현재 설정: `mms_extractor_exp`

다른 디렉토리에도 적용하려면:
```bash
# mms_extractor_dev
cd /path/to/mms_extractor_dev
rm -rf ./dag_images
ln -s /mnt/nas_dag_images/dag_images ./dag_images

# mms_extractor_prd
cd /path/to/mms_extractor_prd
rm -rf ./dag_images
ln -s /mnt/nas_dag_images/dag_images ./dag_images
```

---

## 📞 지원

문제가 발생하면 다음 명령어로 상태를 확인하세요:
```bash
cd $(pwd)/mms_extractor_exp
bash scripts/verify_nas_setup.sh
```

---

## 🔧 Red Hat Linux 8.10 전용 팁

### SELinux 설정 (필요 시)

```bash
# SELinux 상태 확인
getenforce

# NFS 관련 SELinux boolean 설정
sudo setsebool -P use_nfs_home_dirs 1
sudo setsebool -P nfs_export_all_rw 1

# SELinux 컨텍스트 설정
sudo semanage fcontext -a -t nfs_t "/mnt/nas_dag_images(/.*)?"
sudo restorecon -R /mnt/nas_dag_images
```

### 자동 마운트 확인

```bash
# systemd 마운트 유닛 상태 확인
sudo systemctl list-units --type=mount | grep mnt

# 마운트 재시도
sudo systemctl daemon-reload
sudo mount -a
```

### 로그 확인

```bash
# NFS 관련 로그
sudo journalctl -u nfs-client.target -n 50

# 마운트 관련 로그
dmesg | grep -i nfs
```

---

**작성일**: 2024-10-14  
**환경**: Red Hat Enterprise Linux 8.10  
**상태**: Linux 환경 최적화 완료
