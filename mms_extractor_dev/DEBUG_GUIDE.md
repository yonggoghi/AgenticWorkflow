# DAG 이미지 표시 문제 디버깅 가이드

## 🔍 문제 진단 단계

### 1단계: 서버 상태 확인

```bash
# 데모 서버가 실행 중인지 확인
curl http://localhost:8080/health

# DAG 이미지 디렉토리 확인
ls -la dag_images/

# 특정 이미지 파일 존재 확인
ls -la dag_images/dag_7918d9d4d4c330a6d67eec805db2f25918f6da467b9c0f04752a6d08c07595be.png
```

### 2단계: API 엔드포인트 테스트

```bash
# 해시 계산 API 테스트
curl -X POST "http://localhost:8080/api/calculate-hash" \
  -H "Content-Type: application/json" \
  -d '{"message": "[SK텔레콤] ZEM폰 포켓몬에디션3 안내\n(광고)[SKT] 우리 아이 첫 번째 스마트폰, ZEM 키즈폰__#04 고객님, 안녕하세요!\n우리 아이 스마트폰 고민 중이셨다면, 자녀 스마트폰 관리 앱 ZEM이 설치된 SKT만의 안전한 키즈폰,\nZEM폰 포켓몬에디션3으로 우리 아이 취향을 저격해 보세요!\n\n✨ 특별 혜택\n- 월 요금 20% 할인 (첫 6개월)\n- 포켓몬 케이스 무료 증정\n- ZEM 프리미엄 서비스 3개월 무료\n\n📞 문의: 1588-0011 (평일 9시-18시)\n🏪 가까운 T world 매장 방문\n🌐 www.tworld.co.kr\n\n수신거부 080-011-0000"}'

# 직접 이미지 접근 테스트
curl -I "http://localhost:8080/dag_images/dag_7918d9d4d4c330a6d67eec805db2f25918f6da467b9c0f04752a6d08c07595be.png"
```

### 3단계: 브라우저 디버깅

1. **브라우저에서 테스트 페이지 열기**:
   - http://localhost:8080/test_image.html

2. **개발자 도구 열기** (F12):
   - **Console 탭**: JavaScript 오류 확인
   - **Network 탭**: HTTP 요청/응답 확인
   - **Elements 탭**: DOM 구조 확인

3. **메인 데모 페이지 테스트**:
   - http://localhost:8080/
   - 첫 번째 샘플 메시지 클릭
   - "엔티티 DAG 추출" 체크박스 활성화
   - "정보 추출 실행" 버튼 클릭
   - DAG 탭 확인

### 4단계: 예상되는 문제점들

#### 문제 1: CORS 오류
**증상**: 브라우저 콘솔에 CORS 관련 오류
**해결**: 데모 서버에 CORS 헤더가 이미 설정되어 있음

#### 문제 2: 이미지 경로 오류
**증상**: 404 Not Found 오류
**해결**: 
```javascript
// 브라우저 콘솔에서 직접 테스트
fetch('/dag_images/dag_7918d9d4d4c330a6d67eec805db2f25918f6da467b9c0f04752a6d08c07595be.png')
  .then(response => console.log('이미지 응답:', response.status))
```

#### 문제 3: JavaScript 실행 오류
**증상**: 함수 호출 오류, undefined 변수
**해결**: 브라우저 콘솔에서 오류 메시지 확인

#### 문제 4: 이미지 로드 타이밍 문제
**증상**: 이미지가 간헐적으로만 표시됨
**해결**: setTimeout 지연 시간 조정

### 5단계: 수동 테스트

브라우저 콘솔에서 다음 코드를 직접 실행:

```javascript
// 1. 해시 계산 API 테스트
fetch('/api/calculate-hash', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: document.getElementById('message').value})
})
.then(response => response.json())
.then(data => {
    console.log('해시 데이터:', data);
    if (data.image_exists) {
        console.log('이미지 URL:', data.url);
        // 이미지 직접 로드 테스트
        const img = new Image();
        img.onload = () => console.log('✅ 이미지 로드 성공');
        img.onerror = () => console.error('❌ 이미지 로드 실패');
        img.src = data.url;
    }
});

// 2. DAG 이미지 컨테이너 직접 조작
const container = document.getElementById('dagImageContainer');
container.innerHTML = `
    <img src="/dag_images/dag_7918d9d4d4c330a6d67eec805db2f25918f6da467b9c0f04752a6d08c07595be.png" 
         style="max-width: 100%; height: auto;" 
         onload="console.log('직접 삽입 이미지 로드 성공')"
         onerror="console.error('직접 삽입 이미지 로드 실패')">
`;
```

## 🛠️ 즉시 해결 방법

만약 위의 모든 단계가 정상이라면, 가장 간단한 해결책:

1. **브라우저 캐시 강제 새로고침**: Ctrl+Shift+R (또는 Cmd+Shift+R)
2. **다른 브라우저에서 테스트**: Chrome, Firefox, Safari
3. **프라이빗/시크릿 모드에서 테스트**

## 📊 현재 확인된 상태

✅ **서버 실행**: 데모 서버가 포트 8080에서 실행 중
✅ **이미지 파일 존재**: dag_7918d9d4d4c330a6d67eec805db2f25918f6da467b9c0f04752a6d08c07595be.png (151KB)
✅ **API 응답**: 해시 계산 API가 정상 작동
✅ **이미지 접근**: HTTP 200 응답으로 이미지 파일 접근 가능

## 🎯 다음 단계

1. 브라우저에서 http://localhost:8080/test_image.html 접속
2. 개발자 도구에서 오류 메시지 확인
3. 문제가 지속되면 스크린샷과 콘솔 오류 메시지 공유
