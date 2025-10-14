#!/bin/bash

echo "==================================================="
echo "MMS 추출기 API 성능 테스트 스크립트"
echo "==================================================="

# 현재 디렉토리로 이동
cd "$(dirname "$0")"

# 1. API 서버가 실행 중인지 확인
echo "1. API 서버 상태 확인 중..."
if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
    echo "✅ API 서버가 이미 실행 중입니다."
    SERVER_RUNNING=true
else
    echo "❌ API 서버가 실행되지 않았습니다."
    SERVER_RUNNING=false
fi

# 2. 서버가 실행되지 않은 경우 시작 옵션 제공
if [ "$SERVER_RUNNING" = false ]; then
    echo ""
    echo "API 서버를 시작하시겠습니까? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🚀 API 서버 시작 중..."
        echo "   로그를 보려면 별도 터미널에서 다음 명령어를 실행하세요:"
        echo "   tail -f api_server.log"
        echo ""
        
        # 백그라운드에서 API 서버 시작 (로그 파일로 출력)
        nohup python api.py --host 0.0.0.0 --port 8080 > api_server.log 2>&1 &
        SERVER_PID=$!
        echo "서버 PID: $SERVER_PID"
        
        # 서버 시작 대기
        echo "서버 시작 대기 중..."
        for i in {1..30}; do
            if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
                echo "✅ API 서버가 시작되었습니다!"
                break
            fi
            echo -n "."
            sleep 2
        done
        
        if ! curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
            echo ""
            echo "❌ 서버 시작에 실패했습니다. api_server.log를 확인하세요."
            exit 1
        fi
        
        # 서버 종료 트랩 설정
        trap "echo '서버 종료 중...'; kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT
    else
        echo "수동으로 API 서버를 시작한 후 다시 실행하세요:"
        echo "  python api.py --host 0.0.0.0 --port 8080"
        exit 1
    fi
fi

# 3. 성능 테스트 실행
echo ""
echo "3. 성능 테스트 실행 중..."
echo "==================================================="

python test_api_performance.py

TEST_RESULT=$?

# 4. 결과 정리
echo ""
echo "==================================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ 성능 테스트가 성공적으로 완료되었습니다!"
else
    echo "❌ 성능 테스트 중 오류가 발생했습니다."
fi

# 5. 서버를 우리가 시작한 경우 종료 여부 묻기
if [ "$SERVER_RUNNING" = false ] && [ ! -z "$SERVER_PID" ]; then
    echo ""
    echo "테스트용으로 시작한 API 서버를 종료하시겠습니까? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🛑 API 서버 종료 중..."
        kill $SERVER_PID 2>/dev/null
        echo "✅ API 서버가 종료되었습니다."
    else
        echo "ℹ️  API 서버는 계속 실행됩니다. (PID: $SERVER_PID)"
        echo "   수동 종료: kill $SERVER_PID"
        # 트랩 해제 (서버를 계속 실행하기 위해)
        trap - INT TERM EXIT
    fi
fi

echo "==================================================="
echo "테스트 완료!" 