#!/bin/bash

# MMS Extractor Streamlit Demo 실행 스크립트

# 기본값 설정
API_PORT=8000
DEMO_PORT=8082

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-port)
            API_PORT="$2"
            shift 2
            ;;
        --demo-port)
            DEMO_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "사용법: $0 [옵션]"
            echo ""
            echo "옵션:"
            echo "  --api-port PORT     API 서버 포트 (기본값: 8000)"
            echo "  --demo-port PORT    Demo 서버 포트 (기본값: 8082)"
            echo "  -h, --help          이 도움말 표시"
            echo ""
            echo "예시:"
            echo "  $0 --api-port 8000 --demo-port 8082"
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "도움말을 보려면 $0 --help를 사용하세요."
            exit 1
            ;;
    esac
done

echo "🚀 MMS Extractor Streamlit Demo 시작"
echo "📡 API 서버 포트: $API_PORT"
echo "🌐 Demo 서버 포트: $DEMO_PORT"
echo ""

# Streamlit 실행 (우리가 원하는 인수들을 스크립트에 전달)
streamlit run demo_streamlit.py --server.port 8501 -- --api-port $API_PORT --demo-port $DEMO_PORT
