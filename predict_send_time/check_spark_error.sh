#!/bin/bash
#
# Spark Shell 에러 확인 스크립트
# Usage: ./check_spark_error.sh [application_id]
#

if [ -z "$1" ]; then
  echo "최근 실행된 predict_ost 애플리케이션 찾는 중..."
  APP_ID=$(yarn application -list -appStates ALL 2>/dev/null | grep "predict_ost" | head -1 | awk '{print $1}')
  
  if [ -z "$APP_ID" ]; then
    echo "Error: Application ID를 찾을 수 없습니다."
    echo "Usage: $0 <application_id>"
    echo ""
    echo "실행 중인 애플리케이션:"
    yarn application -list -appStates RUNNING 2>/dev/null | grep "predict_ost"
    echo ""
    echo "최근 실패한 애플리케이션:"
    yarn application -list -appStates FAILED 2>/dev/null | grep "predict_ost" | head -5
    exit 1
  fi
  
  echo "Found Application ID: $APP_ID"
else
  APP_ID=$1
fi

echo "========================================"
echo "Checking YARN logs for: $APP_ID"
echo "========================================"
echo ""

echo "1. Container Killed Errors:"
echo "----------------------------------------"
yarn logs -applicationId $APP_ID 2>/dev/null | grep -i "container.*killed" -A 5 | head -30
echo ""

echo "2. Memory Errors:"
echo "----------------------------------------"
yarn logs -applicationId $APP_ID 2>/dev/null | grep -i "exceeding memory\|outofmemory" -A 5 | head -30
echo ""

echo "3. XGBoost Errors:"
echo "----------------------------------------"
yarn logs -applicationId $APP_ID 2>/dev/null | grep -i "xgboost.*error\|tracker.*fail\|barrier" -A 10 | head -30
echo ""

echo "4. Barrier Mode / Dynamic Allocation Errors:"
echo "----------------------------------------"
yarn logs -applicationId $APP_ID 2>/dev/null | grep -i "barrier.*dynamic\|dynamicallocation" -A 5 | head -20
echo ""

echo "5. General Exceptions (stderr):"
echo "----------------------------------------"
yarn logs -applicationId $APP_ID -log_files stderr 2>/dev/null | grep -i "exception" | head -15
echo ""

echo "========================================"
echo "Complete log saved to: /tmp/spark_${APP_ID}_full.log"
echo "Stderr saved to: /tmp/spark_${APP_ID}_stderr.log"
echo "========================================"

# Save full logs
yarn logs -applicationId $APP_ID > /tmp/spark_${APP_ID}_full.log 2>&1
yarn logs -applicationId $APP_ID -log_files stderr > /tmp/spark_${APP_ID}_stderr.log 2>&1

echo ""
echo "To view full logs:"
echo "  cat /tmp/spark_${APP_ID}_full.log"
echo "  cat /tmp/spark_${APP_ID}_stderr.log"
