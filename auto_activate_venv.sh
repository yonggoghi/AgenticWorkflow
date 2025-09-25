#!/bin/bash

# AgenticWorkflow 프로젝트 자동 가상환경 활성화 스크립트
# ~/.zshrc 또는 ~/.bash_profile에 추가하세요:
# source /Users/yongwook/workspace/AgenticWorkflow/auto_activate_venv.sh

# 현재 디렉토리가 프로젝트 디렉토리인지 확인
if [[ "$PWD" == "/Users/yongwook/workspace/AgenticWorkflow"* ]]; then
    # 가상환경이 이미 활성화되어 있는지 확인
    if [[ "$VIRTUAL_ENV" != "/Users/yongwook/workspace/AgenticWorkflow/venv" ]]; then
        # venv 디렉토리가 존재하는지 확인
        if [[ -f "/Users/yongwook/workspace/AgenticWorkflow/venv/bin/activate" ]]; then
            echo "🐍 AgenticWorkflow 가상환경 활성화 중..."
            source /Users/yongwook/workspace/AgenticWorkflow/venv/bin/activate
        fi
    fi
fi
