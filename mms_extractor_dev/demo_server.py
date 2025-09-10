#!/usr/bin/env python3
"""
MMS Extractor Demo Server
========================

이 모듈은 MMS 추출기 API와 함께 웹 데모 인터페이스를 제공합니다.
정적 파일 서빙과 DAG 이미지 접근을 위한 추가 엔드포인트를 포함합니다.

사용법:
    python demo_server.py --port 8080
"""

import os
import sys
import argparse
import mimetypes
import hashlib
from pathlib import Path
from flask import Flask, send_from_directory, jsonify, send_file, request
from flask_cors import CORS

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # CORS 활성화

# 정적 파일 경로 설정
STATIC_DIR = current_dir
DAG_IMAGES_DIR = current_dir / 'dag_images'

@app.route('/')
def index():
    """메인 데모 페이지 제공"""
    return send_from_directory(STATIC_DIR, 'demo.html')

@app.route('/demo')
def demo():
    """데모 페이지 제공 (별칭)"""
    return send_from_directory(STATIC_DIR, 'demo.html')

@app.route('/test_image.html')
def test_image():
    """이미지 테스트 페이지 제공"""
    return send_from_directory(STATIC_DIR, 'test_image.html')

@app.route('/simple_test.html')
def simple_test():
    """간단한 이미지 테스트 페이지 제공"""
    return send_from_directory(STATIC_DIR, 'simple_test.html')

@app.route('/dag_images/<path:filename>')
def dag_image(filename):
    """DAG 이미지 파일 제공"""
    try:
        if DAG_IMAGES_DIR.exists():
            return send_from_directory(DAG_IMAGES_DIR, filename)
        else:
            return jsonify({"error": "DAG images directory not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/api/dag-images')
def list_dag_images():
    """사용 가능한 DAG 이미지 목록 반환"""
    try:
        if not DAG_IMAGES_DIR.exists():
            return jsonify({"images": []})
        
        images = []
        for file_path in DAG_IMAGES_DIR.glob('dag_*.png'):
            file_stat = file_path.stat()
            images.append({
                "filename": file_path.name,
                "url": f"/dag_images/{file_path.name}",
                "size": file_stat.st_size,
                "modified": file_stat.st_mtime
            })
        
        # 수정 시간 기준으로 최신순 정렬
        images.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({"images": images})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/latest-dag-image')
def latest_dag_image():
    """가장 최근에 생성된 DAG 이미지 반환"""
    try:
        if not DAG_IMAGES_DIR.exists():
            return jsonify({"error": "DAG images directory not found"}), 404
        
        # 가장 최근 파일 찾기
        dag_files = list(DAG_IMAGES_DIR.glob('dag_*.png'))
        if not dag_files:
            return jsonify({"error": "No DAG images found"}), 404
        
        latest_file = max(dag_files, key=lambda x: x.stat().st_mtime)
        
        return jsonify({
            "filename": latest_file.name,
            "url": f"/dag_images/{latest_file.name}",
            "size": latest_file.stat().st_size,
            "modified": latest_file.stat().st_mtime
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dag-image/<hash_value>')
def get_dag_image_by_hash(hash_value):
    """특정 해시값으로 DAG 이미지 찾기"""
    try:
        if not DAG_IMAGES_DIR.exists():
            return jsonify({"error": "DAG images directory not found"}), 404
        
        # 특정 해시의 파일 찾기
        expected_filename = f"dag_{hash_value}.png"
        expected_path = DAG_IMAGES_DIR / expected_filename
        
        if expected_path.exists():
            return jsonify({
                "filename": expected_filename,
                "url": f"/dag_images/{expected_filename}",
                "size": expected_path.stat().st_size,
                "modified": expected_path.stat().st_mtime,
                "hash": hash_value
            })
        else:
            return jsonify({"error": f"DAG image not found for hash: {hash_value}"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def sha256_hash(text):
    """
    utils.py와 동일한 해시 함수
    텍스트의 SHA256 해시값 생성
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

@app.route('/api/calculate-hash', methods=['POST'])
def calculate_hash():
    """메시지 텍스트의 해시값을 계산하여 반환"""
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "메시지가 필요합니다"}), 400
        
        message = data['message']
        hash_value = sha256_hash(message)
        expected_filename = f"dag_{hash_value}.png"
        expected_path = DAG_IMAGES_DIR / expected_filename
        
        result = {
            "message": message,
            "hash": hash_value,
            "expected_filename": expected_filename,
            "image_exists": expected_path.exists()
        }
        
        if expected_path.exists():
            result.update({
                "url": f"/dag_images/{expected_filename}",
                "size": expected_path.stat().st_size,
                "modified": expected_path.stat().st_mtime
            })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """404 에러 핸들러"""
    return jsonify({"error": "리소스를 찾을 수 없습니다"}), 404

@app.errorhandler(500)
def internal_error(error):
    """500 에러 핸들러"""
    return jsonify({"error": "서버 내부 오류가 발생했습니다"}), 500

def main():
    """메인 함수 - CLI 진입점"""
    parser = argparse.ArgumentParser(description='MMS 추출기 데모 서버')
    parser.add_argument('--host', default='0.0.0.0', help='바인딩할 호스트 주소')
    parser.add_argument('--port', type=int, default=8080, help='바인딩할 포트 번호')
    parser.add_argument('--debug', action='store_true', help='디버그 모드 활성화')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 MMS Extractor Demo Server")
    print("=" * 60)
    print(f"📱 데모 페이지: http://{args.host}:{args.port}")
    print(f"🖼️  DAG 이미지: http://{args.host}:{args.port}/dag_images/")
    print(f"📊 이미지 목록: http://{args.host}:{args.port}/api/dag-images")
    print("=" * 60)
    print()
    print("💡 사용 방법:")
    print("1. 브라우저에서 데모 페이지에 접속하세요")
    print("2. MMS 추출기 API 서버(포트 8000)가 실행 중인지 확인하세요")
    print("3. 샘플 메시지를 클릭하거나 직접 입력하여 테스트하세요")
    print("4. DAG 추출을 활성화하면 관계 그래프를 볼 수 있습니다")
    print()
    print("⚡ 서버 시작 중...")
    
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
