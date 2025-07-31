#!/usr/bin/env python3
"""
Max Workers 반응 속도 테스트
===========================

다양한 max_workers 설정에 따른 배치 처리 성능을 빠르게 측정합니다.
"""

import requests
import json
import time
import subprocess
import signal
import os
from typing import List, Dict

API_URL = "http://localhost:8000"
SERVER_PROCESS = None

def start_api_server():
    """API 서버 시작"""
    global SERVER_PROCESS
    
    print("🚀 API 서버 시작 중...")
    
    try:
        # 기존 프로세스 종료
        subprocess.run(["pkill", "-f", "python api.py"], capture_output=True)
        time.sleep(2)
        
        # 새 서버 시작
        SERVER_PROCESS = subprocess.Popen(
            ["python", "api.py", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 서버 시작 대기
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{API_URL}/health", timeout=2)
                if response.status_code == 200:
                    print(f"✅ API 서버 시작 완료 ({attempt + 1}번째 시도)")
                    return True
            except:
                pass
            
            print(f"⏳ 서버 시작 대기 중... ({attempt + 1}/{max_attempts})")
            time.sleep(2)
        
        print("❌ 서버 시작 실패")
        return False
        
    except Exception as e:
        print(f"❌ 서버 시작 중 오류: {e}")
        return False

def stop_api_server():
    """API 서버 종료"""
    global SERVER_PROCESS
    
    if SERVER_PROCESS:
        SERVER_PROCESS.terminate()
        SERVER_PROCESS.wait()
    
    # 추가 정리
    subprocess.run(["pkill", "-f", "python api.py"], capture_output=True)
    print("🛑 API 서버 종료됨")

def test_worker_performance():
    """워커 수별 성능 테스트"""
    
    print("\n" + "="*60)
    print("Max Workers 성능 테스트")
    print("="*60)
    
    # 테스트 메시지 (8개 - 적당한 크기)
    messages = [
        "[SK텔레콤] 5G 슈퍼플랜 특가 이벤트",
        "[SK텔레콤] T멤버십 혜택 안내",
        "[SK텔레콤] 갤럭시 S25 사전예약",
        "[SK텔레콤] 넷플릭스 프로모션",
        "[SK텔레콤] 0 day 특별 혜택",
        "[SK텔레콤] 대리점 방문 이벤트",
        "[SK텔레콤] T우주 OTT 서비스",
        "[SK텔레콤] 아이폰 15 Pro 출시"
    ]
    
    # 테스트할 워커 수들
    worker_counts = [1, 2, 4, 6, 8]
    results = []
    
    print(f"📊 테스트 조건:")
    print(f"   메시지 수: {len(messages)}개")
    print(f"   테스트 워커 수: {worker_counts}")
    print(f"   반복 횟수: 각 워커 수당 1회")
    
    for worker_count in worker_counts:
        print(f"\n🔧 워커 수 {worker_count}개 테스트...")
        
        request_data = {
            "messages": messages,
            "llm_model": "gemma",
            "product_info_extraction_mode": "nlp",
            "entity_matching_mode": "logic",
            "max_workers": worker_count,
            "auto_worker_scaling": False
        }
        
        # 클라이언트 측 시간 측정
        client_start = time.time()
        
        try:
            response = requests.post(
                f"{API_URL}/extract/batch",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=120  # 2분 타임아웃
            )
            
            client_end = time.time()
            client_time = client_end - client_start
            
            if response.status_code == 200:
                result = response.json()
                
                # 결과 분석
                summary = result['summary']
                metadata = result['metadata']
                parallel_info = metadata.get('parallel_processing', {})
                
                test_result = {
                    'worker_count': worker_count,
                    'actual_workers': parallel_info.get('max_workers', worker_count),
                    'server_time': metadata.get('processing_time_seconds', 0),
                    'client_time': client_time,
                    'messages_per_second': parallel_info.get('messages_per_second', 0),
                    'avg_time_per_message': parallel_info.get('avg_time_per_message', 0),
                    'estimated_speedup': parallel_info.get('estimated_speedup', 0),
                    'success_count': summary.get('successful', 0),
                    'total_count': summary.get('total_messages', 0),
                    'success_rate': summary.get('successful', 0) / summary.get('total_messages', 1)
                }
                
                results.append(test_result)
                
                print(f"   ✅ 완료:")
                print(f"      실제 워커 수: {test_result['actual_workers']}")
                print(f"      서버 처리 시간: {test_result['server_time']:.3f}초")
                print(f"      클라이언트 시간: {test_result['client_time']:.3f}초")
                print(f"      처리 속도: {test_result['messages_per_second']:.2f} msg/sec")
                print(f"      성공률: {test_result['success_rate']:.1%}")
                
            else:
                print(f"   ❌ API 호출 실패: {response.status_code}")
                print(f"      응답: {response.text[:200]}...")
                
        except requests.exceptions.Timeout:
            print(f"   ❌ 타임아웃 (2분 초과)")
        except Exception as e:
            print(f"   ❌ 오류: {e}")
        
        # 테스트 간 대기 (서버 부하 방지)
        time.sleep(3)
    
    return results

def analyze_results(results: List[Dict]):
    """결과 분석 및 출력"""
    
    if not results:
        print("❌ 분석할 결과가 없습니다.")
        return
    
    print(f"\n" + "="*80)
    print("📊 Max Workers 성능 분석 결과")
    print("="*80)
    
    # 테이블 헤더
    print(f"{'워커수':<6} {'실제워커':<8} {'서버시간':<10} {'클라이언트시간':<12} {'속도(msg/s)':<12} {'성공률':<8}")
    print("-" * 70)
    
    # 결과 출력
    for r in results:
        print(f"{r['worker_count']:<6} {r['actual_workers']:<8} {r['server_time']:<10.3f} "
              f"{r['client_time']:<12.3f} {r['messages_per_second']:<12.2f} {r['success_rate']:<8.1%}")
    
    # 성능 분석
    print(f"\n🔍 성능 분석:")
    
    # 최고 속도
    best_speed = max(results, key=lambda x: x['messages_per_second'])
    print(f"   최고 처리 속도: 워커 {best_speed['worker_count']}개 ({best_speed['messages_per_second']:.2f} msg/sec)")
    
    # 최단 시간
    best_time = min(results, key=lambda x: x['server_time'])
    print(f"   최단 처리 시간: 워커 {best_time['worker_count']}개 ({best_time['server_time']:.3f}초)")
    
    # 효율성 분석 (워커 수 대비 성능)
    efficiency_scores = []
    for r in results:
        if r['worker_count'] > 0:
            efficiency = r['messages_per_second'] / r['worker_count']
            efficiency_scores.append((r['worker_count'], efficiency))
    
    if efficiency_scores:
        best_efficiency = max(efficiency_scores, key=lambda x: x[1])
        print(f"   최고 효율성: 워커 {best_efficiency[0]}개 (워커당 {best_efficiency[1]:.2f} msg/sec)")
    
    # 스케일링 효과 분석
    print(f"\n📈 스케일링 효과:")
    baseline = results[0] if results else None
    if baseline:
        for r in results[1:]:
            speedup = r['messages_per_second'] / baseline['messages_per_second'] if baseline['messages_per_second'] > 0 else 0
            theoretical_speedup = r['worker_count'] / baseline['worker_count']
            efficiency = (speedup / theoretical_speedup * 100) if theoretical_speedup > 0 else 0
            
            print(f"   워커 {baseline['worker_count']}→{r['worker_count']}: "
                  f"{speedup:.2f}x 향상 (이론값: {theoretical_speedup:.2f}x, 효율: {efficiency:.1f}%)")
    
    # 권장사항
    print(f"\n💡 권장사항:")
    if len(results) >= 2:
        # 성능 향상이 둔화되는 지점 찾기
        diminishing_point = None
        for i in range(1, len(results)):
            current_improvement = (results[i]['messages_per_second'] - results[i-1]['messages_per_second']) / results[i-1]['messages_per_second'] if results[i-1]['messages_per_second'] > 0 else 0
            if current_improvement < 0.1:  # 10% 미만 향상
                diminishing_point = results[i]['worker_count']
                break
        
        if diminishing_point:
            print(f"   워커 {diminishing_point}개부터 성능 향상 둔화")
            print(f"   권장 워커 수: {diminishing_point - 1}개")
        else:
            print(f"   현재 테스트 범위에서는 워커 수 증가에 따른 지속적인 성능 향상")
            print(f"   권장 워커 수: {best_speed['worker_count']}개 (최고 속도 기준)")
    
    # CPU 코어 수와 비교
    cpu_cores = os.cpu_count() or 4
    print(f"   시스템 CPU 코어 수: {cpu_cores}개")
    if best_speed['worker_count'] <= cpu_cores:
        print(f"   최적 워커 수가 CPU 코어 수 이하로 적절함")
    else:
        print(f"   최적 워커 수가 CPU 코어 수를 초과 (I/O 대기 시간이 많은 작업)")

def main():
    """메인 실행 함수"""
    
    print("🧪 Max Workers 반응 속도 테스트")
    print("="*50)
    
    try:
        # API 서버 시작
        if not start_api_server():
            print("❌ API 서버 시작 실패. 테스트를 중단합니다.")
            return
        
        # 성능 테스트 실행
        results = test_worker_performance()
        
        # 결과 분석
        analyze_results(results)
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 테스트가 중단되었습니다.")
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
    finally:
        # 서버 정리
        stop_api_server()
        print("\n🎉 테스트 완료!")

if __name__ == "__main__":
    main() 