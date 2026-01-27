#!/usr/bin/env python3
"""
파티션 통합 기능 테스트 스크립트

작은 샘플 데이터로 파티션 통합 기능을 테스트합니다.
"""

import os
import shutil
import tempfile


def create_test_data():
    """테스트용 파티션 데이터 생성"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("Error: pyarrow가 설치되어 있지 않습니다.")
        print("설치: pip install pyarrow")
        return None
    
    # 임시 디렉토리 생성
    test_dir = tempfile.mkdtemp(prefix="test_partition_")
    print(f"테스트 디렉토리 생성: {test_dir}")
    
    # 샘플 데이터 생성
    data_2024_01 = pa.table({
        'id': [1, 2, 3],
        'value': [100, 200, 300],
        'name': ['Alice', 'Bob', 'Charlie']
    })
    
    data_2024_02 = pa.table({
        'id': [4, 5, 6],
        'value': [400, 500, 600],
        'name': ['David', 'Eve', 'Frank']
    })
    
    data_2025_01 = pa.table({
        'id': [7, 8, 9],
        'value': [700, 800, 900],
        'name': ['Grace', 'Henry', 'Iris']
    })
    
    # 파티션 구조로 저장
    partitions = [
        ('year=2024/month=01', data_2024_01),
        ('year=2024/month=02', data_2024_02),
        ('year=2025/month=01', data_2025_01)
    ]
    
    for partition_path, data in partitions:
        full_path = os.path.join(test_dir, partition_path)
        os.makedirs(full_path, exist_ok=True)
        
        # 각 파티션에 2개의 파일 생성 (실제 상황 시뮬레이션)
        for i in range(2):
            file_path = os.path.join(full_path, f'part-0000{i}.parquet')
            pq.write_table(data, file_path)
            print(f"생성: {partition_path}/part-0000{i}.parquet")
    
    return test_dir


def test_merge():
    """파티션 통합 테스트"""
    print("\n=== 파티션 통합 기능 테스트 ===\n")
    
    # 테스트 데이터 생성
    test_dir = create_test_data()
    if not test_dir:
        return False
    
    try:
        # hdfs_transfer 모듈의 함수 import
        from hdfs_transfer import merge_partitioned_parquet
        import pyarrow.parquet as pq
        
        # 출력 파일 경로
        output_file = os.path.join(test_dir, 'merged_test.parquet')
        
        print(f"\n테스트 시작:")
        print(f"  입력: {test_dir}")
        print(f"  출력: {output_file}\n")
        
        # 파티션 통합 실행
        success = merge_partitioned_parquet(
            source_dir=test_dir,
            output_file=output_file,
            batch_size=2,  # 작은 배치 크기로 테스트
            compression='snappy',
            verbose=True
        )
        
        if not success:
            print("\n❌ 테스트 실패: 파티션 통합 실패")
            return False
        
        # 결과 검증
        print("\n=== 결과 검증 ===")
        result_table = pq.read_table(output_file)
        
        print(f"\n통합된 데이터:")
        print(f"  총 row 수: {len(result_table)}")
        print(f"  컬럼: {result_table.column_names}")
        print(f"\n데이터 샘플:")
        print(result_table.to_pandas().head(10))
        
        # 파티션 컬럼 확인
        expected_columns = {'id', 'value', 'name', 'year', 'month'}
        actual_columns = set(result_table.column_names)
        
        if expected_columns == actual_columns:
            print("\n✅ 테스트 성공: 파티션 컬럼이 올바르게 추가되었습니다!")
            
            # 각 파티션의 데이터 수 확인
            df = result_table.to_pandas()
            print("\n파티션별 row 수:")
            print(df.groupby(['year', 'month']).size())
            
            return True
        else:
            print(f"\n❌ 테스트 실패: 컬럼이 일치하지 않습니다")
            print(f"  예상: {expected_columns}")
            print(f"  실제: {actual_columns}")
            return False
            
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 테스트 디렉토리 정리
        print(f"\n테스트 디렉토리 정리: {test_dir}")
        try:
            shutil.rmtree(test_dir)
            print("정리 완료")
        except Exception as e:
            print(f"정리 실패: {e}")


def test_compression_algorithms():
    """다양한 압축 알고리즘 테스트"""
    print("\n=== 압축 알고리즘 테스트 ===\n")
    
    test_dir = create_test_data()
    if not test_dir:
        return
    
    try:
        from hdfs_transfer import merge_partitioned_parquet
        
        compressions = ['snappy', 'gzip', 'zstd', 'none']
        results = {}
        
        for comp in compressions:
            print(f"\n테스트 압축: {comp}")
            output_file = os.path.join(test_dir, f'merged_{comp}.parquet')
            
            success = merge_partitioned_parquet(
                source_dir=test_dir,
                output_file=output_file,
                batch_size=10,
                compression=comp,
                verbose=False
            )
            
            if success:
                file_size = os.path.getsize(output_file)
                results[comp] = file_size
                print(f"  ✅ 성공 - 파일 크기: {file_size:,} bytes")
            else:
                print(f"  ❌ 실패")
        
        if results:
            print("\n=== 압축 알고리즘 비교 ===")
            for comp, size in sorted(results.items(), key=lambda x: x[1]):
                print(f"  {comp:10s}: {size:,} bytes")
                
    finally:
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    import sys
    
    # 기본 테스트
    success = test_merge()
    
    if success:
        # 압축 알고리즘 테스트 (선택적)
        if '--full' in sys.argv:
            test_compression_algorithms()
        
        print("\n" + "="*50)
        print("모든 테스트 완료! 파티션 통합 기능이 정상 작동합니다.")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("테스트 실패! 위의 에러 메시지를 확인하세요.")
        print("="*50)
        sys.exit(1)
