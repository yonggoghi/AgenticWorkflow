#!/usr/bin/env python3
"""
HDFS 데이터 다운로드 및 원격 서버 전송 스크립트
"""

import os
import subprocess
import sys
import argparse
import shutil
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import glob
import re


def run_command(command, shell=True, check=True):
    """명령어를 실행하고 결과를 반환"""
    try:
        result = subprocess.run(
            command,
            shell=shell,
            check=check,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False


def check_sshpass():
    """sshpass 설치 확인 및 설치"""
    print("# sshpass 설치 확인")
    if not run_command("command -v sshpass &> /dev/null", check=False):
        print("Installing sshpass...")
        # RHEL/CentOS
        if run_command("command -v yum &> /dev/null", check=False):
            run_command("sudo yum install -y sshpass")
        # Ubuntu/Debian
        elif run_command("command -v apt-get &> /dev/null", check=False):
            run_command("sudo apt-get install -y sshpass")
        else:
            print("Please install sshpass manually")
            sys.exit(1)


def prepare_local_directory(local_tmp_path, dir_name, archive_name, cleanup=False):
    """
    로컬 임시 디렉토리 준비
    
    Args:
        cleanup: True면 기존 파일 삭제, False면 유지 (기본)
    """
    print("\n# 로컬 임시 디렉토리 준비")
    target_dir = os.path.join(local_tmp_path, dir_name)
    tar_file = os.path.join(local_tmp_path, archive_name)
    
    if cleanup:
        # --cleanup 사용: 기존 파일 삭제
        print("모드: 기존 파일 삭제 (--cleanup)")
        
        # 기존 디렉토리가 있으면 삭제
        if os.path.exists(target_dir):
            print(f"Removing existing local directory: {target_dir}")
            try:
                shutil.rmtree(target_dir)
                print(f"Successfully removed: {target_dir}")
            except Exception as e:
                print(f"Error removing directory: {e}")
                return False
        
        # 기존 tar.gz 파일 삭제
        if os.path.exists(tar_file):
            print(f"Removing existing tar file: {tar_file}")
            try:
                os.remove(tar_file)
                print(f"Successfully removed: {tar_file}")
            except Exception as e:
                print(f"Error removing tar file: {e}")
                return False
    else:
        # 기본 모드: 기존 파일 유지
        print("모드: 기존 파일 유지 (기본)")
        if os.path.exists(target_dir):
            print(f"기존 디렉토리 유지: {target_dir}")
        if os.path.exists(tar_file):
            print(f"기존 tar 파일 유지: {tar_file}")
    
    # 부모 디렉토리가 없으면 생성
    if not os.path.exists(local_tmp_path):
        print(f"Creating local tmp directory: {local_tmp_path}")
        try:
            os.makedirs(local_tmp_path, exist_ok=True)
            print(f"Successfully created: {local_tmp_path}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return False
    
    return True


def remove_remote_files(remote_user, remote_password, remote_ip, remote_path, dir_name, output_filename, archive_name, merge_partitions=False, extract_remote=False):
    """
    원격 서버의 기존 파일 삭제
    
    Args:
        merge_partitions: True면 OUTPUT_FILENAME 고려, False면 ARCHIVE_NAME 고려
        extract_remote: True면 압축 해제된 파일 삭제, False면 tar.gz만 삭제
    """
    print("\n# 원격 서버 기존 파일 삭제")
    
    # EOF 파일 경로 (공통)
    base_name = archive_name.replace('.parquet', '').replace('.tar.gz', '')
    eof_file_path = f"{remote_path}/{base_name}.eof"
    
    if merge_partitions:
        # --merge-partitions 사용: OUTPUT_FILENAME과 EOF 삭제
        print("모드: 파티션 통합 모드")
        output_file_path = f"{remote_path}/{output_filename}"  # dir_name 제거
        
        print("Removing files:")
        print(f"  - {output_file_path} (parquet)")
        print(f"  - {eof_file_path} (eof)")
        
        rm_cmd = f'sshpass -p "{remote_password}" ssh -o StrictHostKeyChecking=no {remote_user}@{remote_ip} "rm -f {output_file_path} {eof_file_path}"'
    else:
        # --merge-partitions 미사용
        tar_gz_path = f"{remote_path}/{archive_name}"
        
        if extract_remote:
            # --extract-remote 사용: 압축 해제된 디렉토리, tar.gz, EOF 삭제
            print("모드: 파티션 구조 유지 + 압축 해제")
            
            extracted_dir = f"{remote_path}/{dir_name}"
            
            print("Removing:")
            print(f"  - {extracted_dir}/ (압축 해제된 디렉토리)")
            print(f"  - {tar_gz_path} (tar.gz)")
            print(f"  - {eof_file_path} (eof)")
            
            # 압축 해제된 디렉토리와 파일들 삭제
            rm_cmd = f'sshpass -p "{remote_password}" ssh -o StrictHostKeyChecking=no {remote_user}@{remote_ip} "rm -rf {extracted_dir} && rm -f {tar_gz_path} {eof_file_path}"'
        else:
            # --extract-remote 미사용: tar.gz, EOF만 삭제
            print("모드: 파티션 구조 유지 + 압축 파일 유지")
            
            print("Removing:")
            print(f"  - {tar_gz_path} (tar.gz)")
            print(f"  - {eof_file_path} (eof)")
            
            # tar.gz와 EOF만 삭제
            rm_cmd = f'sshpass -p "{remote_password}" ssh -o StrictHostKeyChecking=no {remote_user}@{remote_ip} "rm -f {tar_gz_path} {eof_file_path}"'
    
    return run_command(rm_cmd)


def download_from_hdfs(hdfs_path, local_tmp_path):
    """HDFS에서 다운로드"""
    print("\n# HDFS에서 다운로드")
    print("Downloading from HDFS...")
    hdfs_cmd = f"hdfs dfs -get -f {hdfs_path} {local_tmp_path}"
    if not run_command(hdfs_cmd):
        print("Failed to download from HDFS")
        return False
    return True


def compress_data(local_tmp_path, dir_name, archive_name):
    """데이터 압축 (디렉토리 포함)"""
    print("\n# 압축")
    print(f"Compressing to {archive_name}...")
    os.chdir(local_tmp_path)
    # 디렉토리 자체를 압축
    # 압축 해제 시 디렉토리가 생성되고 그 안에 파일들이 풀림
    tar_cmd = f"tar -czf {archive_name} {dir_name}"
    if not run_command(tar_cmd):
        print("Failed to compress")
        return False
    return True


def transfer_data(remote_user, remote_password, remote_ip, remote_path, archive_name):
    """데이터 전송"""
    print("\n# 전송")
    print(f"Transferring {archive_name}...")
    scp_cmd = f'sshpass -p "{remote_password}" scp -o StrictHostKeyChecking=no {archive_name} {remote_user}@{remote_ip}:{remote_path}/'
    if not run_command(scp_cmd):
        print("Failed to transfer")
        return False
    return True


def extract_remote(remote_user, remote_password, remote_ip, remote_path, archive_name):
    """원격 서버에서 압축 해제"""
    print("\n# 압축 해제")
    print("Extracting on remote...")
    extract_cmd = f'sshpass -p "{remote_password}" ssh -o StrictHostKeyChecking=no {remote_user}@{remote_ip} "cd {remote_path} && tar -xzf {archive_name} && rm {archive_name}"'
    if not run_command(extract_cmd):
        print("Failed to extract on remote")
        return False
    return True


def create_eof_file(remote_user, remote_password, remote_ip, remote_path, archive_name):
    """원격 서버에 .eof 파일 생성"""
    print("\n# EOF 파일 생성")
    
    # 압축 파일명에서 .parquet을 제거하고 .eof 추가
    # 예: mth_mms_rcv_ract_score_202601.parquet -> mth_mms_rcv_ract_score_202601.eof
    base_name = archive_name.replace('.parquet', '').replace('.tar.gz', '')
    eof_filename = f"{base_name}.eof"
    
    print(f"Creating EOF file: {eof_filename}")
    
    # 원격 서버에서 touch 명령으로 빈 .eof 파일 생성
    eof_cmd = f'sshpass -p "{remote_password}" ssh -o StrictHostKeyChecking=no {remote_user}@{remote_ip} "touch {remote_path}/{eof_filename}"'
    
    if not run_command(eof_cmd):
        print("Failed to create EOF file")
        return False
    
    print(f"Successfully created: {eof_filename}")
    return True


def cleanup(local_tmp_path, dir_name, archive_name):
    """임시 파일 정리"""
    print("\n# 정리")
    print("Cleaning up...")
    cleanup_files = [
        f"{local_tmp_path}/{dir_name}",
        f"{local_tmp_path}/{archive_name}"
    ]
    for file_path in cleanup_files:
        run_command(f"rm -rf {file_path}", check=False)


def extract_partition_values(file_path, partition_root):
    """
    파일 경로에서 파티션 값을 추출합니다.
    
    예: /path/to/data/year=2024/month=01/file.parquet
        -> {'year': '2024', 'month': '01'}
    """
    partition_info = {}
    relative_path = os.path.relpath(file_path, partition_root)
    path_parts = relative_path.split(os.sep)
    
    # Hive 스타일 파티션 패턴 (key=value)
    partition_pattern = re.compile(r'^([^=]+)=(.+)$')
    
    for part in path_parts[:-1]:  # 마지막은 파일명이므로 제외
        match = partition_pattern.match(part)
        if match:
            key, value = match.groups()
            partition_info[key] = value
    
    return partition_info


def merge_partitioned_parquet(
    source_dir,
    output_file,
    batch_size=100_000,
    compression='snappy',
    verbose=True
):
    """
    파티션된 parquet 파일들을 단일 parquet 파일로 통합합니다.
    PyArrow Streaming 방식을 사용하여 메모리 효율적으로 처리합니다.
    
    Args:
        source_dir: 파티션된 parquet 파일들이 있는 디렉토리
        output_file: 출력 parquet 파일 경로
        batch_size: 한 번에 읽을 row 개수 (메모리 사용량 조절)
        compression: 압축 알고리즘 (snappy, gzip, zstd, none)
        verbose: 진행 상황 출력 여부
    
    Returns:
        bool: 성공 여부
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("Error: pyarrow가 설치되어 있지 않습니다.")
        print("다음 명령으로 설치하세요: pip install pyarrow")
        return False
    
    print("\n# 파티션 통합 (PyArrow Streaming)")
    print(f"Source: {source_dir}")
    print(f"Output: {output_file}")
    print(f"Batch size: {batch_size:,} rows")
    print(f"Compression: {compression}")
    
    # 모든 parquet 파일 찾기
    parquet_files = glob.glob(
        os.path.join(source_dir, '**', '*.parquet'),
        recursive=True
    )
    
    # _SUCCESS 등의 메타 파일 제외
    parquet_files = [
        f for f in parquet_files 
        if not os.path.basename(f).startswith('_')
    ]
    
    if not parquet_files:
        print(f"Error: {source_dir}에서 parquet 파일을 찾을 수 없습니다.")
        return False
    
    print(f"Found {len(parquet_files)} parquet file(s)")
    
    # 첫 번째 파일로 스키마 확인
    first_table = pq.read_table(parquet_files[0])
    original_schema = first_table.schema
    
    # 파티션 컬럼 확인
    first_partition = extract_partition_values(parquet_files[0], source_dir)
    partition_columns = list(first_partition.keys())
    
    if partition_columns:
        print(f"Detected partition columns: {partition_columns}")
        
        # 새로운 스키마 생성 (기존 컬럼 + 파티션 컬럼)
        new_fields = list(original_schema)
        for col in partition_columns:
            if col not in [f.name for f in original_schema]:
                new_fields.append(pa.field(col, pa.string()))
        new_schema = pa.schema(new_fields)
    else:
        print("No partition columns detected (flat structure)")
        new_schema = original_schema
    
    # ParquetWriter 초기화
    writer = None
    total_rows = 0
    
    try:
        # 각 파일을 배치 단위로 읽고 쓰기
        for i, parquet_file in enumerate(parquet_files, 1):
            if verbose:
                print(f"Processing [{i}/{len(parquet_files)}]: {os.path.basename(parquet_file)}")
            
            # 파티션 값 추출
            partition_values = extract_partition_values(parquet_file, source_dir)
            
            # 파일을 배치 단위로 읽기
            parquet_file_reader = pq.ParquetFile(parquet_file)
            
            for batch in parquet_file_reader.iter_batches(batch_size=batch_size):
                # RecordBatch를 Table로 변환
                batch_table = pa.Table.from_batches([batch])
                
                # 파티션 컬럼 추가
                if partition_values:
                    for col_name, col_value in partition_values.items():
                        if col_name not in batch_table.column_names:
                            # 파티션 값으로 새로운 컬럼 추가
                            partition_array = pa.array(
                                [col_value] * len(batch_table),
                                type=pa.string()
                            )
                            batch_table = batch_table.append_column(
                                col_name,
                                partition_array
                            )
                
                # Writer 초기화 (첫 배치에서만)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_file,
                        batch_table.schema,
                        compression=compression
                    )
                
                # 배치 쓰기
                writer.write_table(batch_table)
                total_rows += len(batch_table)
                
                if verbose and total_rows % (batch_size * 10) == 0:
                    print(f"  Processed {total_rows:,} rows...")
        
        print(f"Successfully merged {len(parquet_files)} files")
        print(f"Total rows: {total_rows:,}")
        print(f"Output file: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if writer:
            writer.close()


def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(
        description='HDFS 데이터 다운로드 및 원격 서버 전송 스크립트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
파티션 통합 옵션:
  --merge-partitions를 사용하면 다운로드된 파티션 파일들을 단일 parquet 파일로 통합합니다.
  이 작업은 PyArrow Streaming 방식을 사용하여 메모리 효율적으로 처리됩니다.
  
  환경 변수 설정:
    OUTPUT_FILENAME: 통합된 parquet 파일명 (.env 파일에서 설정 가능)
    ARCHIVE_NAME: tar.gz 압축 파일명 (.env 파일에서 설정 가능)
  
  예제:
    # 기본 사용 (통합 없음, tar.gz 압축 파일만 전송, 로컬 파일 유지)
    python hdfs_transfer.py
    
    # tar.gz 압축 해제까지 수행
    python hdfs_transfer.py --extract-remote
    
    # 파티션 통합 활성화 (로컬 파일 자동 유지)
    python hdfs_transfer.py --merge-partitions
    
    # 로컬 파일 삭제 (cleanup)
    python hdfs_transfer.py --cleanup
    
    # 배치 크기와 압축 알고리즘 지정
    python hdfs_transfer.py --merge-partitions --batch-size 200000 --compression zstd
    
    # 출력 파일명 지정 (.env의 OUTPUT_FILENAME 대신 사용)
    python hdfs_transfer.py --merge-partitions --output-filename merged_data.parquet
        """
    )
    parser.add_argument(
        '--skip-remove',
        action='store_true',
        help='원격 서버의 기존 파일 삭제 단계를 건너뜁니다 (OUTPUT_FILENAME, EOF 파일 등)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='HDFS 다운로드 단계를 건너뜁니다 (로컬에 이미 파일이 있는 경우)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='로컬 임시 파일을 삭제합니다 (기본: 파일 유지)'
    )
    parser.add_argument(
        '--extract-remote',
        action='store_true',
        help='원격 서버에서 tar.gz 압축 해제를 수행합니다 (기본값: 압축 파일 그대로 유지)'
    )
    parser.add_argument(
        '--archive-name',
        help='압축 파일명 (예: mth_mms_rcv_ract_score_202601.tar.gz). 지정하지 않으면 ARCHIVE_NAME 환경변수 또는 data.tar.gz 사용'
    )
    parser.add_argument(
        '--env-file',
        default='.env',
        help='.env 파일 경로 (기본값: .env)'
    )
    
    # 파티션 통합 옵션
    parser.add_argument(
        '--merge-partitions',
        action='store_true',
        help='파티션된 parquet 파일들을 단일 파일로 통합합니다 (PyArrow 필요)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100_000,
        help='파티션 통합 시 배치 크기 (기본값: 100000). 메모리가 부족하면 줄이세요.'
    )
    parser.add_argument(
        '--compression',
        choices=['snappy', 'gzip', 'zstd', 'none'],
        default='snappy',
        help='출력 parquet 파일의 압축 알고리즘 (기본값: snappy)'
    )
    parser.add_argument(
        '--output-filename',
        help='통합된 parquet 파일명 (환경변수 OUTPUT_FILENAME 또는 기본값: mth_mms_rcv_ract_score_202601.parquet)'
    )
    
    args = parser.parse_args()
    
    # .env 파일 로드
    load_dotenv(args.env_file)
    
    # 환경 변수에서 설정 읽기
    HDFS_PATH = os.getenv('HDFS_PATH')
    REMOTE_USER = os.getenv('REMOTE_USER')
    REMOTE_PASSWORD = os.getenv('REMOTE_PASSWORD')
    REMOTE_IP = os.getenv('REMOTE_IP')
    REMOTE_PATH = os.getenv('REMOTE_PATH')
    LOCAL_TMP_PATH = os.getenv('LOCAL_TMP_PATH', '/home/skinet/myfiles/aos_ost/tmp/')
    
    # 압축 파일명 결정 (우선순위: 명령행 인자 > 환경변수 > 기본값)
    # 이것은 tar.gz 압축 파일명입니다
    if args.archive_name:
        ARCHIVE_NAME = args.archive_name
    else:
        ARCHIVE_NAME = os.getenv('ARCHIVE_NAME', 'data.tar.gz')
    
    # .tar.gz 확장자가 없으면 추가
    if not ARCHIVE_NAME.endswith('.tar.gz'):
        ARCHIVE_NAME = f"{ARCHIVE_NAME}.tar.gz"
    
    print(f"Archive file name (tar.gz): {ARCHIVE_NAME}")
    
    # 출력 Parquet 파일명 결정 (우선순위: 명령행 인자 > 환경변수 > 기본값)
    # 이것은 파티션 통합 후 생성되는 parquet 파일명입니다
    if args.output_filename:
        OUTPUT_FILENAME = args.output_filename
    else:
        OUTPUT_FILENAME = os.getenv('OUTPUT_FILENAME', 'merged.parquet')
    
    print(f"Output parquet file name: {OUTPUT_FILENAME}")
    
    # 필수 환경 변수 확인
    required_vars = {
        'HDFS_PATH': HDFS_PATH,
        'REMOTE_USER': REMOTE_USER,
        'REMOTE_PASSWORD': REMOTE_PASSWORD,
        'REMOTE_IP': REMOTE_IP,
        'REMOTE_PATH': REMOTE_PATH
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        print(f"Error: 다음 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
        print(f".env 파일({args.env_file})을 확인해주세요.")
        sys.exit(1)
    
    # 디렉토리 이름 추출
    DIR_NAME = os.path.basename(HDFS_PATH)
    
    # sshpass 설치 확인
    check_sshpass()
    
    # 원격 파일 삭제 (옵션)
    if not args.skip_remove:
        if not remove_remote_files(
            REMOTE_USER, REMOTE_PASSWORD, REMOTE_IP, REMOTE_PATH, 
            DIR_NAME, OUTPUT_FILENAME, ARCHIVE_NAME, 
            merge_partitions=args.merge_partitions,
            extract_remote=args.extract_remote
        ):
            print("Warning: Failed to remove remote files")
    else:
        print("\n# 원격 파일 삭제 단계를 건너뜁니다")
    
    # HDFS에서 다운로드 (옵션)
    if not args.skip_download:
        # 새로 다운로드하는 경우 로컬 디렉토리 정리
        # cleanup 옵션을 전달하여 파일 삭제 여부 결정
        if not prepare_local_directory(LOCAL_TMP_PATH, DIR_NAME, ARCHIVE_NAME, cleanup=args.cleanup):
            print("Error: Failed to prepare local directory")
            sys.exit(1)
        
        if not download_from_hdfs(HDFS_PATH, LOCAL_TMP_PATH):
            sys.exit(1)
    else:
        print("\n# HDFS 다운로드 단계를 건너뜁니다")
        print(f"로컬 파일 사용: {LOCAL_TMP_PATH}/{DIR_NAME}")
        
        # 파일 존재 여부 확인
        local_dir = os.path.join(LOCAL_TMP_PATH, DIR_NAME)
        if not os.path.exists(local_dir):
            print(f"Error: 로컬 디렉토리가 존재하지 않습니다: {local_dir}")
            print("--skip-download 옵션을 사용하려면 먼저 파일을 다운로드해야 합니다.")
            sys.exit(1)
    
    # 파티션 통합 (옵션)
    # 원본 파티션 디렉토리명 저장 (cleanup에서 사용)
    ORIGINAL_DIR_NAME = DIR_NAME
    
    if args.merge_partitions:
        source_dir = os.path.join(LOCAL_TMP_PATH, DIR_NAME)
        
        # 출력 파일명은 위에서 이미 결정됨 (OUTPUT_FILENAME)
        output_filename = OUTPUT_FILENAME
        
        # 통합 파일을 위한 별도 디렉토리 생성 (원본 파티션 디렉토리와 구분)
        merged_dir_name = f"{DIR_NAME}_merged"
        merged_dir = os.path.join(LOCAL_TMP_PATH, merged_dir_name)
        os.makedirs(merged_dir, exist_ok=True)
        
        # 출력 파일 경로 (별도 디렉토리에 직접 생성)
        output_file = os.path.join(merged_dir, output_filename)
        
        print(f"\n# 파티션 통합")
        print(f"원본 디렉토리: {source_dir} (유지됨)")
        print(f"통합 파일: {output_file}")
        
        # 파티션 통합 실행
        if not merge_partitioned_parquet(
            source_dir=source_dir,
            output_file=output_file,
            batch_size=args.batch_size,
            compression=args.compression,
            verbose=True
        ):
            print("Error: Failed to merge partitioned parquet files")
            sys.exit(1)
        
        # 원본 파티션 디렉토리는 유지됨
        print(f"\n원본 파티션 디렉토리 유지: {source_dir}")
        
        # 전송 시 통합 파일이 있는 디렉토리 사용
        DIR_NAME = merged_dir_name
    else:
        print("\n# 파티션 통합을 건너뜁니다 (--merge-partitions 옵션 미사용)")
    
    # 전송 방식 구분
    if args.merge_partitions:
        # 파티션 통합 모드: 단일 parquet 파일 직접 전송
        print("\n# 통합 파일 직접 전송 (압축 없음)")
        
        # 통합된 parquet 파일 경로
        merged_parquet = os.path.join(LOCAL_TMP_PATH, DIR_NAME, OUTPUT_FILENAME)
        
        if not os.path.exists(merged_parquet):
            print(f"Error: 통합 파일이 존재하지 않습니다: {merged_parquet}")
            sys.exit(1)
        
        # 작업 디렉토리를 통합 파일이 있는 디렉토리로 변경
        os.chdir(os.path.join(LOCAL_TMP_PATH, DIR_NAME))
        
        # parquet 파일 직접 전송
        if not transfer_data(REMOTE_USER, REMOTE_PASSWORD, REMOTE_IP, REMOTE_PATH, OUTPUT_FILENAME):
            sys.exit(1)
        
        # 압축 해제 단계 건너뛰기 (단일 파일이므로 불필요)
        print("압축 해제 단계 건너뜀 (단일 파일 전송)")
        
        # EOF 파일 생성 (OUTPUT_FILENAME 기반)
        if not create_eof_file(REMOTE_USER, REMOTE_PASSWORD, REMOTE_IP, REMOTE_PATH, OUTPUT_FILENAME):
            print("Warning: Failed to create EOF file")
    else:
        # 파티션 구조 유지 모드: 압축 후 전송
        print("\n# 파티션 구조 유지 모드: 압축 후 전송")
        
        # 압축
        if not compress_data(LOCAL_TMP_PATH, DIR_NAME, ARCHIVE_NAME):
            sys.exit(1)
        
        # 전송
        if not transfer_data(REMOTE_USER, REMOTE_PASSWORD, REMOTE_IP, REMOTE_PATH, ARCHIVE_NAME):
            sys.exit(1)
        
        # 압축 해제 (옵션)
        if args.extract_remote:
            print("\n# 원격 서버에서 압축 해제 (--extract-remote 옵션 사용)")
            if not extract_remote(REMOTE_USER, REMOTE_PASSWORD, REMOTE_IP, REMOTE_PATH, ARCHIVE_NAME):
                sys.exit(1)
            
            # EOF 파일 생성 (압축 해제된 파일 기반)
            if not create_eof_file(REMOTE_USER, REMOTE_PASSWORD, REMOTE_IP, REMOTE_PATH, ARCHIVE_NAME):
                print("Warning: Failed to create EOF file")
        else:
            print("\n# 압축 해제 건너뜀 (tar.gz 파일 유지)")
            print(f"원격 서버에 {ARCHIVE_NAME} 파일이 유지됩니다")
            print("압축 해제를 원하면 --extract-remote 옵션을 사용하세요")
            
            # EOF 파일 생성 (tar.gz 파일 기반)
            if not create_eof_file(REMOTE_USER, REMOTE_PASSWORD, REMOTE_IP, REMOTE_PATH, ARCHIVE_NAME):
                print("Warning: Failed to create EOF file")
    
    # 정리 (옵션)
    if args.cleanup:
        # --cleanup 사용: 로컬 파일 삭제
        if args.merge_partitions:
            # 파티션 통합 모드: 원본 + 통합 디렉토리 정리
            print("\n# 정리 (--cleanup)")
            print("Cleaning up...")
            original_dir = os.path.join(LOCAL_TMP_PATH, ORIGINAL_DIR_NAME)
            merged_dir = os.path.join(LOCAL_TMP_PATH, DIR_NAME)  # {ORIGINAL_DIR_NAME}_merged
            
            print(f"  - 원본 파티션 디렉토리: {original_dir}")
            print(f"  - 통합 파일 디렉토리: {merged_dir}")
            
            run_command(f"rm -rf {original_dir}", check=False)
            run_command(f"rm -rf {merged_dir}", check=False)
        else:
            # 파티션 구조 유지 모드: 기존 방식
            cleanup(LOCAL_TMP_PATH, DIR_NAME, ARCHIVE_NAME)
    else:
        # 기본 모드: 로컬 파일 유지
        print("\n# 로컬 파일 유지 (기본)")
        if args.merge_partitions:
            print(f"다음 파일들이 유지됩니다:")
            print(f"  - {LOCAL_TMP_PATH}/{ORIGINAL_DIR_NAME} (원본 파티션)")
            print(f"  - {LOCAL_TMP_PATH}/{DIR_NAME} (통합 파일)")
        else:
            print(f"다음 파일들이 유지됩니다:")
            print(f"  - {LOCAL_TMP_PATH}/{DIR_NAME}")
            print(f"  - {LOCAL_TMP_PATH}/{ARCHIVE_NAME}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()