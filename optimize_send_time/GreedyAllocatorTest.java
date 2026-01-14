package optimize_send_time;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.HashMap;
import java.util.Map;

/**
 * GreedyAllocator 테스트 및 실행 클래스
 * 
 * ============================================================
 * 목적: 대규모 사용자 할당 알고리즘의 End-to-End 테스트
 * ============================================================
 * 
 * 기능:
 * 1. 데이터 로드 및 검증
 * 2. 용량 설정 (시간대별)
 * 3. 배치 크기 자동 결정
 * 4. Greedy 할당 실행
 * 5. 결과 저장 및 통계 출력
 * 
 * 실행 흐름:
 * aos/sto/propensityScoreDF (입력)
 *   → GreedyAllocator.allocateLargeScale()
 *   → aos/sto/allocation_result (출력)
 * 
 * 컴파일:
 *   ./build_java.sh
 * 
 * 실행:
 *   spark-submit \
 *     --class optimize_send_time.GreedyAllocatorTest \
 *     --driver-memory 16g \
 *     --executor-memory 16g \
 *     build/greedy-allocator.jar
 * 
 * 메모리 권장사항:
 * - 100만명: 8GB
 * - 1000만명: 16GB
 * - 2500만명: 32GB
 */
public class GreedyAllocatorTest {
    
    public static void main(String[] args) {
        // ===== 1. Spark Session 초기화 =====
        // local[*]: 로컬 모드에서 사용 가능한 모든 코어 사용
        SparkSession spark = SparkSession.builder()
                .appName("GreedyAllocatorTest")
                .master("local[*]")
                .getOrCreate();
        
        // 로그 레벨을 WARN으로 설정하여 INFO 로그 숨김
        spark.sparkContext().setLogLevel("WARN");
        
        System.out.println("\n" + repeat("=", 80));
        System.out.println("GreedyAllocator Large-Scale Test (Java)");
        System.out.println(repeat("=", 80));
        
        try {
            // ===== 2. 데이터 경로 설정 =====
            // 입력: Propensity Score가 계산된 사용자 데이터
            // 컬럼: svc_mgmt_num (사용자ID), send_hour (시간대), propensity_score (예측값)
            String dataPath = "aos/sto/propensityScoreDF";
            
            // ===== 3. 데이터 로드 및 검증 =====
            System.out.println("\n[1] Loading data...");
            
            // Parquet 파일 읽기 및 캐싱 (재사용을 위해)
            Dataset<Row> df = spark.read().parquet(dataPath).cache();
            
            // 고유 사용자 수 계산 (중복 제거)
            // 한 사용자당 10개 레코드(9-18시) 있음
            long totalUsers = df.select("svc_mgmt_num").distinct().count();
            System.out.println(String.format("✓ Total users: %,d", totalUsers));
            
            // ===== 4. 용량 설정 =====
            // 각 시간대별로 얼마나 많은 사용자를 할당할 수 있는지 결정
            System.out.println("\n[2] Setting up capacity...");
            
            // 가능한 발송 시간대: 9시~18시 (10개 시간대)
            int[] hours = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
            
            // 시간대당 용량 = 전체 사용자의 11%
            // 전체 용량 = 전체 사용자의 110% (여유 있게 설정)
            // 이유: 모든 사용자가 할당될 수 있도록 보장
            int capacityPerHour = (int) (totalUsers * 0.11);
            
            // 시간대별 용량 Map 생성
            Map<Integer, Integer> capacity = new HashMap<>();
            for (int hour : hours) {
                capacity.put(hour, capacityPerHour);
            }
            
            // 총 용량 계산 (모든 시간대의 합)
            long totalCapacity = capacity.values().stream()
                    .mapToLong(Integer::longValue)
                    .sum();
            
            System.out.println(String.format("✓ Hours: %d time slots", hours.length));
            System.out.println(String.format("✓ Capacity per hour: %,d", capacityPerHour));
            System.out.println(String.format("✓ Total capacity: %,d (%.2fx)",
                    totalCapacity, (double) totalCapacity / totalUsers));
            
            // ===== 5. 배치 크기 자동 결정 =====
            // 사용자 수에 따라 적절한 배치 크기 선택
            // 배치 크기가 클수록: 빠르지만 메모리 많이 사용
            // 배치 크기가 작을수록: 느리지만 메모리 절약
            System.out.println("\n[3] Determining batch size...");
            int batchSize;
            if (totalUsers > 10000000) {
                // 1000만명 이상: 대규모
                batchSize = 1000000;  // 100만 배치 (메모리 충분한 경우)
            } else if (totalUsers > 1000000) {
                // 100만-1000만: 중규모
                batchSize = 500000;   // 50만 배치 (균형잡힌 선택)
            } else {
                // 100만 이하: 소규모
                batchSize = 100000;   // 10만 배치 (안전한 선택)
            }
            
            // 예상 배치 수 계산
            int numBatches = (int) Math.ceil((double) totalUsers / batchSize);
            System.out.println(String.format("✓ Batch size: %,d", batchSize));
            System.out.println(String.format("✓ Number of batches: %d", numBatches));
            
            // ===== 6. Greedy 할당 실행 =====
            // 가장 중요한 단계: 실제 알고리즘 실행
            System.out.println("\n[4] Running allocation...");
            
            // GreedyAllocator 인스턴스 생성
            GreedyAllocator allocator = new GreedyAllocator();
            
            // 실행 시간 측정 시작
            long startTime = System.currentTimeMillis();
            
            // allocateLargeScale() 호출
            // 입력: DataFrame, 시간대 배열, 용량 Map, 배치 크기
            // 출력: 할당 결과 DataFrame
            // 
            // 내부 동작:
            // 1. 사용자를 점수 기반으로 정렬 및 배치 분할
            // 2. 각 배치를 순차적으로 처리
            // 3. 전역 용량 관리 (배치 간 공유)
            // 4. 모든 배치 결과를 병합하여 반환
            Dataset<Row> result = allocator.allocateLargeScale(df, hours, capacity, batchSize);
            
            // 실행 시간 측정 종료
            long endTime = System.currentTimeMillis();
            
            double totalTime = (endTime - startTime) / 1000.0;
            double totalMinutes = totalTime / 60.0;
            
            // ===== 7. 결과 확인 (샘플) =====
            // 상위 20개 할당 결과 출력
            System.out.println("\n[5] Showing sample results...");
            result.show(20, false);
            
            // ===== 8. 결과 저장 =====
            // Parquet 포맷으로 저장 (압축되고 효율적)
            System.out.println("\n[6] Saving results...");
            String outputPath = "aos/sto/allocation_result";
            
            // mode("overwrite"): 기존 파일이 있으면 덮어쓰기
            // 결과 컬럼: svc_mgmt_num, assigned_hour, score
            result.write().mode("overwrite").parquet(outputPath);
            System.out.println(String.format("✓ Results saved to: %s", outputPath));
            
            // ===== 9. 최종 요약 =====
            System.out.println("\n" + repeat("=", 80));
            System.out.println("Test Complete!");
            System.out.println(repeat("=", 80));
            
            // 실행 시간 출력
            System.out.println(String.format("Total execution time: %.2f seconds (%.2f minutes)",
                    totalTime, totalMinutes));
            
            // 처리 속도 (초당 사용자 수)
            System.out.println(String.format("Throughput: %,.0f users/second",
                    totalUsers / totalTime));
            System.out.println(repeat("=", 80));
            
        } catch (Exception e) {
            System.err.println("\n❌ Error occurred:");
            e.printStackTrace();
        } finally {
            spark.stop();
        }
    }
    
    private static String repeat(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
}
