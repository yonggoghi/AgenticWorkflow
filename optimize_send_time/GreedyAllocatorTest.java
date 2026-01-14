package optimize_send_time;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.HashMap;
import java.util.Map;

/**
 * GreedyAllocator 테스트 클래스
 * 
 * 컴파일:
 *   javac -cp "$(find $SPARK_HOME/jars -name '*.jar' | tr '\n' ':')" \
 *         GreedyAllocator.java GreedyAllocatorTest.java
 * 
 * 실행:
 *   spark-submit \
 *     --class optimize_send_time.GreedyAllocatorTest \
 *     --driver-memory 16g \
 *     --executor-memory 16g \
 *     GreedyAllocatorTest.jar
 */
public class GreedyAllocatorTest {
    
    public static void main(String[] args) {
        // Spark Session 생성
        SparkSession spark = SparkSession.builder()
                .appName("GreedyAllocatorTest")
                .master("local[*]")
                .getOrCreate();
        
        spark.sparkContext().setLogLevel("WARN");
        
        System.out.println("\n" + repeat("=", 80));
        System.out.println("GreedyAllocator Large-Scale Test (Java)");
        System.out.println(repeat("=", 80));
        
        try {
            // 데이터 경로
            String dataPath = "aos/sto/propensityScoreDF";
            
            // 1. 데이터 로드
            System.out.println("\n[1] Loading data...");
            Dataset<Row> df = spark.read().parquet(dataPath).cache();
            long totalUsers = df.select("svc_mgmt_num").distinct().count();
            System.out.println(String.format("✓ Total users: %,d", totalUsers));
            
            // 2. 용량 설정
            System.out.println("\n[2] Setting up capacity...");
            int[] hours = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
            int capacityPerHour = (int) (totalUsers * 0.11);
            
            Map<Integer, Integer> capacity = new HashMap<>();
            for (int hour : hours) {
                capacity.put(hour, capacityPerHour);
            }
            
            long totalCapacity = capacity.values().stream()
                    .mapToLong(Integer::longValue)
                    .sum();
            
            System.out.println(String.format("✓ Hours: %d time slots", hours.length));
            System.out.println(String.format("✓ Capacity per hour: %,d", capacityPerHour));
            System.out.println(String.format("✓ Total capacity: %,d (%.2fx)",
                    totalCapacity, (double) totalCapacity / totalUsers));
            
            // 3. 배치 크기 결정
            System.out.println("\n[3] Determining batch size...");
            int batchSize;
            if (totalUsers > 10000000) {
                batchSize = 1000000;  // 1000만명 이상: 100만 배치
            } else if (totalUsers > 1000000) {
                batchSize = 500000;   // 100만-1000만: 50만 배치
            } else {
                batchSize = 100000;   // 100만 이하: 10만 배치
            }
            
            int numBatches = (int) Math.ceil((double) totalUsers / batchSize);
            System.out.println(String.format("✓ Batch size: %,d", batchSize));
            System.out.println(String.format("✓ Number of batches: %d", numBatches));
            
            // 4. 할당 실행
            System.out.println("\n[4] Running allocation...");
            GreedyAllocator allocator = new GreedyAllocator();
            
            long startTime = System.currentTimeMillis();
            Dataset<Row> result = allocator.allocateLargeScale(df, hours, capacity, batchSize);
            long endTime = System.currentTimeMillis();
            
            double totalTime = (endTime - startTime) / 1000.0;
            double totalMinutes = totalTime / 60.0;
            
            // 5. 결과 확인
            System.out.println("\n[5] Showing sample results...");
            result.show(20, false);
            
            // 6. 저장 (옵션)
            System.out.println("\n[6] Result available for saving:");
            System.out.println("  result.write().mode(\"overwrite\").parquet(\"output/greedy_result_java\");");
            
            // 7. 완료
            System.out.println("\n" + repeat("=", 80));
            System.out.println("Test Complete!");
            System.out.println(repeat("=", 80));
            System.out.println(String.format("Total execution time: %.2f seconds (%.2f minutes)",
                    totalTime, totalMinutes));
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
