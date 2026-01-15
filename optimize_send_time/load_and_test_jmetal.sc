// ============================================================================
// Runner script for Spark Shell (Scala script)
// ============================================================================
// Usage:
//   ./spark-shell-with-lib.sh -i optimize_ost.scala -i load_and_test_jmetal.scala -i load_and_test_jmetal.sc
//
// Notes:
// - load_and_test_jmetal.scala defines LoadAndTestJMetal.run()
// - This .sc file calls it (top-level statements are OK in scripts)
// ============================================================================

LoadAndTestJMetal.run()

