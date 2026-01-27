package optimize_send_time


final class load_and_test_jmetal$_ {
def args = load_and_test_jmetal_sc.args$
def scriptPath = """optimize_send_time/load_and_test_jmetal.sc"""
/*<script>*/
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


/*</script>*/ /*<generated>*//*</generated>*/
}

object load_and_test_jmetal_sc {
  private var args$opt0 = Option.empty[Array[String]]
  def args$set(args: Array[String]): Unit = {
    args$opt0 = Some(args)
  }
  def args$opt: Option[Array[String]] = args$opt0
  def args$: Array[String] = args$opt.getOrElse {
    sys.error("No arguments passed to this script")
  }

  lazy val script = new load_and_test_jmetal$_

  def main(args: Array[String]): Unit = {
    args$set(args)
    val _ = script.hashCode() // hashCode to clear scalac warning about pure expression in statement position
  }
}

export load_and_test_jmetal_sc.script as `load_and_test_jmetal`

