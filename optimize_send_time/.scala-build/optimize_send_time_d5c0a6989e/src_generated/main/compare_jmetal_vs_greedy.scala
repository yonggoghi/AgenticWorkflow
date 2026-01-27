

final class compare_jmetal_vs_greedy$_ {
def args = compare_jmetal_vs_greedy_sc.args$
def scriptPath = """compare_jmetal_vs_greedy.sc"""
/*<script>*/
// ============================================================================
// Runner script (Scala script) for spark-shell
// ============================================================================
// Usage:
//   SPARK_SHELL_XSS=8m ./spark-shell-with-lib.sh \
//     -i optimize_ost.scala \
//     -i greedy_allocation.scala \
//     -i compare_jmetal_vs_greedy.scala \
//     -i compare_jmetal_vs_greedy.sc
// ============================================================================

CompareJMetalVsGreedy.run()


/*</script>*/ /*<generated>*//*</generated>*/
}

object compare_jmetal_vs_greedy_sc {
  private var args$opt0 = Option.empty[Array[String]]
  def args$set(args: Array[String]): Unit = {
    args$opt0 = Some(args)
  }
  def args$opt: Option[Array[String]] = args$opt0
  def args$: Array[String] = args$opt.getOrElse {
    sys.error("No arguments passed to this script")
  }

  lazy val script = new compare_jmetal_vs_greedy$_

  def main(args: Array[String]): Unit = {
    args$set(args)
    val _ = script.hashCode() // hashCode to clear scalac warning about pure expression in statement position
  }
}

export compare_jmetal_vs_greedy_sc.script as `compare_jmetal_vs_greedy`

