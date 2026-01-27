error id: 7B24F233620AC68017BC6B6C3469744D
file://<WORKSPACE>/Main.scala
### dotty.tools.dotc.core.UnpicklingError: Could not read definition object Predef in <HOME>/Library/Caches/Coursier/v1/https/repo1.maven.org/maven2/org/scala-lang/scala-library/3.8.0/scala-library-3.8.0.jar(scala/Predef.tasty). Caused by the following exception:
java.lang.AssertionError: assertion failed: `-Xread-docs` enabled, but no `docCtx` is set.

Run with -Ydebug-unpickling to see full stack trace.

occurred in the presentation compiler.



action parameters:
offset: 9
uri: file://<WORKSPACE>/Main.scala
text:
```scala
object Ma@@

```


presentation compiler configuration:
Scala version: 3.8.0-bin-nonbootstrapped
Classpath:
<WORKSPACE>/.scala-build/AgenticWorkflow_d5c0a6989e/classes/main [exists ], <HOME>/Library/Caches/Coursier/v1/https/repo1.maven.org/maven2/org/scala-lang/scala3-library_3/3.8.0/scala3-library_3-3.8.0.jar [exists ], <HOME>/Library/Caches/Coursier/v1/https/repo1.maven.org/maven2/org/scala-lang/scala-library/3.8.0/scala-library-3.8.0.jar [exists ], <HOME>/Library/Caches/Coursier/v1/https/repo1.maven.org/maven2/com/sourcegraph/semanticdb-javac/0.10.0/semanticdb-javac-0.10.0.jar [exists ], <WORKSPACE>/.scala-build/AgenticWorkflow_d5c0a6989e/classes/main/META-INF/best-effort [missing ]
Options:
-Xsemanticdb -sourceroot <WORKSPACE> -release 8 -Ywith-best-effort-tasty




#### Error stacktrace:

```

```
#### Short summary: 

dotty.tools.dotc.core.UnpicklingError: Could not read definition object Predef in <HOME>/Library/Caches/Coursier/v1/https/repo1.maven.org/maven2/org/scala-lang/scala-library/3.8.0/scala-library-3.8.0.jar(scala/Predef.tasty). Caused by the following exception:
java.lang.AssertionError: assertion failed: `-Xread-docs` enabled, but no `docCtx` is set.

Run with -Ydebug-unpickling to see full stack trace.