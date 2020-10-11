"Tried to reproduce an require-python failure for 
 numpyro, but couldn't"
(ns libpython-examples.jax.numpyro-import-fail
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :refer [py.] :as py]))


;; python script to implement
(def py-script "
import numpyro
import numpyro.distributions as dist
 
def model_simpler():
    numpyro.sample('mu', dist.Normal(0., 5))")

;; works fine
(def bridged-simpler-model (py/run-simple-string py-script))

;; Remake in clj -----
(require-python 'numpyro) ; => :ok
(py/import-as numpyro.distributions dist)

;; doesn't work. numpyro is not an object you can access attribute of.
;; it is a ns. 
(defn model_simple1 []
  (py. numpyro sample "mu" (py. dist Normal 0 5))) ; error: Unable to resolve symbol: numpyro in this context
;=> 
; Syntax error compiling at (src/libpython_examples/jax/numpyro_import_fail.clj:48:3).
; Unable to resolve symbol: numpyro in this context
[{:file "Compiler.java" :line 6808 :method "analyze" :flags [:tooling :java]}
 {:file "Compiler.java" :line 6745 :method "analyze" :flags [:dup :tooling :java]}
 {:file "Compiler.java" :line 3888 :method "parse" :flags [:tooling :java]}
 {:file "Compiler.java" :line 7109 :method "analyzeSeq" :flags [:tooling :java]}
 {:file "Compiler.java" :line 6789 :method "analyze" :flags [:tooling :java]}
 {:file "Compiler.java" :line 7095 :method "analyzeSeq" :flags [:tooling :java]}
 {:file "Compiler.java" :line 6789 :method "analyze" :flags [:tooling :java]}
 {:file "Compiler.java" :line 7095 :method "analyzeSeq" :flags [:tooling :java]}
 {:file "Compiler.java" :line 6789 :method "analyze" :flags [:tooling :java]}
 {:file "Compiler.java" :line 7095 :method "analyzeSeq" :flags [:tooling :java]}
 {:file "Compiler.java" :line 6789 :method "analyze" :flags [:tooling :java]}
 {:file "Compiler.java" :line 6745 :method "analyze" :flags [:dup :tooling :java]}
 {:file "Compiler.java" :line 6120 :method "parse" :flags [:tooling :java]}
 {:file "Compiler.java" :line 5467 :method "parse" :flags [:tooling :java]}
 {:file "Compiler.java" :line 4029 :method "parse" :flags [:tooling :java]}
 {:file "Compiler.java" :line 7105 :method "analyzeSeq" :flags [:tooling :java]}
 {:file "Compiler.java" :line 6789 :method "analyze" :flags [:tooling :java]}
 {:file "Compiler.java" :line 7095 :method "analyzeSeq" :flags [:tooling :java]}
 {:file "Compiler.java" :line 6789 :method "analyze" :flags [:tooling :java]}
 {:file "Compiler.java" :line 38 :method "access$300" :flags [:tooling :java]}
 {:file "Compiler.java" :line 596 :method "parse" :flags [:tooling :java]}
 {:file "Compiler.java" :line 7107 :method "analyzeSeq" :flags [:tooling :java]}
 {:file "Compiler.java" :line 6789 :method "analyze" :flags [:tooling :java]}
 {:file "Compiler.java" :line 6745 :method "analyze" :flags [:dup :tooling :java]}
 {:file "Compiler.java" :line 7181 :method "eval" :flags [:tooling :java]}
 {:file "Compiler.java" :line 7132 :method "eval" :flags [:dup :tooling :java]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/org/clojure/clojure/1.10.1/clojure-1.10.1.jar!/clojure/core.clj:3214" :fn "eval"  :method "invokeStatic" :flags [:clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/org/clojure/clojure/1.10.1/clojure-1.10.1.jar!/clojure/core.clj:3210" :fn "eval"  :method "invoke" :flags [:clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/org/clojure/clojure/1.10.1/clojure-1.10.1.jar!/clojure/main.clj:437" :fn "repl/read-eval-print/fn"  :method "invoke" :flags [:clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/org/clojure/clojure/1.10.1/clojure-1.10.1.jar!/clojure/main.clj:437" :fn "repl/read-eval-print"  :method "invoke" :flags [:dup :clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/org/clojure/clojure/1.10.1/clojure-1.10.1.jar!/clojure/main.clj:458" :fn "repl/fn"  :method "invoke" :flags [:clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/org/clojure/clojure/1.10.1/clojure-1.10.1.jar!/clojure/main.clj:458" :fn "repl"  :method "invokeStatic" :flags [:dup :clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/org/clojure/clojure/1.10.1/clojure-1.10.1.jar!/clojure/main.clj:368" :fn "repl"  :method "doInvoke" :flags [:clj]}
 {:file "RestFn.java" :line 1523 :method "invoke" :flags [:java]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/nrepl/nrepl/0.6.0/nrepl-0.6.0.jar!/nrepl/middleware/interruptible_eval.clj:79" :fn "evaluate"  :method "invokeStatic" :flags [:tooling :clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/nrepl/nrepl/0.6.0/nrepl-0.6.0.jar!/nrepl/middleware/interruptible_eval.clj:55" :fn "evaluate"  :method "invoke" :flags [:tooling :clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/nrepl/nrepl/0.6.0/nrepl-0.6.0.jar!/nrepl/middleware/interruptible_eval.clj:142" :fn "interruptible-eval/fn/fn"  :method "invoke" :flags [:tooling :clj]}
 {:file "AFn.java" :line 22 :method "run" :flags [:java]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/nrepl/nrepl/0.6.0/nrepl-0.6.0.jar!/nrepl/middleware/session.clj:171" :fn "session-exec/main-loop/fn"  :method "invoke" :flags [:tooling :clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/nrepl/nrepl/0.6.0/nrepl-0.6.0.jar!/nrepl/middleware/session.clj:170" :fn "session-exec/main-loop"  :method "invoke" :flags [:tooling :clj]}
 {:file "AFn.java" :line 22 :method "run" :flags [:java]}
 {:file "Thread.java" :line 745 :method "run" :flags [:java]}])

;; does work
(defn model_simple2 []
  (numpyro/sample "mu" (py. dist Normal 0 5)))



;; does work
(py/from-import numpyro sample)

(defn model_simple3 []
  (sample "mu" (py. dist Normal 0 5)))

;; however the following which I believed 
;; should do something similar DOES work. 
;; Am I doing something wrong?
;; 


(py/import-as jax.numpy jnp)
(py. jnp array [2 3]) ; works
(jnp/array [2 3]) ; error

(require-python '[jax.numpy :as jnp2])
(py. jnp2 array [2 3]) ; error
(jnp2/array [2 3]) ; works