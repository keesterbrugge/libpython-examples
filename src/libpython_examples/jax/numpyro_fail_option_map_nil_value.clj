(ns libpython-examples.jax.numpyro-fail-option-map-nil-value
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :refer [py.] :as py]))
   
(require-python '[jax.numpy :as jnp])
(require-python '[jax :refer [random]])
; (require-python '[ax.scipy.special :refer [logsumexp]]) ; errors
(require-python 'numpyro)
(require-python '[numpyro.distributions :as dist])
(require-python '[numpyro.infer :refer [MCMC, NUTS]])
(require-python 'operator)



(def rng_key (py. random PRNGKey 0))
(def new_rng_keys (py. random split rng_key))
(def num_warmup 1000)
(def num_samples 2000)


;; python code 
"def model(divorce=None):
    mu = numpyro.sample('mu', dist.Normal(0., 0.2))
    numpyro.sample('obs', dist.Normal(mu, sigma), obs=divorce)"

(defn model 
  [{:keys [y]}]
  (let [mu (numpyro/sample "mu" (dist/Normal 0. 0.2))]
    (numpyro/sample "obs" (dist/Normal mu 1) :obs y)))

(def kernel (NUTS model))
(def mcmc (MCMC kernel num_warmup num_samples))

;; this does not work, throws error
(py. mcmc run (last new_rng_keys) {})
;; =>
;; ; Execution error at libpython-clj.python.interpreter/check-error-throw (interpreter.clj:499).
; Traceback (most recent call last):
  File "/Users/keesterbrugge/opt/anaconda3/lib/python3.8/site-packages/numpyro/infer/mcmc.py", line 446, in run
    states_flat, last_state = partial_map_fn(map_args)
  File "/Users/keesterbrugge/opt/anaconda3/lib/python3.8/site-packages/numpyro/infer/mcmc.py", line 312, in _single_chain_mcmc
    init_state = self.sampler.init(rng_key, self.num_warmup, init_params,
  File "/Users/keesterbrugge/opt/anaconda3/lib/python3.8/site-packages/numpyro/infer/hmc.py", line 450, in init
    init_params = self._init_state(rng_key_init_model, model_args, model_kwargs, init_params)
  File "/Users/keesterbrugge/opt/anaconda3/lib/python3.8/site-packages/numpyro/infer/hmc.py", line 407, in _init_state
    init_params, potential_fn, postprocess_fn, model_trace = initialize_model(
  File "/Users/keesterbrugge/opt/anaconda3/lib/python3.8/site-packages/numpyro/infer/util.py", line 394, in initialize_model
; 
;     inv_transforms, replay_model, has_enumerate_support, model_trace = _get_model_transforms(
  File "/Users/keesterbrugge/opt/anaconda3/lib/python3.8/site-packages/numpyro/infer/util.py", line 265, in _get_model_transforms
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
  File "/Users/keesterbrugge/opt/anaconda3/lib/python3.8/site-packages/numpyro/handlers.py", line 156, in get_trace
    self(*args, **kwargs)
  File "/Users/keesterbrugge/opt/anaconda3/lib/python3.8/site-packages/numpyro/primitives.py", line 68, in __call__
    return self.fn(*args, **kwargs)
  File "/Users/keesterbrugge/opt/anaconda3/lib/python3.8/site-packages/numpyro/primitives.py", line 68, in __call__
    return self.fn(*args, **kwargs)
  File "/Users/keesterbrugge/opt/anaconda3/lib/python3.8/site-packages/numpyro/primitives.py", line 68, in __call__
    return self.fn(*args, **kwargs)
Exception: java.lang.Exception: KeyError: 'y'
:java.lang.Exception: KeyError: 'y'

 at libpython_clj.python.interpreter$check_error
; _throw.invokeStatic (interpreter.clj:499)
;     libpython_clj.python.interpreter$check_error_throw.invoke (interpreter.clj:497)
    libpython_clj.python.object$wrap_pyobject.invokeStatic (object.clj:173)
    libpython_clj.python.object$wrap_pyobject.invoke (object.clj:120)
    libpython_clj.python.object$eval45745$fn__45746.invoke (object.clj:839)
    libpython_clj.python.protocols$eval40569$fn__40570$G__40560__40579.invoke (protocols.clj:105)
    libpython_clj.python.bridge$py_impl_call_raw.invokeStatic (bridge.clj:336)
    libpython_clj.python.bridge$py_impl_call_raw.invoke (bridge.clj:333)
    libpython_clj.python.bridge$py_impl_call_as.invokeStatic (bridge.clj:344)
    libpython_clj.python.bridge$py_impl_call_as.invoke (bridge.clj:342)
    libpython_clj.python.bridge$generic_python_as_map$py_call__46299.doInvoke (bridge.clj:367)
    clojure.lang.RestFn.invoke (RestFn.java:423)
    libpython_clj.python.bridge$generic_python_as_map$reify__46307.get (bridge.clj:381)
    clojure.lang.RT.getFrom (RT.java:769)
  
;   clojure.lang.RT.get (RT.java:761)
;     libpython_examples.jax.numpyro_fail_option_map_nil_value$model.invokeStatic (NO_SOURCE_FILE:39)
    libpython_examples.jax.numpyro_fail_option_map_nil_value$model.invoke (NO_SOURCE_FILE:39)
    clojure.lang.AFn.applyToHelper (AFn.java:154)
    clojure.lang.AFn.applyTo (AFn.java:144)
    clojure.core$apply.invokeStatic (core.clj:665)
    clojure.core$apply.invoke (core.clj:660)
    libpython_clj.python.object$make_tuple_fn$reify__45514.pyinvoke (object.clj:510)
    sun.reflect.GeneratedMethodAccessor11.invoke (:-1)
    sun.reflect.DelegatingMethodAccessorImpl.invoke (DelegatingMethodAccessorImpl.java:43)
    java.lang.reflect.Method.invoke (Method.java:498)
    com.sun.jna.CallbackReference$DefaultCallbackProxy.invokeCallback (CallbackReference.java:520)
    com.sun.jna.CallbackReference$DefaultCallbackProxy.callback (CallbackReference.java:551)
    libpython_clj.jna.DirectMapped.PyObject_CallObject (DirectMapped.java:-2)
    libpython_clj.jna.protocols.object$PyObject_C
; allObject.invokeStatic (object.clj:305)
;     libpython_clj.jna.protocols.object$PyObject_CallObject.invoke (object.clj:295)
    libpython_clj.python.object$eval45745$fn__45746.invoke (object.clj:836)
    libpython_clj.python.protocols$eval40569$fn__40570$G__40560__40579.invoke (protocols.clj:105)
    libpython_clj.python.bridge$generic_python_as_jvm$reify__46630.do_call_fn (bridge.clj:675)
    libpython_clj.python.protocols$call_attr_kw.invokeStatic (protocols.clj:132)
    libpython_clj.python.protocols$call_attr_kw.invoke (protocols.clj:127)
    libpython_examples.jax.numpyro_fail_option_map_nil_value$eval121553.invokeStatic (NO_SOURCE_FILE:48)
    libpython_examples.jax.numpyro_fail_option_map_nil_value$eval121553.invoke (NO_SOURCE_FILE:48)
    clojure.lang.Compiler.eval (Compiler.java:7177)
    clojure.lang.Compiler.eval (Compiler.java:7132)
    clojure.core$eval.invokeStatic (core.clj:3214)
    clojure.core$eval.invoke (core.clj:3210)
    clojure.main$repl$read_eval_print__9086$fn__9089.invoke (main.clj:43
; 7)
;     clojure.main$repl$read_eval_print__9086.invoke (main.clj:437)
    clojure.main$repl$fn__9095.invoke (main.clj:458)
    clojure.main$repl.invokeStatic (main.clj:458)
    clojure.main$repl.doInvoke (main.clj:368)
    clojure.lang.RestFn.invoke (RestFn.java:1523)
    nrepl.middleware.interruptible_eval$evaluate.invokeStatic (interruptible_eval.clj:79)
    nrepl.middleware.interruptible_eval$evaluate.invoke (interruptible_eval.clj:55)
    nrepl.middleware.interruptible_eval$interruptible_eval$fn__935$fn__939.invoke (interruptible_eval.clj:142)
    clojure.lang.AFn.run (AFn.java:22)
    nrepl.middleware.session$session_exec$main_loop__1036$fn__1040.invoke (session.clj:171)
    nrepl.middleware.session$session_exec$main_loop__1036.invoke (session.clj:170)
    clojure.lang.AFn.run (AFn.java:22)
    java.lang.Thread.run (Thread.java:745)


[{:file "jar:file:/Users/keesterbrugge/.m2/repository/clj-python/libpython-clj/1.46/libpython-clj-1.46.jar!/libpython_clj/python/interpreter.clj:499" :fn "check-error-throw"  :method "invokeStatic" :flags [:clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/clj-python/libpython-clj/1.46/libpython-clj-1.46.jar!/libpython_clj/python/interpreter.clj:497" :fn "check-error-throw"  :method "invoke" :flags [:clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/clj-python/libpython-clj/1.46/libpython-clj-1.46.jar!/libpython_clj/python/object.clj:173" :fn "wrap-pyobject"  :method "invokeStatic" :flags [:clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/clj-python/libpython-clj/1.46/libpython-clj-1.46.jar!/libpython_clj/python/object.clj:120" :fn "wrap-pyobject"  :method "invoke" :flags [:clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/clj-python/libpython-clj/1.46/libpython-clj-1.46.jar!/libpython_clj/python/object.clj:839" :fn "eval45745/fn"  :method "invoke" :flags [:clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/clj-python/libpython-clj/1.46/libpython-clj-1.46.jar!/libpython_clj/python/protocols.clj:105" :fn "eval40569/fn/G"  :method "invoke" :flags [:clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/clj-python/libpython-clj/1.46/libpython-clj-1.46.jar!/libpython_clj/python/bridge.clj:675" :fn "generic-python-as-jvm/reify"  :method "do_call_fn" :flags [:clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/clj-python/libpython-clj/1.46/libpython-clj-1.46.jar!/libpython_clj/python/protocols.clj:132" :fn "call-attr-kw"  :method "invokeStatic" :flags [:clj]}
 {:file "jar:file:/Users/keesterbrugge/.m2/repository/clj-python/libpython-clj/1.46/libpython-clj-1.46.jar!/libpython_clj/python/protocols.clj:127" :fn "call-attr-kw"  :method "invoke" :flags [:clj]}
 {:file "NO_SOURCE_FILE" :line 48 :fn "eval121553"  :method "invokeStatic" :flags [:project :repl :clj]}
 {:file "NO_SOURCE_FILE" :line 48 :fn "eval121553"  :method "invoke" :flags [:dup :project :repl :clj]}
 {:file "Compiler.java" :line 7177 :method "eval" :flags [:tooling :java]}
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
 {:file "Thread.java" :line 745 :method "run" :flags [:java]}]

;; explicit :y key with nil value DOES work
(py. mcmc run (last new_rng_keys) {:y nil})

(defn model2
  [{:keys [y] :or {y nil}}]
  (let [mu (numpyro/sample "mu" (dist/Normal 0. 0.2))]
    (numpyro/sample "obs" (dist/Normal mu 1) :obs y)))

(def kernel (NUTS model2))
(def mcmc (MCMC kernel num_warmup num_samples))

; now this works too
(py. mcmc run (last new_rng_keys) {})





(jnp/add (jnp/array [2 3 ]) 0)















(defn model_simple2 [{:keys [y ]}]
  (let [a (numpyro/sample "a" (dist/Normal 0 3))]
    (numpyro/sample "mu" (dist/Normal a 1) :obs y)))



(def rng_key (py. random PRNGKey 0))

(def new_rng_keys (py. random split rng_key))

(def num_warmup 1000)

(def num_samples 2000)

(def kernel (NUTS model_simple2))

(def mcmc (MCMC kernel num_warmup num_samples))

(py. mcmc run (last new_rng_keys) {:y (jnp/array [0.6 0.7])})

(py. mcmc print_summary)


