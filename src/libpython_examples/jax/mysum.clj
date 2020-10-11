(ns libpython-examples.jax.mysum
  "minimal working test of taking a function created in clojure and using it as 
   argument for higher order function in xla stack, in this case jit."
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py]))


(require-python '[jax.numpy :as jnp])
(py/from-import jax jit)

(def x (jnp/array [2 3]))

(jnp/sum x)

(def jitsum (jit jnp/sum))

(jitsum x )

(defn mysum [x] (jnp/sum x))

(mysum x)

(def jitmysum (jit mysum))

(jitmysum x)

(time (jitmysum x))






"def mysum(x):
    return jnp.sum(x)
 x = jnp.array([2,3])
 mysum(x)
 jitmysum = jit (mysum)
 jitmysum (x)"





