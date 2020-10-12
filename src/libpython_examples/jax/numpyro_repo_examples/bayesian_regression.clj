(ns libpython-examples.jax.numpyro-repo-examples.bayesian-regression
  (:require [clojure.tools.deps.alpha.repl :refer [add-lib]]
            [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :refer [py.] :as py]
            [tech.ml.dataset :as ds]
            [tech.ml.dataset.pipeline :as dsp]
            [camel-snake-kebab.core :as csk]))

;(add-lib 'camel-snake-kebab {:mvn/version "0.4.1"})

   
(require-python '[jax.numpy :as jnp])
(require-python '[jax :refer [random vmap]])
; (require-python '[ax.scipy.special :refer [logsumexp]]) ; errors
(py/from-import jax.scipy.special logsumexp)
(require-python 'numpyro)
(require-python '[numpyro.diagnostics :refer [hpdi]])
(require-python '[numpyro.distributions :as dist])
(py/from-import numpyro handlers)
(require-python '[numpyro.infer :refer [MCMC, NUTS]])
(require-python 'operator)
(py/from-import numpyro.util set_host_device_count)

;; may help stability, 
;; see https://github.com/clj-python/libpython-clj/issues/93#issuecomment-611202595
(set_host_device_count 1) 
;; try out py/with-gil-stack-rc-context if still stability issues

(def dset 
  (ds/->dataset "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv"
                {:separator \;
                 :key-fn csk/->kebab-case-keyword}))

(ds/column-names dset)
;; => (:location
;;     :loc
;;     :population
;;     :median-age-marriage
;;     :marriage
;;     :marriage-se
;;     :divorce
;;     :divorce-se
;;     :waffle-houses
;;     :south
;;     :slaves-1860
;;     :population-1860
;;     :prop-slaves-1860)


(def rng_key (py. random PRNGKey 0))
(def new_rng_keys (py. random split rng_key))
(def num_warmup 1000)
(def num_samples 2000)




(defn model
  
"def model(marriage=None, age=None, divorce=None):
a = numpyro.sample('a', dist.Normal(0., 0.2))
M, A = 0., 0.
if marriage is not None:
    bM = numpyro.sample('bM', dist.Normal(0., 0.5))
    M = bM * marriage
if age is not None:
    bA = numpyro.sample('bA', dist.Normal(0., 0.5))
    A = bA * age
sigma = numpyro.sample('sigma', dist.Exponential(1.))
mu = a + M + A
numpyro.sample('obs', dist.Normal(mu, sigma), obs=divorce)"
  
;[#_#_{:keys [marriage divorce #_median-age-marriage] :or  { median-age-marriage nil} :as m}}

[m]
(let [marriage (m :median-age-marriage)
      divorce (m :median-age-marriage)
      median-age-marriage (m :median-age-marriage)
      
      a (numpyro/sample "a" (dist/Normal 0. 1))

      M
      (if marriage
        (operator/mul marriage 
                      (numpyro/sample "bM" (dist/Normal 0. 0.5)))
        0)
      
      A
      (if median-age-marriage 
        (operator/mul median-age-marriage
                      (numpyro/sample "bA" (dist/Normal 0. 0.5)))
        0)
      
      mu (reduce jnp/add [a M A])
      
      sigma (numpyro/sample "sigma" (dist/Exponential 1.))]
  
    (numpyro/sample "obs" (dist/Normal mu sigma) :obs divorce)))


(def model-input 
  (as-> dset $
    (dsp/std-scale $)
    (select-keys $ [:marriage :divorce])
    (reduce-kv (fn [m k v] (assoc m k (jnp/array v))) {} $)
     ))

(def kernel (NUTS model))
(def mcmc (MCMC kernel num_warmup num_samples))
(py. mcmc run (last new_rng_keys) model-input)
(py. mcmc print_summary)
(take 10 (py. mcmc get_samples))


;; #### Posterior Distribution over the Regression Parameters;

