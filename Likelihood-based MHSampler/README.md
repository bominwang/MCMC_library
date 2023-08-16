# **Markov Chain Monte Carlo algorithm (MH)**

The most classical implementation of Markov chain Monte Carlo sampling
This library is primarily employed for the implementation of the following two algorithms:

(1) Original Metropolis-Hastings algorithm

(2) Adaptive Metropolis-Hastings algorithm

Metropolis-Hastings is a Markov chain monte carlo algorithm often used for arbitrary distributions which are tricky to sample from.

According to Bayes' theorem, the posterior distribution (also referred to as the target distribution) is proportional to the product of the likelihood function and the prior distribution. Leveraging this property, the Metropolis-Hastings (MH) algorithm is capable of generating a Markov chain, wherein the stationary distribution is equivalent to the target distribution. This enables direct sampling from the posterior distribution.
