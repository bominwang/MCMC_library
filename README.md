# MCMC_library: Likelihood-Based/Likelihood-Free Metropolis-Hastings Algorithm

Welcome to the MCMC_library repository, a pristine Python library that offers a comprehensive implementation of the Likelihood-Based and Likelihood-Free Metropolis-Hastings (MH) Algorithm.

## Likelihood-Based MH Sampler

### 1D Mixture Gaussian
![1D Mixture Gaussian](https://github.com/bominwang/MCMC_library/blob/main/Likelihood-based%20MHSampler/figure/mcmc1.png)

### 2D Mixture Gaussian
![2D Mixture Gaussian 1](https://github.com/bominwang/MCMC_library/blob/main/Likelihood-based%20MHSampler/figure/mcmc2.png)
![2D Mixture Gaussian 2](https://github.com/bominwang/MCMC_library/blob/main/Likelihood-based%20MHSampler/figure/mcmc3.png)
![2D Mixture Gaussian 3](https://github.com/bominwang/MCMC_library/blob/main/Likelihood-based%20MHSampler/figure/mcmc4.png)

## Getting Started

The power of the Metropolis-Hastings algorithm is at your fingertips with just a single command:

```python
# Define the target_distribution, proposal_distribution, and other parameters
# target_distribution: The target distribution you want to sample from.
#                     It should be a function that calculates the probability density of a given sample.
# proposal_distribution: The proposal distribution used to generate candidate samples from the current sample.
#                        It determines how new samples are explored in the space.
#                        Common choices include 'uniform' or 'normal'.
# transfer_method: The transfer method specifying how to accept or reject candidate samples generated from the proposal distribution.
#                  Common choices include 'component-wise' or 'block-wise'.
# gpu_flag: Whether to enable GPU acceleration. If set to True, the algorithm will utilize GPU for computation (if available).
samples = MetropolisHastings(target_distribution=target_distribution,
                             proposal_distribution='uniform',
                             transfer_method='component-wise',
                             gpu_flag=False).run(initial_position=initial_state,
                                                 steps_warm_up=10000,
                                                 num_samples=10000,
                                                 width=1.0, sigma=1.0)
