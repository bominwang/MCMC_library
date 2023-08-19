import numpy as np
# import cupy as cp
import scipy.stats


# MetropolisHastings
class MetropolisHastings(object):
    def __init__(self, target_distribution, proposal_distribution, transfer_method='block-wise', gpu_flag=False):

        self.target_distribution = target_distribution
        self.proposal_distribution = proposal_distribution
        self.transfer_method = transfer_method
        self.gpu_flag = gpu_flag
        self.device(self.gpu_flag)

    def device(self, gpu_flag):
        # Running in CPU\GPU
        if gpu_flag:
            self.library = cp
        else:
            self.library = np
        return self.library

    def check_numpy_or_cupy(self, x):
        if self.gpu_flag:
            if isinstance(x, cp.ndarray):
                return x.astype(float)
            else:
                return cp.asarray(x, dtype=float)
        else:
            if isinstance(x, np.ndarray):
                return x.astype(float)
            else:
                raise ValueError('Unsupported array type')

    def check_hyperparameters(self, x, hyperparameter):
        size, dim = self.library.shape(x)
        if isinstance(hyperparameter, float):
            hyperparameter = self.library.ones(shape=[1, dim]) * hyperparameter
            return hyperparameter
        elif isinstance(hyperparameter, int):
            hyperparameter = float(hyperparameter)
            hyperparameter = self.library.ones(shape=[1, dim]) * hyperparameter
            return hyperparameter
        else:
            if self.gpu_flag:
                if isinstance(hyperparameter, cp.ndarray):
                    if self.library.shape(hyperparameter) == (1, dim):
                        return hyperparameter
                    else:
                        raise ValueError('Hyperparameter Dimension Mismatch')
            else:
                if isinstance(hyperparameter, np.ndarray):
                    if self.library.shape(hyperparameter) == (1, dim):
                        return hyperparameter
                    else:
                        raise ValueError('Hyperparameter Dimension Mismatch')

    def block_wise_conditional_distribution(self, x, width, sigma):
        # block-wise transfer
        # return a conditional_state [size, dim]
        size, dim = self.library.shape(x)
        conditional_state = self.library.zeros_like(x)
        if self.proposal_distribution == 'uniform':
            width = width.reshape(1, dim)
            delta = width * self.library.ones_like(x)
            low = x - delta
            high = x + delta
            conditional_state = self.library.random.uniform(low=low, high=high, size=[size, dim])
        elif self.proposal_distribution == 'normal':
            sigma = sigma.reshape(1, dim)
            conditional_state = self.library.random.normal(x, sigma, size=[size, dim])
        return conditional_state

    def component_wise_conditional_distribution(self, x, width, sigma):
        # component-wise transfer
        # x[num, 1] width&sigma[1]
        conditional_state = self.library.zeros_like(x)
        if self.proposal_distribution == 'uniform':
            conditional_state = self.library.random.uniform(x - width, x + width, size=self.library.shape(x))
        elif self.proposal_distribution == 'normal':
            conditional_state = self.library.random.normal(x, sigma * self.library.ones_like(x),
                                                           size=self.library.shape(x))
        return conditional_state

    def multivariate_gaussian_logpdf(self, x, mu, sigma):
        size, dim = self.library.shape(x)
        sigma = sigma.reshape(1, dim)
        logpdf = self.library.zeros([size, 1])
        covariance_matrix = self.library.diagflat(sigma)
        for i in range(size):
            logpdf[i, :] = scipy.stats.multivariate_normal.logpdf(x[i, :], mean=mu[i, :], cov=covariance_matrix)
        return logpdf

    def calculate_acceptance_ratio(self, current_state, conditional_state, sigma):
        num, dim = self.library.shape(current_state)
        alpha_1 = self.target_distribution(conditional_state) - self.target_distribution(current_state)
        alpha_1 = alpha_1.reshape(num, 1)
        alpha_2 = None
        if self.proposal_distribution == 'uniform':
            alpha_2 = 0
        elif self.proposal_distribution == 'normal':
            alpha_2_0 = self.multivariate_gaussian_logpdf(current_state, conditional_state, sigma)
            alpha_2_1 = - self.multivariate_gaussian_logpdf(conditional_state, current_state, sigma)
            alpha_2 = alpha_2_1 + alpha_2_0
        alpha = alpha_1 + alpha_2
        return alpha

    def metropolis_hastings_iterator(self, current_state, width, sigma):
        if self.transfer_method == 'block-wise':
            conditional_state = self.block_wise_conditional_distribution(current_state, width, sigma)
            acceptance_ratio = self.calculate_acceptance_ratio(current_state, conditional_state, sigma)
            u = self.library.random.random(acceptance_ratio.shape)
            condition = (acceptance_ratio >= 0) | (self.library.exp(acceptance_ratio) > u)
            acceptance_index = self.library.where(condition)[0]
            current_state[acceptance_index, :] = conditional_state[acceptance_index, :]
        elif self.transfer_method == 'component-wise':
            size, dim = self.library.shape(current_state)
            for component_index in range(dim):
                conditional_state = current_state.copy()
                component_state = current_state[:, component_index].reshape(-1, 1)
                component_state = self.component_wise_conditional_distribution(component_state,
                                                                               width[0, component_index],
                                                                               sigma[0, component_index])
                conditional_state[:, component_index] = component_state.squeeze()
                acceptance_ratio = self.calculate_acceptance_ratio(current_state, conditional_state, sigma)
                u = self.library.random.random(acceptance_ratio.shape)
                condition = (acceptance_ratio >= 0) | (self.library.exp(acceptance_ratio) > u)
                acceptance_index = self.library.where(condition)[0]
                current_state[acceptance_index, :] = conditional_state[acceptance_index, :]
        else:
            raise ValueError("Transfer method is not correctly specified. Program terminated.")
        return current_state

    def run(self, initial_position, steps_warm_up, num_samples, width=1.0, sigma=1.0):
        size, dim = self.library.shape(initial_position)
        chains = self.library.zeros([num_samples, size, dim])
        initial_position = self.check_numpy_or_cupy(initial_position)
        width = self.check_hyperparameters(initial_position, width)
        sigma = self.check_hyperparameters(initial_position, sigma)
        all_steps = steps_warm_up + num_samples
        current_state = initial_position

        print("Running warm-up phase:")
        for _ in range(steps_warm_up):
            conditional_state = self.metropolis_hastings_iterator(current_state=current_state, width=width, sigma=sigma)
            current_state = conditional_state
            progress = int((_ + 1) / steps_warm_up * 20)
            print("\r[{}{}] {:.2f}%".format("=" * progress, " " * (20 - progress), (_ + 1) / steps_warm_up * 100),
                  end="", flush=True)
        print()

        print("Running sampling phase:")
        for i in range(num_samples):
            conditional_state = self.metropolis_hastings_iterator(current_state=current_state, width=width, sigma=sigma)
            current_state = conditional_state
            chains[i, :, :] = conditional_state
            progress = int((i + 1) / num_samples * 20)
            print("\r[{}{}] {:.2f}%".format("=" * progress, " " * (20 - progress), (i + 1) / num_samples * 100), end="",
                  flush=True)
        print()

        return chains
