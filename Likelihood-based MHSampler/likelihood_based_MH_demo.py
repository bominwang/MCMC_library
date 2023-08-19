import numpy as np
import scipy.stats
from Likelihood_based_MH import MetropolisHastings
import matplotlib.pyplot as plt

"""
sample for mixture normal
"""


class MixtureNormal(object):
    def __init__(self, log_flag):
        self.log_flag = log_flag

    def pdf(self, x):
        prob1 = np.exp(-0.5 * ((x - 0) ** 2) / 1) / np.sqrt(2 * np.pi * 1)
        prob2 = np.exp(-0.5 * ((x - 5) ** 2) / 2) / np.sqrt(2 * np.pi * 2)
        mixture_prob = 0.6 * prob1 + 0.4 * prob2
        if self.log_flag:
            mixture_prob = np.log(mixture_prob)
        return mixture_prob


class MixtureNormal2d(object):
    def __init__(self, log_flag):
        self.log_flag = log_flag
        self.mean1 = np.array([3, 2])
        self.cov1 = np.array([[1, 0], [0, 1]])  # Covariance matrix
        self.mean2 = np.array([5, 6])
        self.cov2 = np.array([[1, 0], [0, 1]])  # Covariance matrix
        self.mixture_ratio = np.array([0.5, 0.5])

    def pdf(self, x):
        prob1 = scipy.stats.multivariate_normal(self.mean1, self.cov1).pdf(x)
        prob2 = scipy.stats.multivariate_normal(self.mean2, self.cov2).pdf(x)
        mixture_prob = self.mixture_ratio[0] * prob1 + self.mixture_ratio[1] * prob2

        if self.log_flag:
            if self.log_flag:
                mixture_prob = np.log(mixture_prob + 1e-10)

        return mixture_prob

def complex_mixture_pdf(x):
    component_means = np.array([[2, 2], [6, 6], [10, 2]])
    component_covs = np.array([[[1, 0.5], [0.5, 1]],[[1, -0.3], [-0.3, 1]],[[1, 0], [0, 1]]])
    component_weights = np.array([0.4, 0.3, 0.3])
    logpdf_vals = [np.log(weight) + scipy.stats.multivariate_normal(mean, cov).logpdf(x) for weight, mean, cov in zip(component_weights,
                                                                                                                      component_means,
                                                                                                                      component_covs)]
    total_logpdf = np.logaddexp.reduce(logpdf_vals)
    return total_logpdf


if __name__ == '__main__':

    """
    1-d mixture gaussian sampling
    """

    target_distribution = MixtureNormal(log_flag=True).pdf
    initial_state = np.array([[1], [2], [3], [4]]).reshape(4, 1)
    samples = MetropolisHastings(target_distribution=target_distribution,
                                 proposal_distribution='uniform',
                                 transfer_method='component-wise',
                                 gpu_flag=False).run(initial_position=initial_state,
                                                     steps_warm_up=10000,
                                                     num_samples=10000,
                                                     width=1.0, sigma=1.0)

    x_values = np.linspace(-10, 15, 10000)
    true_pdf_values = MixtureNormal(log_flag=False).pdf(x_values)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(x_values, true_pdf_values, label='True PDF')
        ax.hist(samples[:10000, i, 0], bins=50, density=True, alpha=0.5,
                label=f'Generated Samples (Histogram) i={i + 1}')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability Density')
        # ax.set_title(f'Comparison of True PDF and Generated Samples (Histogram) i={i + 1}')
        ax.legend()

    plt.tight_layout()
    plt.show()

    """
    2-d mixture gaussian sampling
    """
    target_distribution = MixtureNormal2d(log_flag=True).pdf
    initial_state = np.array([[0, 0], [3, 0], [0, 4], [2, 3]]).reshape(4, -1)
    width = np.array([1.0, 2.0]).reshape(1, 2)
    sigma = np.array([1.0, 2.0]).reshape(1, 2)
    samples = MetropolisHastings(target_distribution=target_distribution,
                                 proposal_distribution='normal',
                                 transfer_method='component-wise',
                                 gpu_flag=False).run(initial_position=initial_state,
                                                     steps_warm_up=1000,
                                                     num_samples=1000,
                                                     width=width,
                                                     sigma=sigma)
    mixture = MixtureNormal2d(log_flag=False)
    x = np.linspace(-2, 10, 300)
    y = np.linspace(-2, 10, 300)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    pdf_values = mixture.pdf(np.vstack([X.ravel(), Y.ravel()]).T).reshape(X.shape)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i, ax in enumerate(axes.flatten()):
        contour = ax.contourf(X, Y, pdf_values, levels=20, cmap='viridis', alpha=0.4)
        ax.scatter(samples[:, i, 0], samples[:, i, 1], s=10, color='red', alpha=1,label=f'Generated Samples i={i + 1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_title(f'2D Mixture Gaussian PDF (Contour) and Generated Samples (Scatter) i={i + 1}')
        ax.legend()
    plt.tight_layout()
    plt.show()

    """
    2d mixture gaussian
    """

    initial_state = np.array([[0, 0], [3, 0], [0, 4], [2, 3]]).reshape(4, -1)
    width = np.array([1.0, 2.0]).reshape(1, 2)
    sigma = np.array([1.0, 2.0]).reshape(1, 2)
    samples = MetropolisHastings(target_distribution=complex_mixture_pdf,
                                 proposal_distribution='uniform',
                                 transfer_method='component-wise',
                                 gpu_flag=False).run(initial_position=initial_state,
                                                     steps_warm_up=1000,
                                                     num_samples=1000,
                                                     width=width,
                                                     sigma=sigma)
    x, y = np.meshgrid(np.linspace(-2, 14, 300), np.linspace(-2, 10, 300))
    pos = np.dstack((x, y))
    pdf_values = complex_mixture_pdf(pos)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, ax in enumerate(axes.flatten()):
        contour = ax.contourf(x, y, pdf_values, levels=20, cmap='viridis', alpha=0.4)
        ax.scatter(samples[:, i, 0], samples[:, i, 1], s=10, color='red', alpha=1, label=f'Generated Samples i={i + 1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        # ax.set_title(f'Complex 2D Mixture Distribution Contour Plot (Dimension {i+1})')
    plt.tight_layout()
    plt.show()

    """
    Moving Trajectory
    """
    positions = samples[:, 0, :].squeeze()
    plt.figure(figsize=(8, 6))
    plt.contourf(x, y, pdf_values, levels=20, cmap='viridis', alpha=0.4)
    plt.plot(positions[:, 0], positions[:, 1], marker='o', linestyle='-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Moving Trajectory')
    plt.grid(True)
    plt.show()
