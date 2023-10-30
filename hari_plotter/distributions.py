import numpy as np


def generate_mixture_of_gaussians(n_samples, number_of_peaks, opinion_limits=(-1, 1),
                                  mean_opinion=0.5, size_of_each_peak=None, seed=None):
    """
    Generates a mixture of Gaussians.

    :param n_samples: int, number of samples to be drawn.
    :param number_of_peaks: int, number of peaks or Gaussian components.
    :param opinion_limits: tuple(float, float), range of the opinions.
    :param mean_opinion: float, mean opinion.
    :param size_of_each_peak: List[int], size of each peak.
    :param seed: int, random seed.
    :return: np.array, generated opinions.
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate the means for each Gaussian component
    component_means = np.linspace(
        opinion_limits[0], opinion_limits[1], number_of_peaks)

    # Normalize the size of each peak to sum to 1
    weights = np.array(size_of_each_peak, dtype=float)
    weights /= weights.sum()

    # Adjust the component means to ensure the overall mean is mean_opinion
    delta = mean_opinion - np.dot(weights, component_means)
    component_means += delta

    # Generate samples from the mixture of Gaussians
    samples = []
    for i in range(number_of_peaks):
        n = int(weights[i] * n_samples)
        samples.extend(np.random.normal(component_means[i], 0.1, n))

    return np.array(samples)
