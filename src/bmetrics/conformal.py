import numpy as np
import matplotlib.pyplot as plt


def within_interval(y: float, lower: float, upper) -> bool:
    return lower <= y <= upper


def bin_to_indices(bin_indices, bin_edges):
    binned_samples = {i: [] for i in range(1, len(bin_edges))}
    for idx, bin_index in enumerate(bin_indices):
        if bin_index in binned_samples:  # Ensure the index is within range
            binned_samples[bin_index].append(idx)
    return binned_samples


def get_binned_indices(data, bins):
    _, bin_edges = np.histogram(data, bins=bins)
    bin_indices = np.digitize(data, bin_edges)
    binned_indices = bin_to_indices(bin_indices, bin_edges)
    return binned_indices


def get_bin_coverage(bin, y, lower, upper):
    num_in_bin = len(bin)
    num_in_bounds = 0
    for i in bin:
        if within_interval(y=y[i], lower=lower[i], upper=upper[i]):
            num_in_bounds += 1
    if num_in_bin == 0:
        pass
    else:
        coverage = num_in_bounds / num_in_bin
        return coverage


def fsc_metric(binned_indices, y, lower, upper):
    """
    Feature Stratified Coverage Metric
    """
    coverages = []
    for bin in binned_indices.values():
        coverage = get_bin_coverage(bin, y=y, lower=lower, upper=upper)
        coverages.append(coverage)
    return np.min(coverages)


# Generate 1000 random data points from a normal distribution
median_data = np.random.normal(0, 1, 1000)
lower_data = np.random.normal(-1, 0.5, 1000)
upper_data = np.random.normal(1, 0.5, 1000)
data_3d = np.stack((lower_data, median_data, upper_data), axis=1)
data = data_3d[:, 1]
bins = 10
binned_indices = get_binned_indices(data, bins=10)
fsc = fsc_metric(binned_indices, y=median_data, lower=lower_data, upper=upper_data)
print(fsc)
plt.hist(data, bins=bins, alpha=0.5)
plt.show()
