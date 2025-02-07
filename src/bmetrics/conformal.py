import numpy as np
import matplotlib.pyplot as plt


def within_interval(y: float, lower_bound: float, upper_bound) -> bool:
    return lower_bound <= y <= upper_bound


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


def get_bin_coverage(bin, data):
    num_in_bin = len(bin)
    num_in_bounds = 0
    for i in bin:
        if within_interval(data[i][1], data[i][0], data[i][2]):
            num_in_bounds += 1
    if num_in_bin == 0:
        pass
    else:
        coverage = num_in_bounds / num_in_bin
        return coverage


def fsc_metric(binned_indices, data):
    """
    Feature Stratified Coverage Metric
    """
    coverages = []
    for bin in binned_indices.values():
        coverage = get_bin_coverage(bin, data)
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
fsc = fsc_metric(binned_indices, data_3d)
print(fsc)
plt.hist(data, bins=bins, alpha=0.5)
plt.show()
