import torch
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


# def count_samples_in_interval(data, lower_bound, upper_bound):


# Generate 1000 random data points from a normal distribution
median_data = np.random.normal(0, 1, 1000)
lower_data = np.random.normal(-1, 0.5, 1000)
upper_data = np.random.normal(1, 0.5, 1000)
data = np.stack((lower_data, median_data, upper_data), axis=1)
# Create the histogram with 50 bins
counts, bin_edges = np.histogram(data[:, 1], bins=10)
# Extract indices of each sample in each bin
bin_indices = np.digitize(data[:, 1], bin_edges)

binned_indices = bin_to_indices(bin_indices, bin_edges)

bin_coverages = []
for bin in binned_indices.values():
    num_in_bin = len(bin)
    num_in_bounds = 0
    for i in bin:
        if within_interval(data[i][1], data[i][0], data[i][2]):
            num_in_bounds += 1
    if num_in_bin == 0:
        pass
    else:
        bin_coverages.append(num_in_bounds / num_in_bin)

print(bin_coverages)
# Plot the histogram
plt.hist(data[:, 1], bins=bin_edges, alpha=0.5)
plt.show()
