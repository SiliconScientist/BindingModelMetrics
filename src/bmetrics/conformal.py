import torch


def within_interval(y: float, lower_bound: float, upper_bound) -> bool:
    return lower_bound <= y <= upper_bound


x = torch.tensor([1, 2.0, 1, 8, 6, 6, 8, 12])
hist = torch.histogram(x, bins=4, range=(0.0, 10.0))
print(hist.hist)
print(hist.bin_edges)

# y = [2.4, 3.5, 4.5, 5.6, 6.7, 7.8]
# n = 2

# lower_bound = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

# upper_bound = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

# chunks = [y[i : i + n] for i in range(0, len(y), n)]
# print(chunks)
# bool_list = [
#     within_interval(y[i], lower_bound[i], upper_bound[i]) for i in range(len(y))
# ]
# print(sum(bool_list) / len(bool_list))
