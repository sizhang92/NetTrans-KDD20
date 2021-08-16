import torch


def remove_by_threshold(index, value, threshold):

    mask = (value >= threshold)
    row, col = index
    value = value[mask]
    row, col = row[mask], col[mask]

    return torch.stack([row, col]), value
