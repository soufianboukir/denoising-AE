import torch

def mse(x, y):
    return ((x - y) ** 2).mean().item()


def psnr(x, y):
    mse_val = ((x - y) ** 2).mean()
    return 10 * torch.log10(1 / mse_val).item()