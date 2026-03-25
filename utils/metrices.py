import torch
from pytorch_msssim import ssim

def mse(x, y):
    return ((x - y) ** 2).mean().item()

def psnr(x, y):
    mse_val = ((x - y) ** 2).mean()
    if mse_val == 0:
        return float('inf')
    return 10 * torch.log10(1 / mse_val).item()

def ssim_score(x, y):
    return ssim(x, y, data_range=1.0).item()

def mae(x, y):
    return (x - y).abs().mean().item()

def compute_all_metrics(original, reconstructed):
    return {
        "MSE":  mse(original, reconstructed),
        "PSNR": psnr(original, reconstructed),
        "SSIM": ssim_score(original, reconstructed),
        "MAE":  mae(original, reconstructed),
    }