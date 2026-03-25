import torch

def add_gaussian_noise(images, noise_factor):
    noise = torch.randn_like(images) * noise_factor
    noisy = images + noise
    return torch.clamp(noisy, 0., 1.)
