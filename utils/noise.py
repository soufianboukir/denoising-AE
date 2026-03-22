import torch

def add_gaussian_noise(images, noise_factor):
    noise = torch.randn_like(images) * noise_factor
    noisy = images + noise
    return torch.clamp(noisy, 0., 1.)


def add_salt_pepper_noise(images, prob=0.1):
    noisy = images.clone()
    rand = torch.rand_like(images)

    noisy[rand < prob/2] = 0
    noisy[rand > 1 - prob/2] = 1

    return noisy