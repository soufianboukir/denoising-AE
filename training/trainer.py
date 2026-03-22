import torch
from utils.noise import add_gaussian_noise

def train_one_epoch(model, loader, optimizer, criterion, device, noise_factor):

    model.train()
    total_loss = 0

    for images, _ in loader:

        images = images.to(device)
        noisy = add_gaussian_noise(images, noise_factor)

        outputs = model(noisy)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device, noise_factor):

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy = add_gaussian_noise(images, noise_factor)

            outputs = model(noisy)
            loss = ((outputs - images) ** 2).mean()

            total_loss += loss.item()

    return total_loss / len(loader)