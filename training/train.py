import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import Config
from training.trainer import train_one_epoch
from utils.visualization import plot_losses

from models.cnn_autoencoder import CNNAutoencoder
from models.mlp_autoencoder import MLPAutoencoder

import torchvision
import torchvision.transforms as transforms


def get_model():
    if Config.model_type == "cnn":
        return CNNAutoencoder()
    else:
        return MLPAutoencoder()


def train():

    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.FashionMNIST(
        root=Config.data_path,
        train=True,
        transform=transform,
        download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True
    )

    model = get_model().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    losses = []

    for epoch in range(Config.epochs):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, Config.noise_factor
        )

        losses.append(loss)
        print(f"Epoch {epoch+1}: {loss:.4f}")

    torch.save(model.state_dict(), Config.model_path)
    plot_losses(losses)


if __name__ == "__main__":
    train()