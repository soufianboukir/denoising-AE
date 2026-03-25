import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from configs.config import Config
from training.trainer import train_one_epoch, evaluate
from utils.visualization import plot_losses
from models.cnn_autoencoder import CNNAutoencoder
from models.mlp_autoencoder import MLPAutoencoder


def get_model():
    if Config.model_type == "cnn":
        return CNNAutoencoder()
    return MLPAutoencoder()


def train():
    device    = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.FashionMNIST(
        root=Config.data_path, train=True, transform=transform, download=True
    )
    val_dataset = torchvision.datasets.FashionMNIST(
        root=Config.data_path, train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=Config.batch_size, shuffle=False
    )

    model     = get_model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.learning_rate)

    train_losses = []
    val_losses   = []

    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, Config.noise_factor
        )
        val_loss = evaluate(model, val_loader, device, Config.noise_factor)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{Config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), Config.model_path)
    print(f"Model saved to {Config.model_path}")

    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    train()