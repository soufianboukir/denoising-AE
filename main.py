# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import random

# dataset = torchvision.datasets.FashionMNIST(
#     root="./data",
#     train=True,
#     transform=transforms.ToTensor(),
#     download=True
# )

# idx = random.randint(0, len(dataset))
# image, label = dataset[idx]

# classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
#            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(f"Label: {classes[label]}")
# plt.axis("off")
# plt.show()




import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.cnn_autoencoder import CNNAutoencoder
from utils.noise import add_gaussian_noise

# ── Config ────────────────────────────────────────────────────────
EPOCHS       = 15
BATCH_SIZE   = 128
LR           = 1e-3
NOISE_FACTOR = 0.3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ──────────────────────────────────────────────────────────
transform = transforms.ToTensor()
dataset   = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True
)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ── Train function ────────────────────────────────────────────────
def train(optimizer_name):
    model     = CNNAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=LR)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    losses = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, _ in loader:
            images = images.to(DEVICE)
            noisy  = add_gaussian_noise(images, NOISE_FACTOR)

            outputs = model(noisy)
            loss    = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"{optimizer_name} | Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    return losses

# ── Run all optimizers ────────────────────────────────────────────
optimizers = ["Adam", "SGD"]
all_losses = {}

for opt in optimizers:
    print(f"\n{'='*40}")
    print(f"  Training with {opt}")
    print(f"{'='*40}")
    all_losses[opt] = train(opt)

# ── Plot ──────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
for opt_name, losses in all_losses.items():
    plt.plot(losses, marker="o", linewidth=2, label=opt_name)

plt.title("Optimizer Comparison — Training Loss", fontsize=13)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ── Final scores ──────────────────────────────────────────────────
print(f"\n{'='*40}")
print(f"  Final Loss Comparison")
print(f"{'='*40}")
for opt_name, losses in all_losses.items():
    print(f"{opt_name:<10} → Final Loss: {losses[-1]:.4f}")