import matplotlib.pyplot as plt
import torch

def show_images(original, noisy, reconstructed, sigma=None):
    original      = original.cpu().detach()
    noisy         = noisy.cpu().detach()
    reconstructed = reconstructed.cpu().detach()

    n = min(6, original.size(0))
    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6))
    title = f"σ = {sigma}" if sigma is not None else ""
    fig.suptitle(title, fontsize=13)

    row_labels = ["Original", "Noisy", "Reconstructed"]
    for i in range(n):
        for row, img in enumerate([original, noisy, reconstructed]):
            axes[row, i].imshow(img[i].squeeze(), cmap="gray")
            axes[row, i].axis("off")
        axes[0, i].set_title(f"#{i+1}", fontsize=9)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, val_losses=None):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    if val_losses:
        plt.plot(val_losses, label="Validation Loss", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_error_map(original, reconstructed, n=6):
    original      = original.cpu().detach()
    reconstructed = reconstructed.cpu().detach()
    error         = (original - reconstructed).abs()

    fig, axes = plt.subplots(1, n, figsize=(n * 2, 3))
    fig.suptitle("Pixel Error Map (|original - reconstructed|)", fontsize=12)
    for i in range(min(n, original.size(0))):
        im = axes[i].imshow(error[i].squeeze(), cmap="hot", vmin=0, vmax=0.5)
        axes[i].axis("off")
    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_metrics_by_sigma(results: dict):
    sigmas  = [0.1, 0.3, 0.5, 1.0]
    metrics = ["MSE", "PSNR", "SSIM", "MAE"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("CNN vs MLP — Metrics by Noise Level", fontsize=14)
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for model_name, values in results.items():
            ax.plot(sigmas, values[metric], marker="o", label=model_name, linewidth=2)
        ax.set_title(metric)
        ax.set_xlabel("Noise σ")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_histogram(original, reconstructed, sigma=None):
    original      = original.cpu().detach().numpy().flatten()
    reconstructed = reconstructed.cpu().detach().numpy().flatten()

    plt.figure(figsize=(8, 4))
    plt.hist(original,      bins=50, alpha=0.6, label="Original",      color="steelblue")
    plt.hist(reconstructed, bins=50, alpha=0.6, label="Reconstructed", color="coral")
    title = f"Pixel Distribution — σ={sigma}" if sigma else "Pixel Distribution"
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()