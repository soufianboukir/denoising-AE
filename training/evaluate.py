import torch
import torchvision
import torchvision.transforms as transforms

from configs.config import Config
from models.cnn_autoencoder import CNNAutoencoder
from models.mlp_autoencoder import MLPAutoencoder
from utils.noise import add_gaussian_noise
from utils.visualization import show_images, show_error_map, plot_histogram
from utils.metrices import compute_all_metrics


SIGMAS = [0.1, 0.3, 0.5, 1.0]


def get_model():
    if Config.model_type == "cnn":
        return CNNAutoencoder()
    return MLPAutoencoder()


def evaluate():
    device    = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    test_dataset = torchvision.datasets.FashionMNIST(
        root=Config.data_path, train=False, transform=transform, download=True
    )
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = get_model()
    model.load_state_dict(torch.load(Config.model_path, map_location=device))
    model.to(device)
    model.eval()

    images, _ = next(iter(loader))
    images = images.to(device)

    print(f"\n{'='*55}")
    print(f"  Model: {Config.model_type.upper()} — Evaluation across noise levels")
    print(f"{'='*55}")
    print(f"{'Sigma':<8} {'MSE':<10} {'PSNR':<10} {'SSIM':<10} {'MAE':<10}")
    print(f"{'-'*55}")

    for sigma in SIGMAS:
        noisy = add_gaussian_noise(images, sigma)

        with torch.no_grad():
            outputs = model(noisy)

        metrics = compute_all_metrics(images, outputs)

        print(f"{sigma:<8.1f} "
              f"{metrics['MSE']:<10.4f} "
              f"{metrics['PSNR']:<10.2f} "
              f"{metrics['SSIM']:<10.4f} "
              f"{metrics['MAE']:<10.4f}")

        show_images(images, noisy, outputs, sigma=sigma)
        show_error_map(images, outputs)
        plot_histogram(images, outputs, sigma=sigma)

    print(f"{'='*55}\n")


if __name__ == "__main__":
    evaluate()