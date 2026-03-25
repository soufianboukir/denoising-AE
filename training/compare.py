import torch
import torchvision
import torchvision.transforms as transforms

from configs.config import Config
from models.cnn_autoencoder import CNNAutoencoder
from models.mlp_autoencoder import MLPAutoencoder
from utils.noise import add_gaussian_noise
from utils.metrices import compute_all_metrics
from utils.visualization import plot_metrics_by_sigma, show_images


SIGMAS      = [0.1, 0.3, 0.5, 1.0]
MODEL_PATHS = {
    "CNN": "./saved_model_cnn.pth",
    "MLP": "./saved_model_mlp.pth",
}


def load_model(model_type, path, device):
    model = CNNAutoencoder() if model_type == "CNN" else MLPAutoencoder()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def compare():
    device    = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()

    test_dataset = torchvision.datasets.FashionMNIST(
        root=Config.data_path, train=False, transform=transform, download=True
    )
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    images, _ = next(iter(loader))
    images = images.to(device)

    results = {
        "CNN": {"MSE": [], "PSNR": [], "SSIM": [], "MAE": []},
        "MLP": {"MSE": [], "PSNR": [], "SSIM": [], "MAE": []},
    }

    print(f"\n{'='*65}")
    print(f"  CNN vs MLP — Comparison across noise levels")
    print(f"{'='*65}")
    print(f"{'Model':<6} {'Sigma':<8} {'MSE':<10} {'PSNR':<10} {'SSIM':<10} {'MAE':<10}")
    print(f"{'-'*65}")

    for model_name, path in MODEL_PATHS.items():
        model = load_model(model_name, path, device)

        for sigma in SIGMAS:
            noisy = add_gaussian_noise(images, sigma)

            with torch.no_grad():
                outputs = model(noisy)

            metrics = compute_all_metrics(images, outputs)

            for key in metrics:
                results[model_name][key].append(metrics[key])

            print(f"{model_name:<6} {sigma:<8.1f} "
                  f"{metrics['MSE']:<10.4f} "
                  f"{metrics['PSNR']:<10.2f} "
                  f"{metrics['SSIM']:<10.4f} "
                  f"{metrics['MAE']:<10.4f}")

            show_images(images, noisy, outputs,
                        sigma=sigma)

        print(f"{'-'*65}")

    print(f"{'='*65}\n")
    plot_metrics_by_sigma(results)


if __name__ == "__main__":
    compare()