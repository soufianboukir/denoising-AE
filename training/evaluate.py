import torch
import torchvision
import torchvision.transforms as transforms

from configs.config import Config
from utils.noise import add_gaussian_noise
from utils.visualization import show_images


def evaluate():

    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()

    test_dataset = torchvision.datasets.FashionMNIST(
        root=Config.data_path,
        train=False,
        transform=transform,
        download=True
    )

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=6)

    model = torch.load(Config.model_path, map_location=device)
    model.eval()

    images, _ = next(iter(loader))
    images = images.to(device)

    noisy = add_gaussian_noise(images, Config.noise_factor)

    with torch.no_grad():
        outputs = model(noisy)

    show_images(images, noisy, outputs)


if __name__ == "__main__":
    evaluate()