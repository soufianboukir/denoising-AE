import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

idx = random.randint(0, len(dataset))
image, label = dataset[idx]

classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {classes[label]}")
plt.axis("off")
plt.show()
