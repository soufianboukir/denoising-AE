import matplotlib.pyplot as plt

def show_images(original, noisy, reconstructed):

    original = original.cpu().detach()
    noisy = noisy.cpu().detach()
    reconstructed = reconstructed.cpu().detach()

    fig, axes = plt.subplots(3, 6, figsize=(10,5))

    for i in range(6):
        axes[0,i].imshow(original[i].squeeze(), cmap="gray")
        axes[0,i].axis("off")

        axes[1,i].imshow(noisy[i].squeeze(), cmap="gray")
        axes[1,i].axis("off")

        axes[2,i].imshow(reconstructed[i].squeeze(), cmap="gray")
        axes[2,i].axis("off")

    axes[0,0].set_ylabel("Original")
    axes[1,0].set_ylabel("Noisy")
    axes[2,0].set_ylabel("Reconstructed")

    plt.show()


def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()