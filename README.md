# Denoising Autoencoder

This project implements a Denoising Autoencoder using PyTorch to remove noise from images. It features both Convolutional Neural Network (CNN) and Multi-Layer Perceptron (MLP) architectures.

## Features

- **Denoising Autoencoder**: Learns to reconstruct clean images from noisy inputs by injecting Gaussian noise.
- **PyTorch Backend**: Built with the modern deep learning library PyTorch.
- **Multiple Architectures**: Configurable to use either CNN or MLP autoencoders.
- **FashionMNIST Dataset**: Trained and evaluated on the FashionMNIST dataset.
- **Comprehensive Evaluation**: Metrics include MSE, PSNR, SSIM, and MAE across various noise levels ($\sigma$).
- **Interactive Web App**: Includes a Streamlit application to upload images, add adjustable noise, and visualize the denoised output in real-time.
- **Visualization Tools**: Generates comparisons of original, noisy, and reconstructed images, along with pixel error maps, training loss curves, and pixel distribution histograms.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/soufianboukir/denoising-AE.git
   cd denoising-AE
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Core requirements include `torch`, `torchvision`, `streamlit`, `matplotlib`, `numpy`, and `Pillow`)*

## Usage

### Configuration
You can modify the model type (`cnn` or `mlp`), batch size, learning rate, number of epochs, and default noise factor in `configs/config.py`.

### Training the Model
Run the following command to train the model on FashionMNIST:
```bash
python -m training.train
```

### Evaluating the Model
To evaluate the pre-trained model on the test dataset across different levels of noise:
```bash
python -m training.evaluate
```

### Running the Web App
To start the interactive Streamlit application (accessible via your browser):
```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
├── app/
│   └── streamlit_app.py        # Interactive Streamlit interface
├── configs/
│   └── config.py               # Hyperparameters and model settings
├── models/
│   ├── cnn_autoencoder.py      # CNN Autoencoder definition
│   └── mlp_autoencoder.py      # MLP Autoencoder definition
├── training/
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Evaluation script for metrics & plots
│   └── trainer.py              # Epoch training and validation logic
├── utils/
│   ├── metrices.py             # Functions to calculate MSE, PSNR, SSIM, MAE
│   ├── noise.py                # Gaussian noise injection utility
│   └── visualization.py        # Utility functions for plotting results
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── main.py                     # Simple script to visualize FashionMNIST dataset
```

## Credits

**Developed by:**
- SOUFIAN BOUKIR
- MOHAMED BELALIA
- MOUAD ELOUICHOUANI

**Supervised by:**
- Pr. Nasreddine HAFIDI
