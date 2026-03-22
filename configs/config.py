class Config:
    batch_size = 128
    learning_rate = 1e-3
    epochs = 15
    noise_factor = 0.3

    device = "cuda"

    model_type = "cnn"  # "cnn" or "mlp"

    data_path = "./data"
    model_path = "./saved_model.pth"