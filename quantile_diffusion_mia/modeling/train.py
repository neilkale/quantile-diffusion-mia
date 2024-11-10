from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from quantile_diffusion_mia.config import MODELS_DIR, PROCESSED_DATA_DIR, DATASET_CONFIG, MODEL_CONFIG
from quantile_diffusion_mia.dataset import QuantileRegressionDataset
from quantile_diffusion_mia.modeling.resnet import ResNet18

app = typer.Typer()

@app.command()
def create_quantile_model(dataset_name: str, alpha: float = 0.01, device: str = 'cuda', random_seed: int = 42):
    config = DATASET_CONFIG[dataset_name]

    # Load the train indices
    logger.info(f"Loading train indices for {dataset_name}")
    split_path = config['quantile_regression_split_path']
    output = np.load(split_path)
    quantile_train_indices = output['quantile_train_indices']
    logger.success("Train indices loaded successfully.")

    # Load the dataset
    data_path = config['quantile_regression_data_path']
    logger.info(f"Loading quantile regresion dataset from {data_path}")
    train_dataset = QuantileRegressionDataset.load(data_path)
    train_dataset.set_indices(quantile_train_indices)
    logger.success("Quantile regression dataset loaded successfully.")

    # Train the quantile regressor
    model = train_quantiles(dataset_name, train_dataset, alpha=alpha, device=device, random_seed=random_seed)

    # Save the model weights
    model_path = config['quantile_regression_model_path'].format(alpha=alpha)
    logger.info(f"Saving quantile regressor to {model_path}")    
    torch.save(model.state_dict(), model_path)
    logger.success("Quantile regressor saved successfully.")

def load_quantile_model(model_path, alpha=0.01, device='cuda'):
    model = ResNet18(in_channels=6, num_classes=1).to(device)
    model_path = model_path.format(alpha=alpha)
    if not Path(model_path).exists():
        logger.error(f"Model path {model_path} does not exist. Train the quantile model with alpha={alpha} first.")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model

def train_quantiles(dataset_name, train_dataset, alpha: float = 0.01, device="cuda", random_seed=42):
    config = MODEL_CONFIG[f'{dataset_name}_QUANTILE']
    # Set the random seed for reproducibility
    torch.manual_seed(random_seed)

    # Create a DataLoader for the training dataset
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    # Define a quantile loss function
    def quantile_loss(predictions, targets, q):
        errors = targets - predictions
        return torch.max((q - 1) * errors, q * errors).mean()
        # return torch.mean(torch.abs(predictions - targets))        


    # Define the model
    model = ResNet18(in_channels=6, num_classes=1).to(device)

    # Define the optimizer
    optim = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)

    # Train the model
    logger.info("Training quantile regressor...")
    model.train()
    for epoch in tqdm(range(config['n_epochs'])):
        
        running_loss = 0.0
        for i, (original_image, reconstructed_image, targets, _) in tqdm(enumerate(train_loader), leave=False):
            features = torch.cat((original_image, reconstructed_image), dim=1).to(device)
            targets = targets.to(device) * 10 ** 10

            # Forward pass
            predictions = model(features).squeeze()

            quantile_losses = quantile_loss(predictions, targets, alpha)

            # Backward pass
            optim.zero_grad()
            quantile_losses.backward()
            optim.step()

            running_loss += quantile_losses.item()
        logger.info(f"Epoch {epoch + 1} loss: {running_loss / len(train_loader)}")
    logger.success("Training complete.")

    return model




@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()