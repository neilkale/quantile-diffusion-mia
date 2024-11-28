from pathlib import Path
import os

import typer
from loguru import logger
from tqdm import tqdm
from typing import List
import wandb
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from quantile_diffusion_mia.config import MODELS_DIR, PROCESSED_DATA_DIR, DATASET_CONFIG, MODEL_CONFIG
from quantile_diffusion_mia.dataset import QuantileRegressionDataset
from quantile_diffusion_mia.modeling.resnet import ResNet18
from quantile_diffusion_mia.utils import quantile_loss
from itertools import product

app = typer.Typer()
wandb.login()

def worker(task_id, params, dataset_name):
    """Worker function to train a single configuration."""
    lr, weight_decay, batch_size, target_alpha, num_alpha, optimizer, scheduler, resnet_channel_reduce = params

    # Update model config for this task
    MODEL_CONFIG[f'{dataset_name}_QUANTILE'].update({
        "num_attackers": 1,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "target_alpha": target_alpha,
        "num_quantiles": num_alpha,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "resnet_channel_reduce": resnet_channel_reduce
    })

    # Assign a specific GPU based on the task ID
    gpu_id = task_id % 2 + 2 # torch.cuda.device_count()
    torch.cuda.set_device(gpu_id)

    logger.info(f"Running task {task_id} on GPU {gpu_id} with config: {MODEL_CONFIG[f'{dataset_name}_QUANTILE']}")
    
    # Call your training function
    create_quantile_bag(dataset_name, retrain=True)
    return task_id

@app.command()
def hyperparameter_grid_search(dataset_name: str):
    param_grid = {
        "lr": [1e-3, 1e-4],
        "weight_decay": [1e-4, 1e-5],
        "batch_size": [32, 64],
        "target_alpha": [0.001, 0.005, 0.01],
        "num_alpha": [10, 50, 100],
        "optimizer": ['adam', 'adamw'],
        "scheduler": ['cosine', 'cosine_warm_restarts'],
        "resnet_channel_reduce": [1, 2, 4]
    }

    param_combinations = list(product(*param_grid.values()))

    # Parallel execution using ProcessPoolExecutor
    max_workers = min(20, len(param_combinations))  # Limit to 100 tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task_id, params in enumerate(param_combinations):
            futures.append(executor.submit(worker, task_id, params, dataset_name))
        
        for future in tqdm(futures, desc="Hyperparameter Grid Search"):
            future.result()  # Wait for each task to complete

    logger.info("Hyperparameter grid search completed.")

@app.command()
def create_quantile_bag(dataset_name: str, device: str = 'cuda', random_seed: int = 42, retrain: bool = False):
    config = DATASET_CONFIG[dataset_name]
    model_config = MODEL_CONFIG[f'{dataset_name}_QUANTILE']
    rng = np.random.default_rng(random_seed)

    # Create a list of alpha values
    alphas = np.logspace(np.log10(model_config['alpha_min']), np.log10(model_config['alpha_max']), model_config['num_quantiles'])
    alphas = torch.tensor(alphas).to(device)

    # Load the train indices
    logger.info(f"Loading train indices for {dataset_name}")
    split_path = config['quantile_regression_split_path']
    output = np.load(split_path)
    quantile_train_indices = output['quantile_train_indices']
    quantile_eval_indices = output['quantile_eval_indices']
    logger.success("Train indices loaded successfully.")

    # Load the dataset
    data_path = config['quantile_regression_data_path']
    logger.info(f"Loading quantile regression dataset from {data_path}")
    train_dataset = QuantileRegressionDataset.load(data_path)
    train_dataset.set_indices(quantile_train_indices)
    eval_dataset = QuantileRegressionDataset.load(data_path)
    eval_dataset.set_indices(quantile_eval_indices)
    logger.success("Quantile regression dataset loaded successfully.")

    # Train the quantile regressor
    logger.info("Training quantile regressors...")
    for i in tqdm(range(model_config['num_attackers']), desc="Attackers"):

        # Bootstrap the quantile_train_indices
        logger.info("Bootstrapping the quantile train indices")
        bootstrapped_indices = rng.choice(quantile_train_indices, size=len(quantile_train_indices), replace=True)
        train_dataset.set_indices(bootstrapped_indices)
        logger.success("Bootstrapping complete.")

        model_seed = random_seed + i
        model_path = config['quantile_regression_model_path'].format(alpha=f'{alphas[0]:.1e}to{alphas[-1]:.1e}', attacker=i)
        
        if not retrain and Path(model_path).exists():
            logger.info(f"Quantile regressor {i} already exists at {model_path}. Skipping training.")
            continue
        
        model = train_quantiles(dataset_name, train_dataset, alphas, eval_dataset=eval_dataset, device=device, random_seed=model_seed)

        # Save the model weights and training losses
        model_path = config['quantile_regression_model_path'].format(alpha=f'{alphas[0]:.1e}to{alphas[-1]:.1e}', attacker=i)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        logger.info(f"Saving quantile regressor to {model_path}")    
        torch.save(model.state_dict(), model_path)
        logger.success("Quantile regressor saved successfully.")

    logger.success("All quantile regressors saved successfully.")

def load_quantile_model(model_path, channel_reduce=1, alphas=(0.01), attacker=0, device='cuda'):
    model = ResNet18(in_channels=6, channel_reduce=channel_reduce, num_classes=len(alphas)).to(device)
    model_path = model_path.format(alpha=f'{alphas[0]:.1e}to{alphas[-1]:.1e}', attacker=attacker)
    if not Path(model_path).exists():
        logger.error(f"Model path {model_path} does not exist. Train enough quantile models with alphas={alphas} first.")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model

def compute_tpr_fpr(predictions, targets, labels, alpha_index, device='cuda'):
    # Compute membership from quantile predictions
    membership_pred = targets[:] > predictions[:, alpha_index]

    # Compute FPR: False Positives / Total Negatives
    total_negatives = torch.sum(labels == 0)
    if total_negatives == 0:
        fpr = 0.0
    else:
        fpr = torch.sum((membership_pred == 1) & (labels == 0)) / total_negatives

    # Compute TPR: True Positives / Total Positives
    total_positives = torch.sum(labels == 1)
    if total_positives == 0:  # No members in the dataset
        tpr = 0.0
    else:
        tpr = torch.sum((membership_pred == 1) & (labels == 1)) / total_positives

    # Compute Precision: True Positives / (True Positives + False Positives)
    total_predicted_positives = torch.sum(membership_pred)
    if total_predicted_positives == 0:
        precision = 0.0
    else:
        precision = torch.sum((membership_pred == 1) & (labels == 1)) / total_predicted_positives

    return tpr, fpr, precision

def compute_evaluation_loss(model, test_dataset, alphas, alpha_index, config, device='cuda'):
    test_loader = DataLoader(test_dataset, batch_size=config['eval_batch_size'], shuffle=False, num_workers=config['num_eval_workers'])
    
    # Use GPU tensors to accumulate results
    all_predictions = []
    all_targets = []
    all_labels = []
    
    for original_image, reconstructed_image, target, labels in tqdm(test_loader, desc="Evaluating", leave=False):
        features = torch.cat((original_image, reconstructed_image), dim=1).to(device)
        target = target.to(device)
        labels = torch.tensor([1 if label == "member" else 0 for label in labels], dtype=torch.float32).to(device)
        
        prediction = model(features).squeeze()
        all_predictions.append(prediction)  # Append GPU tensors
        all_targets.append(target)         # Append GPU tensors
        all_labels.append(labels)           # Append GPU tensors

    # Concatenate tensors on GPU
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Move the final results to CPU as numpy arrays
    test_loss = quantile_loss(all_predictions, all_targets, alphas).cpu().numpy()
        
    test_tpr, test_fpr, test_precision = compute_tpr_fpr(all_predictions, all_targets, all_labels, alpha_index=alpha_index)

    return test_loss, test_tpr, test_fpr, test_precision

def train_quantiles(dataset_name, train_dataset, alphas, eval_dataset=None, device="cuda", random_seed=42):
    config = MODEL_CONFIG[f'{dataset_name}_QUANTILE']
    # Set the random seed for reproducibility
    torch.manual_seed(random_seed)

    # Compute the closest alpha value to the target alpha, and update the configuration
    alpha_index = torch.argmin(torch.abs(alphas - config["target_alpha"]))
    config["target_alpha"] = alphas[alpha_index]

    # Define the model and add parameter count to the configuration
    model = ResNet18(in_channels=6, channel_reduce=config['resnet_channel_reduce'], num_classes=len(alphas), dropout_rate=config['dropout_rate']).to(device)
    config["model_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Initialize W&B with the configuration
    logger.info("Initializing W&B...")
    wandb.init(
        project="quantile_diffusion_mia", 
        config=config,
    )
    config = wandb.config
    logger.success("W&B initialized successfully.")

    # Create a DataLoader for the training dataset
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    # Define the optimizer
    # optim = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'], nesterov=config['nesterov'])
    if config['optimizer'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Define the scheduler
    if config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer=optim, T_max=config['n_epochs'])
    elif config['scheduler'] == 'cosine_warm_restarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer=optim, T_0=10, T_mult=2)
    else:
        raise ValueError(f"Invalid scheduler: {config['scheduler']}")

    # Move the alpha values to the device
    alphas = torch.tensor(alphas).to(device)

    # Train the model
    logger.info("Training quantile regressor...")

    for epoch in tqdm(range(config['n_epochs']), desc="Epochs"):
        model.train()
        running_train_loss = 0.0
        total_tpr = 0.0
        total_fpr = 0.0
        total_samples = 0

        for i, (original_image, reconstructed_image, targets, labels) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}", leave=False):
            batch_size = len(labels)
            total_samples += batch_size
            
            # Transfer the data to the device
            features = torch.cat((original_image, reconstructed_image), dim=1).to(device)
            targets = targets.to(device)
            labels = torch.tensor([1 if label == "member" else 0 for label in labels], dtype=torch.float32).to(device)

            # Forward pass. Handles a matrix of predictions (batch size * num quantiles/classes)
            predictions = model(features).squeeze()

            # Returns a vector of quantiles across several alpha values
            loss = quantile_loss(predictions, targets, alphas)
            tpr, fpr, _ = compute_tpr_fpr(predictions, targets, labels, alpha_index=alpha_index)

            # Backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()

            running_train_loss += loss.item() * batch_size
            total_tpr += tpr * batch_size
            total_fpr += fpr * batch_size

        # Update and log the learning rate
        current_lr = scheduler.get_last_lr()[-1]
        wandb.log({"learning_rate": current_lr})
        scheduler.step()

        # Compute averages
        average_train_loss = running_train_loss / total_samples
        average_tpr = total_tpr / total_samples
        average_fpr = total_fpr / total_samples

        # Log train metrics
        wandb.log({"train_loss": average_train_loss, "train_tpr": average_tpr, "train_fpr": average_fpr})
        
        # Log the test loss for the epoch
        if eval_dataset:
            model.eval()
            with torch.no_grad():
                test_loss, test_tpr, test_fpr, test_precision = compute_evaluation_loss(model, eval_dataset, alphas, alpha_index, config, device=device)
            wandb.log({"test_loss": test_loss, "test_tpr": test_tpr, "test_fpr": test_fpr, "test_precision": test_precision})

        logger.info(
            f"Epoch {epoch + 1} Summary:"
            f"\n  Train Metrics:"
            f"\n    Loss: {average_train_loss:.4f}"
            # f"\n    TPR:  {average_tpr:.4f}"
            f"\n    FPR:  {average_fpr:.4f}"
            f"\n  Test Metrics:"
            f"\n    Loss: {test_loss:.4f}"
            f"\n    TPR:  {test_tpr:.4f}"
            f"\n    FPR:  {test_fpr:.4f}"
            f"\n    Precision: {test_precision:.4f}"
        )

    logger.success("Training complete.")
    wandb.finish()

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
