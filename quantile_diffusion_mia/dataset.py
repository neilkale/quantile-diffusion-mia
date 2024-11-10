from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100

from quantile_diffusion_mia.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, DATASET_CONFIG
from quantile_diffusion_mia.utils import read_flags, get_diffusion_model, fix_seed
from quantile_diffusion_mia.modeling.predict import noise_and_denoise

app = typer.Typer()

# TODO: Lazy loading
class QuantileRegressionDataset(Dataset):
    def __init__(self, dataset_name, original_images=None, reconstructed_images=None, t_errors=None,  labels=None, augment=None, indices=None):
        self.dataset_name = dataset_name
        self.original_images = original_images
        self.reconstructed_images = reconstructed_images
        self.t_errors = t_errors
        self.labels = labels
        self.augment = augment
        self.indices = list(range(len(original_images))) if indices is None else indices

        if augment:
            self.transform = transforms.Compose([
                transforms.RandAugment(num_ops=5),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Load the original image using the index
        actual_idx = self.indices[idx]
        original_image = self.original_images[actual_idx]

        # Apply transforms to the original image
        original_image = self.transform(original_image)

        # Get reconstructed image and t_error if they exist
        if self.reconstructed_images is not None:
            reconstructed_image = self.reconstructed_images[actual_idx]
        else:
            reconstructed_image = torch.zeros_like(original_image)

        if self.t_errors is not None:
            t_error = self.t_errors[actual_idx]
        else:
            t_error = 0

        # Get the label
        if self.labels is not None:
            label = self.labels[actual_idx]
        else:
            label = ''

        return original_image, reconstructed_image, t_error, label

    def set_indices(self, indices):
        self.indices = indices

    def save(self, file_path):
        # Save the stacked dataset containing original images, reconstructed images, and t-errors
        torch.save({
            'dataset_name': self.dataset_name,
            'original_images': self.original_images,
            'reconstructed_images': self.reconstructed_images,
            't_errors': self.t_errors,
            'labels': self.labels,
            'augment': self.augment,
            'indices': self.indices,
        }, file_path)

    @staticmethod
    def load(file_path):
        # Load the stacked dataset from a file
        loaded_data = torch.load(file_path, weights_only=False)
    
        return QuantileRegressionDataset(
                dataset_name=loaded_data['dataset_name'],
                original_images=loaded_data['original_images'],
                reconstructed_images=loaded_data['reconstructed_images'],
                t_errors=loaded_data['t_errors'],
                labels=loaded_data['labels'],
                augment=loaded_data['augment'],
                indices=loaded_data['indices']
        )   

@app.command()
def load_reconstructed_images_and_t_errors(dataset_name: str, dataset, dt: int = 1, steps: int = 50, device: str = 'cuda'):
    # Load reconstructed images and t_errors if they exist
    config = DATASET_CONFIG[dataset_name]
    output_path = PROCESSED_DATA_DIR / f'{dataset_name}' / f't{steps}' / 'reconstructed_images_and_t_errors.pt'
    if output_path.exists():
        logger.info(f'Loading existing t={steps} reconstructed images and t_errors for {dataset_name} at {output_path}')
        return torch.load(output_path, weights_only=True)

    # Read the flag file to get the diffusion model configuration
    logger.info(f'Reading flags from {config["diffusion_model_flag_path"]}')
    flag_path = config['diffusion_model_flag_path']
    flags = read_flags(flag_path)

    # Load the diffusion model
    logger.info(f'Loading diffusion model from {config["diffusion_model_path"]}')
    model = get_diffusion_model(config["diffusion_model_path"], flags, WA=True).to(device)
    logger.success('Diffusion model loaded successfully')

    # Create a DataLoader for the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Generate reconstructed images and t_errors
    logger.info(f'Generating reconstructed images for {dataset_name}')
    output = noise_and_denoise(data_loader, model=model, dt=dt, steps=steps, flags=flags)
    reconstructed_images = output['reconstructed_images']
    t_errors = output['t_errors']
    logger.success(f'Reconstructed images generated successfully')
    
    # Save the reconstructed images and t_errors
    output_path = PROCESSED_DATA_DIR / f'{dataset_name}' / f't{steps}' / 'reconstructed_images_and_t_errors.pt'
    if not output_path.parent.exists():
        logger.info(f'Creating directory {output_path.parent}')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.success(f'Directory created successfully')
    logger.info(f'Saving reconstructed images and t_errors to {output_path}')
    torch.save({'reconstructed_images': reconstructed_images, 't_errors': t_errors}, output_path)

    return {'reconstructed_images': reconstructed_images, 't_errors': t_errors}
    
@app.command()
def create_quantileregression_dataset(dataset_name: str, dt: int = 1, steps: int = 50, augment: bool = False, device: str = 'cuda', random_seed: int = 42):
    # Fix seed for reproducibility
    fix_seed(random_seed)

    # Set transforms based on augmentation flag
    if augment:
        transform = transforms.Compose([
            transforms.RandAugment(num_ops=5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # Load dataset based on the dataset_name
    config = DATASET_CONFIG[dataset_name]
    if dataset_name == 'CIFAR10':
        dataset = CIFAR10(root=config['data_path'], train=True, download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        dataset = CIFAR100(root=config['data_path'], train=True, download=True, transform=transform)
    else:
        logger.error(f"Unsupported dataset: {dataset_name}")
    # You can add support for more datasets here.

    # Load reconstructed_images and t_errors
    output = load_reconstructed_images_and_t_errors(dataset_name, dataset, dt=dt, steps=steps, device=device)
    logger.success("Loaded reconstructed images and t_errors")
    reconstructed_images = output['reconstructed_images'].cpu()
    t_errors = output['t_errors'].cpu()

    # Load the dataset with the specified indices and apply augmentation if requested
    logger.info(f'Transforming original images for {dataset_name}')
    original_images = []
    for idx in tqdm(range(len(dataset))):
        original_images.append(transform(dataset.data[idx]))
    original_images = torch.stack(original_images).cpu()
    logger.success(f'Original images transformed successfully')

    # Load the labels
    train_indices_path = config['diffusion_model_split_path']
    logger.info(f'Loading diffusion model membership labels from {train_indices_path}')
    indices_data = np.load(train_indices_path)
    member_indices = indices_data['mia_train_idxs'] 
    labels = ['member' if idx in member_indices else 'nonmember' for idx in range(len(dataset))]
    logger.success(f'Diffusion model membership labels loaded successfully')

    dataset = QuantileRegressionDataset(dataset_name, original_images=original_images, reconstructed_images=reconstructed_images, t_errors=t_errors, labels=labels, augment=augment)
    
    # Save the combined dataset
    output_path = PROCESSED_DATA_DIR / f'{dataset_name}' / f'combined_dataset.pt'
    logger.info(f'Saving combined dataset to {output_path}')
    dataset.save(output_path)
    logger.success(f'Combined dataset saved successfully')

    # Split the dataset into train and evaluation sets
    split_quantile_regression_dataset(dataset_name)



@app.command()
def split_quantile_regression_dataset(dataset_name: str, train_split: float = 0.5, random_seed: int = 42):
    config = DATASET_CONFIG[dataset_name]

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Set the indices for the train and evaluation sets
    train_indices_path = config['diffusion_model_split_path']
    logger.info(f'Loading train indices from {train_indices_path}')
    indices_data = np.load(train_indices_path)
    logger.success(f'Diffusion model train/eval indices loaded successfully')
    nonmember_indices = indices_data['mia_eval_idxs']
    member_indices = indices_data['mia_train_idxs'] 

    # Shuffle non_train_indices
    np.random.shuffle(nonmember_indices)
    # Calculate split index
    split_index = int(len(nonmember_indices) * train_split)
    # Split non_train_indices into quantile_eval_indices and quantile_train_indices
    logger.info(f'Splitting indices into train and eval for quantile regression')
    quantile_train_indices = nonmember_indices[:split_index]
    quantile_eval_indices = np.concatenate((member_indices, nonmember_indices[split_index:]))

    # Save the indices
    output_path = config['quantile_regression_split_path']
    logger.info(f'Saving quantile regression split indices to {output_path}')
    np.savez(output_path, quantile_train_indices=quantile_train_indices, quantile_eval_indices=quantile_eval_indices)
    logger.success(f'Quantile regression split indices saved successfully')

# Kept as an example for future reference
@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
