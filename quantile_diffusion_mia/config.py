from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Datasets
DATASET_CONFIG = {
    'CIFAR10': {
        'data_path': 'data/raw/cifar10/',
        'train_ratio': 0.5,
        'image_size': (32, 32),
        'channels': 3,
        'batch_size': 128,
        'num_workers': 2,
        'normalization_transform': [(0, 0, 0), (1, 1, 1)],
        'diffusion_model_path': 'models/ddpm-CIFAR10/checkpoint.pt',
        'diffusion_model_flag_path': 'models/ddpm-CIFAR10/flagfile.txt',
        'diffusion_model_split_path': 'models/ddpm-CIFAR10/CIFAR10_train_ratio0.5.npz',
        # 'quantile_regression_data_path': 'data/processed/CIFAR10/combined_dataset.pt',
        # 'quantile_regression_split_path': 'data/interim/CIFAR10_quantile_split.npz',
        'quantile_regression_data_path': 'data/processed/CIFAR10/secmi_combined_dataset_CIFAR10_dt1_t50.pt',  # Use the data from SecMI
        'quantile_regression_split_path': 'data/interim/secmi_CIFAR10_quantile_split.npz',  # Use the split from SecMI
        'quantile_regression_model_path': 'models/qr-CIFAR10/secmi_alpha{alpha}_attacker{attacker}_checkpoint.pt', # Save the model for SecMI data
        'quantile_regression_model_log_path': 'models/qr-CIFAR10/logs/secmi_alpha{alpha}_attacker{attacker}_checkpoint.npy', # Log the model for SecMI data
    },
    'CIFAR100': {
        'data_path': 'data/raw/cifar100/',
        'train_ratio': 0.5,
        'image_size': (32, 32),
        'channels': 3,
        'batch_size': 128,
        'num_workers': 4,
        'normalization_transform': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
        'diffusion_model_path': 'models/ddpm-CIFAR100/checkpoint.pt',
        'diffusion_model_flag_path': 'models/ddpm-CIFAR100/flagfile.txt',
        'diffusion_model_split_path': 'models/ddpm-CIFAR100/CIFAR100_train_ratio0.5.npz',
        'quantile_regression_data_path': 'data/processed/CIFAR100/combined_dataset.pt',
        'quantile_regression_split_path': 'models/interim/CIFAR100_quantile_split.npz',
        'quantile_regression_model_path': 'models/qr-CIFAR10/alpha{alpha}_attacker{attacker}_checkpoint.pt',
        'quantile_regression_model_log_path': 'models/qr-CIFAR100/logs/alpha{alpha}_attacker{attacker}_checkpoint.npy',
    },
}

MODEL_CONFIG = {
    'CIFAR10_QUANTILE': {
        'n_epochs': 1000,
        'resnet_channel_reduce': 1,
        'lr': 1e-3,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 5e-4,

        'batch_size': 128,
        'num_workers': 2,
        'eval_batch_size': 2048,
        'num_eval_workers': 4,

        'alpha_min': 1e-8, 
        'alpha_max': 1e-1, 
        'num_quantiles': 300,
        'target_alpha': 1e-2,
        'num_attackers': 7,
    },
    'CIFAR100_QUANTILE': {
        'n_epochs': 200,
        'resnet_channel_reduce': 64,
        'lr': 1e-3,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 5e-4,
        'batch_size': 128,
        'num_workers': 4,
        'alpha_min': 1e-8, 
        'alpha_max': 1e-1, 
        'num_quantiles': 300,
        'target_alpha': 1e-2,
        'num_attackers': 7,
    }
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
