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
        'diffusion_model_path': 'models/ddpm-CIFAR10/checkpoint.pt',
        'diffusion_model_flag_path': 'models/ddpm-CIFAR10/flagfile.txt',
        'diffusion_model_split_path': 'models/ddpm-CIFAR10/CIFAR10_train_ratio0.5.npz',
        'quantile_regression_data_path': 'data/processed/CIFAR10/combined_dataset.pt',
        'quantile_regression_split_path': 'data/interim/CIFAR10_quantile_split.npz',
        'quantile_regression_model_path': 'models/qr-CIFAR10/alpha{alpha}_checkpoint.pt'
    },
    'CIFAR100': {
        'data_path': 'data/raw/cifar100/',
        'train_ratio': 0.5,
        'image_size': (32, 32),
        'channels': 3,
        'batch_size': 128,
        'diffusion_model_path': 'models/ddpm-CIFAR100/checkpoint.pt',
        'diffusion_model_flag_path': 'models/ddpm-CIFAR100/flagfile.txt',
        'diffusion_model_split_path': 'models/ddpm-CIFAR100/CIFAR100_train_ratio0.5.npz',
        'quantile_regression_data_path': 'data/processed/CIFAR100/combined_dataset.pt',
        'quantile_regression_split_path': 'models/interim/CIFAR100_quantile_split.npz',
        'quantile_regression_model_path': 'models/qr-CIFAR100/alpha{alpha}_checkpoint.pt'
    },
}

MODEL_CONFIG = {
    'CIFAR10_QUANTILE': {
        'n_epochs': 50,
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'batch_size': 128,
    },
    'CIFAR100_QUANTILE': {
        'n_epochs': 15,
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'batch_size': 128,
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
