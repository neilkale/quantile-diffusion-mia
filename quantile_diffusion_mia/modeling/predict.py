from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from quantile_diffusion_mia.config import MODELS_DIR, PROCESSED_DATA_DIR
from quantile_diffusion_mia.utils import ddim_multistep, ddim_singlestep

import torch
import torch.nn.functional as F

app = typer.Typer()

@app.command()
def noise_and_denoise(
    data_loader, 
    model,
    dt: int, 
    steps: int,
    flags: dict
):
    logger.info("Noise and denoise data...")
    target_steps = list(range(0, steps, dt))[1:]

    internal_diffusion_list = []
    internal_denoised_list = []
    t_errors_list = []

    for batch_idx, x in enumerate(tqdm(data_loader)):
        x = x[0].cuda()
        x = x * 2 - 1

        x_sec = ddim_multistep(model, flags, x, t_c=0, target_steps=target_steps)
        x_sec = x_sec['x_t_target']
        x_sec_recon = ddim_singlestep(model, flags, x_sec, t_c=target_steps[-1], t_target=target_steps[-1] + dt)
        x_sec_recon = ddim_singlestep(model, flags, x_sec_recon['x_t_target'], t_c=target_steps[-1] + dt, t_target=target_steps[-1])
        x_sec_recon = x_sec_recon['x_t_target']

        internal_diffusion_list.append(x_sec)
        internal_denoised_list.append(x_sec_recon)

        loss = F.mse_loss(x_sec_recon, x_sec, reduction='none').mean(dim=(1, 2, 3))
        t_errors_list.append(loss)

    logger.success("Noise and denoise complete.")

    return {
        'noised_images': torch.concat(internal_diffusion_list),
        'reconstructed_images': torch.concat(internal_denoised_list),
        't_errors': torch.concat(t_errors_list)
    }


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
