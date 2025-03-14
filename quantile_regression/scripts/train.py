"""
Train a quantile regression model on images.
"""

import argparse

from quantile_regression import dist_util, logger
from quantile_regression.image_datasets import load_data
from quantile_regression.script_util import (
    model_defaults,
    create_model,
    args_to_dict,
    add_dict_to_argparser,
)
from quantile_regression.train_util import TrainLoop

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args.logdir)

    logger.log("creating regression model...")
    model, diffusion = create_model(
        **args_to_dict(args, model_defaults().keys())
    )
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

def create_argparser():
    defaults = dict(
        data_dir="",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        logdir=None,
    )
    defaults.update(model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()