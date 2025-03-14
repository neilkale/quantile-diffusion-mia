import argparse
import inspect

from resnet import ResNet18

def model_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=32,
        num_in_channels=3,
        channel_reduce=1, 
        num_classes=1, 
        dropout_rate=0,
    )
    return res

def create_model(
    image_size,
    num_in_channels,
    channel_reduce, 
    num_classes, 
    dropout_rate,
):
    return ResNet18(image_size=image_size, num_in_channels=num_in_channels, channel_reduce=channel_reduce, num_classes=num_classes, dropout_rate=dropout_rate)

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
