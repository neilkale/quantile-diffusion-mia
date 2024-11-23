import random
import numpy as np
import torch
from quantile_diffusion_mia.modeling.diffusion_model import UNet
from quantile_diffusion_mia.modeling.resnet import ResNet18

def read_flags(filename):
    # Define keys that should always be lists, even if they appear only once
    list_keys = {"attn"}
    
    flags = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                # Check if the line contains an '=' indicating a key-value pair
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip("--").strip()
                    value = value.strip() if value else ""
                    
                    # Attempt to convert the value to int or float, if applicable
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Leave as string if it can't be converted
                    
                    # Handle keys that should always be lists or have multiple occurrences
                    if key in flags:
                        # If key exists, convert it to a list if it's not one already
                        if not isinstance(flags[key], list):
                            flags[key] = [flags[key]]
                        flags[key].append(value)
                    elif key in list_keys:
                        # For specified keys, store as a one-element list even if it appears once
                        flags[key] = [value]
                    else:
                        # Store as a single value for non-list keys
                        flags[key] = value
                else:
                    # If there's no '=', treat it as a boolean flag (True if present)
                    key = line.strip("--")
                    if key in flags:
                        # Append to list if it's already there or convert to list if needed
                        if not isinstance(flags[key], list):
                            flags[key] = [flags[key]]
                        flags[key].append(True)
                    elif key in list_keys:
                        # Store as a one-element list for specified keys
                        flags[key] = [True]
                    else:
                        # Store as a single boolean value for non-list keys
                        flags[key] = True
    return flags

def get_diffusion_model(ckpt, FLAGS, WA=True):
    model = UNet(
        T=FLAGS["T"], ch=FLAGS["ch"], ch_mult=FLAGS["ch_mult"], attn=FLAGS["attn"],
        num_res_blocks=FLAGS["num_res_blocks"], dropout=FLAGS["dropout"])
    # load model and evaluate
    ckpt = torch.load(ckpt, weights_only=True)

    if WA:
        weights = ckpt['ema_model']
    else:
        weights = ckpt['net_model']

    new_state_dict = {}
    for key, val in weights.items():
        if key.startswith('module.'):
            new_state_dict.update({key[7:]: val})
        else:
            new_state_dict.update({key: val})

    model.load_state_dict(new_state_dict)

    model.eval()

    return model

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def ddim_singlestep(model, FLAGS, x, t_c, t_target, requires_grad=False, device='cuda'):

    x = x.to(device)

    t_c = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_c)
    t_target = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_target)

    betas = torch.linspace(FLAGS["beta_1"], FLAGS["beta_T"], FLAGS["T"]).double().to(device)
    alphas = 1. - betas
    alphas = torch.cumprod(alphas, dim=0)

    alphas_t_c = extract(alphas, t=t_c, x_shape=x.shape)
    alphas_t_target = extract(alphas, t=t_target, x_shape=x.shape)

    if requires_grad:
        epsilon = model(x, t_c)
    else:
        with torch.no_grad():
            epsilon = model(x, t_c)

    pred_x_0 = (x - ((1 - alphas_t_c).sqrt() * epsilon)) / (alphas_t_c.sqrt())
    x_t_target = alphas_t_target.sqrt() * pred_x_0 \
                 + (1 - alphas_t_target).sqrt() * epsilon

    return {
        'x_t_target': x_t_target,
        'epsilon': epsilon
    }

def ddim_multistep(model, FLAGS, x, t_c, target_steps, clip=False, device='cuda', requires_grad=False):
    for idx, t_target in enumerate(target_steps):
        result = ddim_singlestep(model, FLAGS, x, t_c, t_target, requires_grad=requires_grad, device=device)
        x = result['x_t_target']
        t_c = t_target

    if clip:
        result['x_t_target'] = torch.clip(result['x_t_target'], -1, 1)

    return result

# Define a quantile loss function
def quantile_loss(predictions, targets, qs):
    qs = 1 - qs
    # Expand dimensions to align predictions/targets with quantiles
    targets = targets.unsqueeze(1)  
    errors = targets - predictions  # Shape: (batch_size, num_quantiles)
    qs = qs.view(1, -1)  # Shape: (1, num_quantiles)
    # Compute quantile losses for all quantiles in a single operation
    losses = torch.max((qs - 1) * errors, qs * errors)
    # Mean across batch dimension
    return losses.mean()  # Returns a single value for loss    