
import copy
import sys
import os
import numpy as np
import random
import tqdm
import argparse
import json

from sklearn import metrics
from dataset_utils import load_member_data
from absl import flags
from model import UNet
import torch
import resnet

def ddim_singlestep(model, FLAGS, x, t_c, t_target, requires_grad=False, device='cuda'):

    x = x.to(device)

    t_c = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_c)
    t_target = x.new_ones([x.shape[0], ], dtype=torch.long) * (t_target)

    betas = torch.linspace(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).double().to(device)
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


class MIDataset():

    def __init__(self, member_data, nonmember_data, member_label, nonmember_label):
        self.data = torch.concat([member_data, nonmember_data])
        self.label = torch.concat([member_label, nonmember_label]).reshape(-1)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, item):
        data = self.data[item]
        return data, self.label[item]


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_FLAGS(flag_path):
    FLAGS = flags.FLAGS
    flags.DEFINE_bool('train', False, help='train from scratch')
    flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
    # UNet
    flags.DEFINE_integer('ch', 128, help='base channel of UNet')
    flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
    flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
    flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
    flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
    # Gaussian Diffusion
    flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
    flags.DEFINE_float('beta_T', 0.02, help='end beta value')
    flags.DEFINE_integer('T', 1000, help='total diffusion steps')
    flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
    flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
    # Training
    flags.DEFINE_float('lr', 2e-4, help='target learning rate')
    flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
    flags.DEFINE_integer('total_steps', 800000, help='total training steps')
    flags.DEFINE_integer('img_size', 32, help='image size')
    flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
    flags.DEFINE_integer('batch_size', 128, help='batch size')
    flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
    flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
    flags.DEFINE_bool('parallel', False, help='multi gpu training')
    # Logging & Sampling
    flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
    flags.DEFINE_integer('sample_size', 64, "sampling size of images")
    flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
    # Evaluation
    flags.DEFINE_integer('save_step', 80000, help='frequency of saving checkpoints, 0 to disable during training')
    flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
    flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
    flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
    flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

    FLAGS.read_flags_from_files(flag_path)
    return FLAGS


def get_model(ckpt, FLAGS, WA=True):
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    # load model and evaluate
    ckpt = torch.load(ckpt)

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


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def norm(x):
    return (x + 1) / 2

def get_intermediate_results(model, FLAGS, data_loader, t_sec, timestep):
    target_steps = list(range(0, t_sec, timestep))[1:]

    internal_diffusion_list = []
    internal_denoised_list = []
    for batch_idx, x in enumerate(tqdm.tqdm(data_loader)):
        x = x[0].cuda()
        x = x * 2 - 1

        x_sec = ddim_multistep(model, FLAGS, x, t_c=0, target_steps=target_steps)
        x_sec = x_sec['x_t_target']
        x_sec_recon = ddim_singlestep(model, FLAGS, x_sec, t_c=target_steps[-1], t_target=target_steps[-1] + timestep)
        x_sec_recon = ddim_singlestep(model, FLAGS, x_sec_recon['x_t_target'], t_c=target_steps[-1] + timestep, t_target=target_steps[-1])
        x_sec_recon = x_sec_recon['x_t_target']

        internal_diffusion_list.append(x_sec)
        internal_denoised_list.append(x_sec_recon)

    return {
        'internal_diffusions': torch.concat(internal_diffusion_list),
        'internal_denoise': torch.concat(internal_denoised_list)
    }

def calculate_t_error(model, FLAGS, dataset_root, timestep=10, t_sec=100, batch_size=128, dataset='cifar10', output_dir='t_errors'):

    # load splits
    member_set, nonmember_set, member_loader, nonmember_loader = load_member_data(dataset_root=dataset_root, dataset_name=dataset, batch_size=batch_size,
                                                             shuffle=False, randaugment=False)

    member_results = get_intermediate_results(model, FLAGS, member_loader, t_sec, timestep)
    nonmember_results = get_intermediate_results(model, FLAGS, nonmember_loader, t_sec, timestep)

    t_results = {
        'member_diffusions': member_results['internal_diffusions'],
        'member_internal_samples': member_results['internal_denoise'],
        'nonmember_diffusions': nonmember_results['internal_diffusions'],
        'nonmember_internal_samples': nonmember_results['internal_denoise'],
    }

    member_t_errors = torch.sum((t_results['member_diffusions'] - t_results['member_internal_samples']) ** 2, dim=[1, 2, 3])
    nonmember_t_errors = torch.sum((t_results['nonmember_diffusions'] - t_results['nonmember_internal_samples']) ** 2, dim=[1, 2, 3])

    # Collect data for JSON output with only image_id and t_error
    member_data = [{'image_id': int(idx), 't_error': float(t_error.cpu().numpy())}
                   for idx, t_error in zip(member_set.idxs, member_t_errors)]

    nonmember_data = [{'image_id': int(idx), 't_error': float(t_error.cpu().numpy())}
                      for idx, t_error in zip(nonmember_set.idxs, nonmember_t_errors)]

    # Save results to JSON
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'member_results.json'), 'w') as f:
        json.dump(member_data, f, indent=4)

    with open(os.path.join(output_dir, 'nonmember_results.json'), 'w') as f:
        json.dump(nonmember_data, f, indent=4)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='checkpoints/CIFAR10')
    parser.add_argument('--data_root', type=str, default='datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--t_sec', type=int, default=100)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='t_errors')
    args = parser.parse_args()

    fix_seed(0)
    ckpt = os.path.join(args.model_dir, 'checkpoint.pt')
    flag_path = os.path.join(args.model_dir, 'flagfile.txt')
    device = 'cuda'
    FLAGS = get_FLAGS(flag_path)
    
    FLAGS([flag_path])
    model = get_model(ckpt, FLAGS, WA=True).to(device)

    # Get the posterior estimation error at step args.t_sec for each sample in args.data_root
    # Write the output to directory args.output_dir
    calculate_t_error(model, FLAGS, dataset_root=args.data_root, t_sec=args.t_sec, timestep=args.k, batch_size=1024, dataset=args.dataset, output_dir=args.output_dir)
