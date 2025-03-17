import torch
from resnet import ResNet18, ResNet34, ResNet50
import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2

from torchvision import transforms
from torchvision.datasets import CIFAR10

import torch.nn.functional as F

import argparse
from tqdm import tqdm

SECMI_EVALS_BASE='SecMI/mia_evals/'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Evaluation')
parser.add_argument('--in_channel', type=int, default=9)
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--n_quantiles', type=int, default=50)
parser.add_argument('--nonmember_data_path', type=str, default=None)
parser.add_argument('--member_data_path', type=str, default=None)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='./models/')
parser.add_argument('--class_cond', action='store_true', help='Use class conditioning for the model')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for classification (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

#
nonmember_data = torch.load(args.nonmember_data_path)
nonmember_images = nonmember_data['images'].cpu()
nonmember_diffusions = nonmember_data['diffusions'].cpu()
nonmember_internal_samples = nonmember_data['internal_samples'].cpu()
nonmember_labels = nonmember_data['labels'].cpu()

member_data = torch.load(args.member_data_path)
member_images = member_data['images'].cpu()
member_diffusions = member_data['diffusions'].cpu()
member_internal_samples = member_data['internal_samples'].cpu()
member_labels = member_data['labels'].cpu()

device = "cuda"

alphas = torch.logspace(-5, 0, args.n_quantiles, base=10).to(device)
#
#
if args.class_cond:
    in_channels = args.in_channel + args.num_classes
else:
    in_channels = args.in_channel
model = ResNet50(num_classes=len(alphas), in_channels = in_channels).to(device)
model.load_state_dict(torch.load(args.model_path))

print()

nonmember_loader = DataLoader(
    TensorDataset(nonmember_diffusions, nonmember_internal_samples, nonmember_images, nonmember_labels),
    batch_size = args.batch_size,
    shuffle = False)
member_loader = DataLoader(
    TensorDataset(member_diffusions, member_internal_samples, member_images, member_labels),
    batch_size=args.batch_size,
    shuffle=False)

def tpr_and_fpr(predictions, membership_labels):
    #
    true_positive = torch.sum((predictions==1) & (membership_labels==1))
    #
    false_positive = torch.sum((predictions==1) & (membership_labels==0))
    #
    true_negative = torch.sum((predictions==0) & (membership_labels==0))
    #
    false_negative = torch.sum((predictions==0) & (membership_labels==1))

    denominator_tpr = true_positive + false_negative
    denominator_fpr = false_positive + true_negative

    true_positive_rate = true_positive / denominator_tpr if denominator_tpr != 0 else 0.0
    false_positive_rate = false_positive / denominator_fpr if denominator_fpr != 0 else 0.0

    return true_positive_rate, false_positive_rate

cifar10_transform_test = v2.Compose([
    #
    #
    #
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
    #
])

def distance(diffusions, internal_samples):
    return torch.log10((diffusions - internal_samples).pow(2).sum(dim=(1,2,3))).view(-1, 1)
#

member_t_errors = []
member_outputs = []
nonmember_t_errors = []
nonmember_outputs = []

model.eval()
with torch.no_grad():
    for i, (diffusions, internal_samples, images, labels) in tqdm(enumerate(member_loader), desc='Member', total=len(member_loader)):
        diffusions = diffusions.to(device)
        internal_samples = internal_samples.to(device)
        images = cifar10_transform_test(images).to(device)
        labels = labels.to(device)

        #
        if args.class_cond:
            labels = F.one_hot(labels, num_classes=10).float().unsqueeze(-1).unsqueeze(-1)
            labels = labels*torch.ones(1, 1, 32, 32).to(device)
            inputs = torch.cat((images, diffusions, internal_samples, labels), dim=1)
        else:
            inputs = torch.cat((images, diffusions, internal_samples), dim=1)
        
        #
        targets = -distance(diffusions, internal_samples)
        outputs = model(inputs)
        
        member_outputs.append(outputs)
        member_t_errors.append(targets)
    for i, (diffusions, internal_samples, images, labels) in tqdm(enumerate(nonmember_loader), desc='Nonmember', total=len(nonmember_loader)):
        diffusions = diffusions.to(device)
        internal_samples = internal_samples.to(device)
        images = cifar10_transform_test(images).to(device)
        labels = labels.to(device)

        #
        if args.class_cond:
            labels = F.one_hot(labels, num_classes=10).float().unsqueeze(-1).unsqueeze(-1)
            labels = labels*torch.ones(1, 1, 32, 32).to(device)
            inputs = torch.cat((images, diffusions, internal_samples, labels), dim=1)
        else:
            inputs = torch.cat((images, diffusions, internal_samples), dim=1)
        
        #
        targets = -distance(diffusions, internal_samples)
        outputs = model(inputs)
        
        nonmember_outputs.append(outputs)
        nonmember_t_errors.append(targets)

member_outputs = torch.cat(member_outputs)
nonmember_outputs = torch.cat(nonmember_outputs)
member_t_errors = torch.cat(member_t_errors)
nonmember_t_errors = torch.cat(nonmember_t_errors)

all_outputs = torch.cat((member_outputs, nonmember_outputs), dim=0)
all_targets = torch.cat((member_t_errors, nonmember_t_errors), dim=0)
membership_labels = torch.cat((torch.ones(member_outputs.shape[0]), torch.zeros(nonmember_outputs.shape[0]))).to(device)

all_predictions = all_outputs < all_targets
for predictions in all_predictions.T:
    tpr, fpr = tpr_and_fpr(predictions, membership_labels)
    print(f'TPR: {tpr.item():.4f}, FPR: {fpr.item():.4f}')

import os
os.makedirs(args.output_dir, exist_ok=True)

eval_qr_results = {
    'images': nonmember_images,
    'labels': nonmember_labels,
    'outputs': nonmember_outputs,
    'targets': nonmember_t_errors,
}

train_qr_results = {
    'images': member_images,
    'labels': member_labels,
    'outputs': member_outputs,
    'targets': member_t_errors,
}

torch.save(eval_qr_results, os.path.join(args.output_dir, 'qr_results_eval.pt'))
torch.save(train_qr_results, os.path.join(args.output_dir, 'qr_results_train.pt'))