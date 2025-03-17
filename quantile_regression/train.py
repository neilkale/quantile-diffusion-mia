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

SECMI_EVALS_BASE='SecMI/mia_evals/'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--in_channel', type=int, default=9)
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_quantiles', type=int, default=50)
parser.add_argument('--nonmember_data_path', type=str, default=None)
parser.add_argument('--member_data_path', type=str, default=None)
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

train_diffusions, test_diffusions, train_internal_samples, test_internal_samples, train_images, test_images, train_labels, test_labels \
    = train_test_split(nonmember_diffusions, nonmember_internal_samples, nonmember_images, nonmember_labels, test_size=0.2, random_state=42)

all_test_diffusions = torch.concat((member_diffusions, test_diffusions), axis=0)
all_test_internal_samples = torch.concat((member_internal_samples, test_internal_samples), axis=0)
membership_labels = np.concatenate((np.ones(len(member_diffusions)), np.zeros(len(test_diffusions))), axis=0)
all_test_images = torch.concat((member_images, test_images), axis=0)
all_test_labels = torch.concat((member_labels, test_labels), axis=0)

device = "cuda"
membership_labels = torch.from_numpy(membership_labels).to(device)

alphas = torch.logspace(-5, 0, args.n_quantiles, base=10).to(device)
#
#
if args.class_cond:
    in_channels = args.in_channel + args.num_classes
else:
    in_channels = args.in_channel
model = ResNet50(num_classes=len(alphas), in_channels = in_channels).to(device)

print()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, fused=True)
#
#
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

train_loader = DataLoader(
    TensorDataset(train_diffusions, train_internal_samples, train_images, train_labels),
    batch_size = args.batch_size,
    shuffle = True)
test_loader = DataLoader(
    TensorDataset(all_test_diffusions, all_test_internal_samples, all_test_images, all_test_labels),
    batch_size=args.batch_size*10,
    shuffle=False)

def pinball_loss(outputs, targets, alphas):
    diff = outputs - targets
    alphas = alphas.view(1, -1)
    losses = torch.max(alphas * diff, (alphas-1)*diff)
    return losses.sum(-1).mean()

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

cifar10_transform_train = v2.Compose([
    v2.RandomCrop(32, padding=4),
    #
    v2.RandomHorizontalFlip(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
    #
])

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

for epoch in range(args.n_epochs):
    model.train()
    running_loss = 0
    for i, (diffusions, internal_samples, images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        diffusions = diffusions.to(device)
        internal_samples = internal_samples.to(device)
        images = cifar10_transform_train(images).to(device)
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
        #
        loss = pinball_loss(outputs, targets, alphas)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    running_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch+1}, Training Loss: {running_loss:.4f}')

    loss = 0.
    all_outputs = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for i, (diffusions, internal_samples, images, labels) in enumerate(test_loader):
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

            loss = pinball_loss(outputs, targets, alphas)
            
            all_outputs.append(outputs)
            all_targets.append(targets.view(-1, 1))
            loss += pinball_loss(outputs, targets, alphas)*inputs.size(0)
        loss /= len(train_loader.dataset)
        print(f'Epoch {epoch+1}, Test Loss: {loss.item():.4f}')
        
    all_predictions = torch.cat(all_outputs) <= torch.cat(all_targets)
    for predictions in all_predictions.T:
        tpr, fpr = tpr_and_fpr(predictions, membership_labels)
        print(f'Epoch {epoch+1}, TPR: {tpr.item():.4f}, FPR: {fpr.item():.4f}')

    scheduler.step()

import os
os.makedirs(args.output_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))