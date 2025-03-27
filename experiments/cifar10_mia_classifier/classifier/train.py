import torch
from resnet import ResNet18, ResNet34, ResNet50
from scheduler import CosineAnnealingWarmRestartsWithDecay
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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

nm_train_diffusions, nm_test_diffusions, nm_train_internal_samples, nm_test_internal_samples, nm_train_images, nm_test_images, nm_train_labels, nm_test_labels \
    = train_test_split(nonmember_diffusions, nonmember_internal_samples, nonmember_images, nonmember_labels, test_size=0.2, random_state=42)

m_train_diffusions, m_test_diffusions, m_train_internal_samples, m_test_internal_samples, m_train_images, m_test_images, m_train_labels, m_test_labels \
    = train_test_split(member_diffusions, member_internal_samples, member_images, member_labels, test_size=0.2, random_state=42)

all_train_diffusions = torch.concat((nm_train_diffusions, m_train_diffusions), axis=0)
all_train_internal_samples = torch.concat((nm_train_internal_samples, m_train_internal_samples), axis=0)
all_train_images = torch.concat((nm_train_images, m_train_images), axis=0)
all_train_labels = torch.concat((nm_train_labels, m_train_labels), axis=0)
train_membership_labels = np.concatenate((np.zeros(len(nm_train_diffusions)), np.ones(len(m_train_diffusions))), axis=0).astype(np.int64)
train_membership_labels = np.eye(2)[train_membership_labels]

all_test_diffusions = torch.concat((nm_test_diffusions, m_test_diffusions), axis=0)
all_test_internal_samples = torch.concat((nm_test_internal_samples, m_test_internal_samples), axis=0)
all_test_images = torch.concat((nm_test_images, m_test_images), axis=0)
all_test_labels = torch.concat((nm_test_labels, m_test_labels), axis=0)
test_membership_labels = np.concatenate((np.zeros(len(nm_test_diffusions)), np.ones(len(m_test_diffusions))), axis=0).astype(np.int64)
test_membership_labels = np.eye(2)[test_membership_labels]

device = "cuda"
test_membership_labels = torch.from_numpy(test_membership_labels).to(device)
train_membership_labels = torch.from_numpy(train_membership_labels).to(device)
#
#
if args.class_cond:
    in_channels = args.in_channel + args.num_classes
else:
    in_channels = args.in_channel
model = ResNet18(num_classes=2, in_channels = in_channels).to(device)

print()

optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=1e-4)
#
#
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

train_loader = DataLoader(
    TensorDataset(all_train_diffusions, all_train_internal_samples, all_train_images, all_train_labels, train_membership_labels),
    batch_size = args.batch_size,
    shuffle = True)
test_loader = DataLoader(
    TensorDataset(all_test_diffusions, all_test_internal_samples, all_test_images, all_test_labels, test_membership_labels),
    batch_size=args.batch_size*10,
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
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0.0

    return true_positive_rate, false_positive_rate, precision

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

# def distance(diffusions, internal_samples):
    # return torch.log10((diffusions - internal_samples).pow(2).sum(dim=(1,2,3))).view(-1, 1)
#

train_loss_history = []
test_loss_history = []
tpr_history = []
fpr_history = []
precision_history = []
lrs = []

import os
os.makedirs(args.output_dir, exist_ok=True)

for epoch in range(args.n_epochs):
    model.train()
    running_loss = 0
    for i, (diffusions, internal_samples, images, labels, membership) in enumerate(train_loader):
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
        targets = membership
        outputs = model(inputs)
        #
        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='sum')
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    running_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch+1}, Training Loss: {running_loss:.4f}')
    
    loss = 0
    all_outputs = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for i, (diffusions, internal_samples, images, labels, membership) in enumerate(test_loader):
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
            targets = membership
            outputs = model(inputs)
            batch_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='sum')
            loss += batch_loss.item()

            all_outputs.append(outputs)
            all_targets.append(targets)
    
    loss /= len(test_loader.dataset)

    all_predictions = torch.cat(all_outputs).sigmoid().cpu()
    membership_labels = torch.cat(all_targets).cpu()

    tpr, fpr, precision = tpr_and_fpr(torch.argmax(all_predictions, dim=1), torch.argmax(membership_labels, dim=1))
    print(f'Epoch {epoch+1}, Test Loss: {loss:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}')

    scheduler.step()

    train_loss_history.append(running_loss)
    test_loss_history.append(loss)
    tpr_history.append(tpr)
    fpr_history.append(fpr)
    precision_history.append(precision)

    epoch_list = list(range(1, epoch + 2))
    plt.figure()
    plt.plot(epoch_list, train_loss_history, label='Train Loss', marker='o', color='steelblue', markersize=2)
    plt.plot(epoch_list, test_loss_history, label='Test Loss', marker='o', color='peru', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training vs Test Loss')
    plt.savefig(f"{args.output_dir}/loss_plot.png")
    plt.close()

    # At the epoch, plot the history of TPR at FPR ~ 0.01
    plt.figure()
    plt.plot(epoch_list, tpr_history, label='TPR', marker='o', color='forestgreen', markersize=2)
    plt.plot(epoch_list, fpr_history, label='FPR', marker='o', color='indianred', markersize=2)
    plt.plot(epoch_list, precision_history, label='Precision', marker='o', color='steelblue', markersize=2)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Rate")
    plt.title("Validation Metrics Over Epochs")
    plt.savefig(f"{args.output_dir}/tpr_at_fpr_plot.png")
    plt.close()

    # Plot the learning rate schedule
    lrs.append(optimizer.param_groups[0]['lr'])
    plt.figure()
    plt.plot(epoch_list, lrs, marker='o', markersize=2, color='steelblue')
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.savefig(f"{args.output_dir}/lr_schedule_plot.png")
    plt.close()

torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))