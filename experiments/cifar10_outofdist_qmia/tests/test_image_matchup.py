from torchvision.datasets import CIFAR10
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torchvision.datasets import ImageFolder

SECMI_EVALS_BASE='SecMI/mia_evals/'
cifar10_member = CIFAR10(SECMI_EVALS_BASE + 'datasets/cifar10/', train=True, download=True, transform=transforms.ToTensor())
cifar10_nonmember = CIFAR10(SECMI_EVALS_BASE + 'datasets/cifar10/', train=True, download=True, transform=transforms.ToTensor())
member_splits = np.load(SECMI_EVALS_BASE + 'member_splits/CIFAR10_train_ratio0.5.npz')

print("Sample member indices: ", member_splits["mia_train_idxs"][:10])
print("Sample non-member indices: ", member_splits["mia_eval_idxs"][:10])

cifar10_member.data = cifar10_member.data[member_splits["mia_train_idxs"]]
cifar10_nonmember.data = cifar10_nonmember.data[member_splits["mia_eval_idxs"]]
cifar10_member.targets = np.array(cifar10_member.targets)[member_splits["mia_train_idxs"]]
cifar10_nonmember.targets = np.array(cifar10_nonmember.targets)[member_splits["mia_eval_idxs"]]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 10, figsize=(15, 3))

for i in range(10):
    axes[0, i].imshow(cifar10_member.data[i])
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Members', fontsize=12)

for i in range(10):
    axes[1, i].imshow(cifar10_nonmember.data[i])
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Non-members', fontsize=12)

plt.tight_layout()
plt.savefig('experiments/cifar10_outofdist_qmia/cifar10_member_nonmember_samples_shuai.png')

transform = transforms.ToTensor()
fig2, axes2 = plt.subplots(2, 10, figsize=(15, 3))

for i in range(10):
    member_idx = member_splits["mia_train_idxs"][i]
    img_path = f"data/temp/cifar10_trainval_combo/{member_idx}.png"
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    axes2[0, i].imshow(image.permute(1, 2, 0))
    axes2[0, i].axis('off')
    if i == 0:
        axes2[0, i].set_title('Members', fontsize=12)

for i in range(10):
    nonmember_idx = member_splits["mia_eval_idxs"][i]
    img_path = f"data/temp/cifar10_trainval_combo/{nonmember_idx}.png"
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    axes2[1, i].imshow(image.permute(1, 2, 0))
    axes2[1, i].axis('off')
    if i == 0:
        axes2[1, i].set_title('Non-members', fontsize=12)

plt.tight_layout()
plt.savefig('experiments/cifar10_outofdist_qmia/cifar10_member_nonmember_samples_neil.png')