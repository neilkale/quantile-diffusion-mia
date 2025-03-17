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

member_data = np.load(SECMI_EVALS_BASE + 'secmi_results_t50.npz') # Changed to full recon
member_diffusions = member_data['member_diffusions']
member_internal_samples = member_data['member_internal_samples']
nonmember_diffusions = member_data['nonmember_diffusions'] ## TODO: should this be nonmember_data...?
nonmember_internal_samples = member_data['nonmember_internal_samples']

member_diffusions = torch.from_numpy(member_diffusions)
member_internal_samples = torch.from_numpy(member_internal_samples)
nonmember_diffusions = torch.from_numpy(nonmember_diffusions)
nonmember_internal_samples = torch.from_numpy(nonmember_internal_samples)

def distance(diffusions, internal_samples):
    return torch.log10((diffusions - internal_samples).pow(2).sum(dim=(1,2,3))).view(-1, 1)

member_distance = distance(member_diffusions[:10], member_internal_samples[:10])
nonmember_distance = distance(nonmember_diffusions[:10], nonmember_internal_samples[:10])
print("Distance between first 10 member diffusions and internal samples (shuai):")
print(member_distance)
print("Distance between first 10 nonmember diffusions and internal samples (shuai):")
print(nonmember_distance)

######################

# Load the torch results dictionary
t_results = torch.load("experiments/cifar10_outofdist_qmia/t_errors/t_results.pt")
results_indices = t_results["indices"]

# Get the first 10 member and nonmember indices from the splits
member_target_idxs = member_splits["mia_train_idxs"][:10]
nonmember_target_idxs = member_splits["mia_eval_idxs"][:10]

def transform_idx(idx):
    return f'data/temp/cifar10_trainval_combo/{idx}.png'    

# Find the corresponding positions in the results using the indices list
member_positions = [results_indices.index(transform_idx(idx)) for idx in member_target_idxs]
nonmember_positions = [results_indices.index(transform_idx(idx)) for idx in nonmember_target_idxs]

# Extract images, diffusions, and internal samples for members and nonmembers
member_images = t_results["images"][member_positions]
nonmember_images = t_results["images"][nonmember_positions]
member_diffusions = t_results["diffusions"][member_positions]
nonmember_diffusions = t_results["diffusions"][nonmember_positions]
member_internal_samples = t_results["internal_samples"][member_positions]
nonmember_internal_samples = t_results["internal_samples"][nonmember_positions]

# Calculate and print distances
def distance(diffusions, internal_samples):
    return torch.log10((diffusions - internal_samples).pow(2).sum(dim=(1,2,3))).view(-1, 1)

member_distance = distance(member_diffusions, member_internal_samples)
nonmember_distance = distance(nonmember_diffusions, nonmember_internal_samples)

print("Distance between member diffusions and internal samples from t_results (neil):")
print(member_distance)
print("Distance between nonmember diffusions and internal samples from t_results (neil):")
print(nonmember_distance)

import pdb; pdb.set_trace()  # Debugging line to inspect the split results
