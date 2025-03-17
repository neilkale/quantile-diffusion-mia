import numpy as np
import torch

# File paths
folder = 'experiments/cifar10_outofdist_qmia/t_errors/cifar10/'
idxs_file = 'CIFAR10_train_ratio0.5.npz'
t_results_file = 't_results.pt'  # adjust if needed

# Load the indices npz file
data = np.load(folder + idxs_file)
mia_train_idxs = data['mia_train_idxs']
mia_eval_idxs = data['mia_eval_idxs']
    
# Load t_results dict from the .pt file
t_results = torch.load(folder + t_results_file)

# Extract the 'indices' tensor to use for validation of dimensions
indices = t_results['indices']

# Create dictionaries to hold the split results
t_results_train = {}
t_results_eval = {}

def transform_idx(idx):
    return f'data/temp/cifar10_trainval_combo/{idx}.png'    

eval_positions = [indices.index(transform_idx(idx)) for idx in mia_eval_idxs]
train_positions = [indices.index(transform_idx(idx)) for idx in mia_train_idxs]

for key, tensor in t_results.items():
    print(f"Processing key: {key}")
    # Check if the entry is a tensor and its first dimension matches the length of indices.
    if isinstance(tensor, torch.Tensor) and tensor.size(0) == len(indices):
        t_results_train[key] = t_results[key][train_positions]
        t_results_eval[key] = t_results[key][eval_positions]
    else:
        # For non-tensor entries or those that don't match the expected shape, simply copy over.
        t_results_train[key] = [tensor[i] for i in train_positions]
        t_results_eval[key] = [tensor[i] for i in eval_positions]

# Optionally, save the split results to new files
torch.save(t_results_train, folder + 't_results_train.pt')
torch.save(t_results_eval, folder + 't_results_eval.pt')
