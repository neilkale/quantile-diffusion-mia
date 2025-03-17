import numpy as np
import torch
import os
import pickle

np.random.seed(1111)

folder = 'experiments/cifar10_outofdist_qmia/t_errors/cifar10/'

excluded_classes = []
for i in range(0,10):
    classes = np.random.choice(np.arange(10), i, replace=False)
    excluded_classes.append([classes])

output_file = os.path.join(folder, 'excluded_classes.npz')
np.savez(output_file, excluded_classes=np.array(excluded_classes, dtype=object))

#####

t_results = torch.load(folder + 't_results_eval.pt')
all_classes = t_results['labels']

for i, classes in enumerate(excluded_classes):
    output_file = f'{folder}t_results_eval_exclude{i}_2000.pt'
    indices = np.isin(all_classes, classes[0], invert=True)

    filtered_t_results = {}
    for key, tensor in t_results.items():
        if isinstance(tensor, torch.Tensor) and tensor.size(0) == len(indices):
            filtered_t_results[key] = t_results[key][indices]
        else:
            filtered_t_results[key] = [tensor[i] for i in indices]

    # Downsize the filtered results to 2000 samples, evenly distributed across classes
    labels = filtered_t_results['labels'].numpy()
    unique_classes = np.unique(labels)
    samples_per_class = 2000 // len(unique_classes)    
    
    selected_indices = []
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        chosen = np.random.choice(cls_indices, samples_per_class, replace=False)
        selected_indices.extend(chosen)
    selected_indices = np.array(selected_indices)
    
    # Update filtered_t_results with the selected indices
    for key, tensor in filtered_t_results.items():
        if isinstance(tensor, torch.Tensor) and tensor.size(0) == len(labels):
            filtered_t_results[key] = tensor[selected_indices]
        else:
            filtered_t_results[key] = [tensor[i] for i in selected_indices]

    torch.save(filtered_t_results, output_file)
    print(f"Saved filtered results to {output_file} excluding classes: {classes[0]}. Total samples: {len(selected_indices)}")
   

    