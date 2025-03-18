import numpy as np
import torch

np.random.seed(1111)

folder = 'experiments/cifar10_outofdist_qmia/t_errors/cifar10/'

#####

t_results = torch.load(folder + 't_results_eval.pt')
all_classes = t_results['labels']

for excluded_class in np.arange(0, 10):
    output_file = f'{folder}t_results_eval_excludeclass{excluded_class}_22500.pt'
    indices = np.isin(all_classes, excluded_class, invert=True)

    filtered_t_results = {}
    for key, tensor in t_results.items():
        if isinstance(tensor, torch.Tensor) and tensor.size(0) == len(indices):
            filtered_t_results[key] = t_results[key][indices]
        else:
            filtered_t_results[key] = [tensor[i] for i in indices]

    torch.save(filtered_t_results, output_file)
    print(f"Saved filtered results to {output_file} excluding class: {excluded_class}. Total samples: {len(filtered_t_results['labels'])}")