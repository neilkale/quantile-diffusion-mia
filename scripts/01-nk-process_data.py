import os
import argparse
import shutil
import pickle
import random
import numpy as np
from PIL import Image
import pandas as pd

def process_cifar10(name, output_dir):
    # Clear the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    source_dir = 'data/raw/cifar10/cifar-10-batches-py'

    meta_file = os.path.join(source_dir, 'batches.meta')
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    label_names = [t.decode('utf-8') for t in meta[b'label_names']]

    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch_file = os.path.join(source_dir, f'data_batch_{i}')
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']  # numpy array of shape (10000, 3072)
        labels = batch[b'labels']  # list of 10000 labels
        train_data.append(data)
        train_labels.extend(labels)
    train_data = np.concatenate(train_data, axis=0)

    test_batch_file = os.path.join(source_dir, 'test_batch')
    with open(test_batch_file, 'rb') as f:
        test_batch = pickle.load(f, encoding='bytes')
    test_data = test_batch[b'data']  # shape (10000, 3072)
    test_labels = test_batch[b'labels']

    num_train = train_data.shape[0]  # 50000 images
    indices = list(range(num_train))
    random.shuffle(indices)
    split_point = int(0.8 * num_train)  # 40000 for train

    train_idx = indices[:split_point]   # 40000 indices
    val_idx = indices[split_point:]       # 10000 indices
    test_idx = list(range(num_train, num_train + test_data.shape[0]))  # 10000 indices

    # Concatenate training and test data into a single array of 60000 images
    data = np.concatenate([train_data, test_data], axis=0)

    # Save images into their respective folders
    for split_name, idx in zip(['train', 'val', 'test'], [train_idx, val_idx, test_idx]):
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for i in idx:
            img = Image.fromarray(data[i].reshape(3, 32, 32).transpose(1, 2, 0))
            img.save(os.path.join(split_dir, f'{i}.png'))
    
    # Save label names
    with open(os.path.join(output_dir, 'label_names.txt'), 'w') as f:
        f.write('\n'.join(label_names))
    
    # Create a CSV for labels
    labels = [train_labels[i] for i in train_idx + val_idx] + test_labels
    all_indices = train_idx + val_idx + test_idx
    df = pd.DataFrame({'index': all_indices, 'label': labels})
    df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)
    
def process_data(dataset, name, traits=None, wnid=None, age_group=None, overwrite=False):

    output_dir = f'data/processed/{name}'
    if not overwrite and os.path.exists(output_dir):
        print(f'Dataset <{name}> already processed. Skipping...')
        return
    else:
        os.makedirs(output_dir, exist_ok=True)

    if dataset == 'celeba':
        process_celeba(name=name, traits=traits, output_dir=output_dir)
    elif dataset == 'imagenet':
        process_imagenet(name=name, wnid=wnid, output_dir=output_dir)
    elif dataset == 'hda-syn-child-faces':
        process_hdasynchildfaces(name=name, age_group=age_group, output_dir=output_dir)
    elif dataset == 'cifar10':
        process_cifar10(name=name, output_dir=output_dir)
    else:
        raise ValueError('Dataset not found')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, required=True)
    args.add_argument('--name', type=str, required=True)
    args.add_argument('--traits', type=str, default=None, help='JSON traits for celeba dataset')
    args.add_argument('--wnid', type=str, default=None, help='WordNet ID for imagenet dataset')
    args.add_argument('--age_group', type=int, default=None, help='Age group for hda-syn-child-faces dataset')
    args.add_argument('--overwrite', action='store_true')

    args = args.parse_args()
    process_data(dataset=args.dataset, name=args.name, traits=args.traits, wnid=args.wnid, age_group=args.age_group, overwrite=args.overwrite)
    