import os
import argparse



def process_data(dataset, name, traits=None, wnid=None, age_group=None):
    if dataset == 'celeba':
        process_celeba(name=name, traits=traits)
    elif dataset == 'imagenet':
        process_imagenet(name=name, wnid=wnid)
    elif dataset == 'hda-syn-child-faces':
        process_hdasynchildfaces(name=name, age_group=age_group)
    else:
        raise ValueError('Dataset not found')

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, required=True)
    args.add_argument('--name', type=str, required=True)
    args.add_argument('--traits', type=str, default=None, help='JSON traits for celeba dataset')
    args.add_argument('--wnid', type=str, default=None, help='WordNet ID for imagenet dataset')
    args.add_argument('--age_group', type=int, default=None, help='Age group for hda-syn-child-faces dataset')

    args = args.parse_args()
    process_data(dataset=args.dataset, name=args.name, traits=args.traits, wnid=args.wnid, age_group=args.age_group)
    