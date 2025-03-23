#!/bin/bash

# process_data --dataset='celeba' --name='c'
# process_data --dataset='celeba' --name='bald'       --traits='{"Bald": 1}' 
# process_data --dataset='celeba' --name='glasses'    --traits='{"Eyeglasses": 1}' 
# process_data --dataset='celeba' --name='hair'       --traits='{"Bald":-1}' 
# process_data --dataset='celeba' --name='eyes'       --traits='{"Eyeglasses":-1}' 

# process_data --dataset='inet'   --name='inet'
# process_data --dataset='inet'   --name='dog'        --wnid='n02099601' 
# process_data --dataset='inet'   --name='axl'        --wnid='n01632777' 
# process_data --dataset='inet'   --name='site'       --wnid='n06359193' 

# process_data --dataset='hda'    --name='hc'         --age_group=-1  
# process_data --dataset='hda'    --name='ha'         --age_group=4 

# python scripts/01-nk-process_data.py --dataset='cifar10'  --name='cifar10' --overwrite

python scripts/01-nk-process_data.py --dataset='cifar100'  --name='cifar100' --overwrite