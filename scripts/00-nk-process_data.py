import os

if __name__ == '__main__':
    process_celeba(name='c')
    process_celeba(traits={"Bald": 1}, name="bald")
    process_celeba(traits={"Eyeglasses": 1}, name="glasses")
    process_celeba(traits={"Bald":-1}, name="hair")
    process_celeba(traits={"Eyeglasses":-1}, name="eyes")
    process_imagenet(name='inet')
    process_imagenet(wnid="n02099601", name="dog")
    process_imagenet(wnid="n01632777", name="axl")
    process_imagenet(wnid="n06359193", name="site")
    process_hdasynchildfaces(age_group=-1, name="hc")
    process_hdasynchildfaces(age_group=4, name="ha")

    for entry in os.listdir('data/datasets/gts'):
        create_mid_masks(entry, p=0.5, overwrite=True)
        create_mid_masks(entry, p=0.2, overwrite=True)
        create_evNpx_masks(entry, N=2, overwrite=True)
    