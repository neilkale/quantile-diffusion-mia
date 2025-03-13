# Pretrained Models List

1. c256_250000

    Description: This model was taken from **RePaint** (2022).

2. inet256_classifier, inet256_diffusion, inet_256_diffusion_uncond

    Description: This model family was used in RePaint, but originally trained in **Diffusion Models Beat GANs on Image Synthesis** (2021). 

# Finetuned Models List

1. celeba256_ftdog

    Description: We finetune c256_250000 on dog256_train, which contains images from the Golden Retriever class of Imagenet train subset.

2. celeba256_fthda

    Description: We finetune c256_250000 on hc256_train and ha256_train, which contain a subset of child and adult images from HDA-SynChildFaces.

_Add additional models as needed and update their descriptions._