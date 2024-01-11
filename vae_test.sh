#!/bin/bash

python vae_test.py\
    --exp=test_vit_vae_mvtec_metal_nut\
    --dataset=mvtec\
    --lr=1e-3\
    --img_size=224\
    --batch_size=8\
    --batch_size_test=8\
    --latent_img_size=16\
    --z_dim=256\
    --beta=1\
    --nb_channels=3\
    --model=vitvae\
    --corr_type=corr_id\
    --params_id=100\
