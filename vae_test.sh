#!/bin/bash

python vae_test.py\
    --exp=test_vae_miad_photovoltaic\
    --dataset=miad\
    --lr=1e-4\
    --img_size=256\
    --batch_size=8\
    --batch_size_test=8\
    --latent_img_size=32\
    --z_dim=256\
    --beta=1\
    --nb_channels=3\
    --model=vae\
    --corr_type=corr_id\
    --params_id=100\
