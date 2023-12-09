#!/bin/bash

python vae_train.py\
    --exp=test_vqvae_miad_wind_turbine\
    --dataset=miad\
    --category=wood\
    --lr=1e-4\
    --num_epochs=100\
    --img_size=256\
    --batch_size=8\
    --batch_size_test=8\
    --latent_img_size=32\
    --z_dim=256\
    --beta=1\
    --nb_channels=3\
    --model=vq_vae\
    --corr_type=corr_id\
    --force_train\
    --intest

