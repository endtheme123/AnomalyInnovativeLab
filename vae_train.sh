#!/bin/bash

python vae_train.py\
    --exp=test_vitvae_mvtec_tile\
    --dataset=mvtec\
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
    --model=vitvae\
    --corr_type=corr_id\
    --force_train\
    --intest

