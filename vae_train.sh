#!/bin/bash

python vae_train.py\
    --exp=test_vit_vae_mvtec_screw\
    --dataset=mvtec\
    --category=wood\
    --lr=1e-3\
    --num_epochs=100\
    --img_size=224\
    --batch_size=8\
    --batch_size_test=8\
    --latent_img_size=16\
    --z_dim=256\
    --beta=1\
    --nb_channels=3\
    --model=vitvae\
    --corr_type=corr_id\
    --force_train\
    --intest

