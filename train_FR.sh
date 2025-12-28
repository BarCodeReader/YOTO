#!/bin/bash

python train_FRNR.py \
--root_dir "/mnt/bn/bytenas-yunyike/AAA/seg_aqi" \
--network "model" \
--GPU 0 \
--model_name "FR_only" \
--dataset "csiq" \
--learning_rate 1e-4 \
--save_sh "./train_FR.sh" \
--n_epoch 600 \
--T_max 50 \
--batch_size 16 \
--num_avg_val 20 \
--output_path "./output_mix" \
--training_mode "FR"