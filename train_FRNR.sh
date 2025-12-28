#!/bin/bash

python train_FRNR.py \
--root_dir "/your/root/dir" \
--network "model" \
--GPU 0 \
--model_name "FRNR_mix" \
--dataset "csiq" \
--learning_rate 1e-4 \
--save_sh "./train_FRNR.sh" \
--n_epoch 600 \
--T_max 50 \
--batch_size 16 \
--num_avg_val 20 \
--output_path "./output_mix" \
--training_mode "mix"
