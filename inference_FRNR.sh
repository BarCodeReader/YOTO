#!/bin/bash

python inference_FRNR.py \
--root_dir "/mnt/bn/bytenas-yunyike/AAA/seg_aqi" \
--network "model" \
--dataset "csiq" \
--GPU 0 \
--batch_size 16 \
--num_avg_val 5 \
--checkpoint "/mnt/bn/bytenas-yunyike/AAA/seg_aqi/output_mix/csiq/FRNR_mix/NR_epoch1_plcc_0.8087_srocc_0.7537.pth" \
--infer_mode "NR"