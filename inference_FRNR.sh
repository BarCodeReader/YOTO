#!/bin/bash

python inference_FRNR.py \
--root_dir "your/root/dir" \
--network "model" \
--dataset "csiq" \
--GPU 0 \
--batch_size 16 \
--num_avg_val 5 \
--checkpoint "your/ckpt.pth" \
--infer_mode "NR"