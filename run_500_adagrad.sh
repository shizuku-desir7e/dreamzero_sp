#!/bin/bash
Z=/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan
cd $Z/dreamzero_sp
source $Z/auto_research_lightningrl/env.sh >/dev/null 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$Z/dreamzero_sp/shim:$Z/dreamzero_sp/dreamzero:$PYTHONPATH"
exec python -m torch.distributed.run --nproc_per_node=4 --master_port=29740 \
  train_real_data.py --frames 33 --steps 500 --lr 1e-4 --zero2 1 --opt adagrad_bf16 \
  --order 4 --eval_after 0 --log_every 10 \
  --out $Z/dreamzero_sp/results/real_data_adagrad500.json \
  --save_ckpt $Z/dreamzero_sp/results/real_data_500_adagrad.pt
