#!/bin/bash
Z=/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan
cd $Z/dreamzero_sp
source $Z/auto_research_lightningrl/env.sh >/dev/null 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$Z/dreamzero_sp/shim:$Z/dreamzero_sp/dreamzero:$PYTHONPATH"
exec python -m torch.distributed.run --nproc_per_node=4 --master_port=29684 \
  opt_test.py --frames 33 --steps 6 --lr 1e-4 --no_load 1 \
  --opts adamw_bf16,adagrad_bf16,adafactor
