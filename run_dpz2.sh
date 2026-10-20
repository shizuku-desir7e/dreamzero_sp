#!/bin/bash
Z=/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan
cd $Z/dreamzero_sp
source $Z/auto_research_lightningrl/env.sh >/dev/null 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$Z/dreamzero_sp/shim:$Z/dreamzero_sp/dreamzero:$PYTHONPATH"
exec python -m torch.distributed.run --nproc_per_node=4 --master_port=29661   train_real.py --no_load 1 --zero2 1 --order 1 --dp_steps 1   --frames 9 --frames_list 9,17,33 --lr 4e-3
