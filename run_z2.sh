#!/bin/bash
Z=/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan
cd $Z/dreamzero_sp
source $Z/auto_research_lightningrl/env.sh >/dev/null 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$Z/dreamzero_sp/shim:$Z/dreamzero_sp/dreamzero:$PYTHONPATH"
exec python -m torch.distributed.run --nproc_per_node=4 --master_port=29641   train_real.py --frames 33 --steps 4 --lr 4e-3 --zero2 1 --order 4
