#!/bin/bash
Z=/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan
cd $Z/dreamzero_sp
source $Z/auto_research_lightningrl/env.sh >/dev/null 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$Z/dreamzero_sp/shim:$Z/dreamzero_sp/dreamzero:$PYTHONPATH"
exec python -m torch.distributed.run --nproc_per_node=4 --master_port=29611   offload_baseline.py --offload 1 --spec 1:9,1:17,1:33 --iters 2 --warmup 1
