#!/bin/bash
# 从干净备份 .bak_v1 重建全部补丁（四层，顺序不能变）
#   v3: 按 frame 切分的基础实现（会被后面覆盖，但提供了 _sp_segs/_sp_loc 等骨架）
#   v4: 给集合通信加 autograd（反向用逆置换）
#   v5: 零复制分段（首帧/body/action 三段均分，取代复制段）
#   v6: 多段 a2a 合并成一次扁平 all_to_all_single
set -e
Z=/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan
cd $Z/dreamzero_sp
F=$Z/dreamzero_sp/dreamzero/groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py
[ -f "$F.bak_v1" ] || { echo "缺少干净备份 $F.bak_v1"; exit 1; }
python3 sp_v3_patch.py      > /tmp/p3.log && echo "[1/4] v3 ok"
python3 sp_v4_grad.py       > /tmp/p4.log && echo "[2/4] v4 ok"
python3 sp_v5_zerocopy.py   > /tmp/p5.log && echo "[3/4] v5 ok"
python3 sp_v6_flat.py       > /tmp/p6.log && echo "[4/4] v6 ok"
grep -c '_SPFlatA2AFn\|_SPTokenGather\|_SPA2AFn' $F | xargs echo 'patch markers:'
python3 -c "compile(open('$F').read(),'m','exec'); print('syntax ok')"
