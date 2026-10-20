#!/usr/bin/env python3
"""从 checkpoint header 反推真实视觉 token 数（不加载权重）。"""
import json, struct

Z = "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan/model"
d = json.load(open(Z + "/model.safetensors.index.json"))
wm = d["weight_map"]


def hdr(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


want = [k for k in wm if ("patch_embedding" in k) or ("img_emb" in k) or ("vae." in k and "conv" in k)]
print("n_keys_total", len(wm))
cache = {}
for k in want[:12]:
    f = wm[k]
    p = Z + "/" + f
    if p not in cache:
        cache[p] = hdr(p)
    h = cache[p].get(k)
    if h:
        print("  %-62s shape=%s dtype=%s" % (k, h.get("shape"), h.get("dtype")))

# 找 DiT block 的 dim / 层数
import re
layers = sorted({int(m.group(1)) for k in wm for m in [re.search(r"blocks\.(\d+)\.", k)] if m})
print("\nDiT block count:", (max(layers) + 1) if layers else "?")
print("sample DiT keys:", [k for k in wm if "blocks.0." in k][:4])
