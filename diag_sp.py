import os, sys
import torch
import torch.distributed as dist
Z = os.environ.get("ZROOT", "/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/zhengmenghan")
sys.path.insert(0, Z + "/dreamzero_sp/shim")
sys.path.insert(0, Z + "/dreamzero_sp/dreamzero")
import groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk as DZMOD
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import CausalWanModel
_orig_blk = DZMOD.CausalWanAttentionBlock.forward
def _unwrap(self, x, *a, **k):
    out = _orig_blk(self, x, *a, **k)
    return out[0] if isinstance(out, tuple) else out
DZMOD.CausalWanAttentionBlock.forward = _unwrap
FSL=880; F,H,W=int(os.environ["FR"]),40,88; N=F*FSL
CFG = dict(model_type="ti2v", patch_size=(1,2,2), frame_seqlen=FSL, text_len=512,
           in_dim=36, dim=512, ffn_dim=2048, freq_dim=256, text_dim=4096, out_dim=16,
           num_heads=8, num_layers=2, num_frame_per_block=2, action_dim=32,
           num_action_per_block=24, num_state_per_block=1, max_state_dim=64,
           max_num_embodiments=32, hidden_size=1024, concat_first_frame_latent=True,
           qk_norm=True, cross_attn_norm=True, max_chunk_size=4, sink_size=0)
dev="cuda:0"
torch.manual_seed(0)
m = CausalWanModel(**CFG).to(dev).to(torch.bfloat16); m.eval(); m.gradient_checkpointing=False
for p in m.parameters(): p.requires_grad_(False)
torch.manual_seed(1234)
x = torch.randn(1,36,F,H,W,device=dev,dtype=torch.bfloat16)
ctx = torch.randn(1,512,4096,device=dev,dtype=torch.bfloat16)
t = torch.randint(0,1000,(1,F),device=dev)
print("[in ] x nan=%d inf=%d absmax=%.3f" % (x.isnan().sum(), x.isinf().sum(), x.abs().max()), flush=True)
DZMOD.set_sp_state(group=None, world=1, rank=0)
with torch.no_grad():
    out = m(x, t, None, ctx, seq_len=N, clean_x=x.clone(), action=None, state=None, embodiment_id=None)
v = out[0]
print("[out] shape=%s nan=%d inf=%d absmax=%.3f" % (tuple(v.shape), v.isnan().sum(), v.isinf().sum(), v.abs().max()), flush=True)
