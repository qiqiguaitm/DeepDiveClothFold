"""P2: LMWMasLAM adapter —— 用我们的 LMWM 替换 LaWM 的世界模型 decoder,塞进 starVLA。

最小干净 swap(契约见 lmwm/docs/LAM_starVLA_contract_2026-07-12.md):
  - 保留原 LAM 的 extract_vision_features(DINOv3-vitb16, 共享) + 全部属性不变。
  - 只把 lam.decoder 换成 LMWM MilestoneGenerator(预测 next-MILESTONE 特征, 替 LaWM 的 next-frame)。
  - 可选把 get_latent_action(teacher) 换成 LMWM InverseEnc; 否则关 enable_loss_distill。
→ 唯一变量 = 世界模型预测目标(next-milestone vs next-frame)。

用法: 在 build_lawam framework 前 monkeypatch,或改 config 让 load_latent_action_model 返回 make_lmwm_lam(...)。
"""
import torch, torch.nn as nn


class MilestoneGenerator(nn.Module):  # = 训好的 LMWM 生成器(din=768,code_dim=32)
    def __init__(self, din, code_dim, hid=512, nblk=4):
        super().__init__()
        self.nblk, self.hid = nblk, hid
        self.proj = nn.Conv2d(din, hid, 3, 1, 1)
        self.gn = nn.ModuleList([nn.GroupNorm(8, hid) for _ in range(nblk)])
        self.blk = nn.ModuleList([nn.Sequential(nn.Conv2d(hid, hid, 3, 1, 1), nn.GELU(), nn.Conv2d(hid, hid, 3, 1, 1)) for _ in range(nblk)])
        self.mod = nn.Linear(code_dim, nblk * 3 * hid)
        self.out = nn.Conv2d(hid, din, 3, 1, 1)

    def forward(self, gt, code):  # gt[B,din,P,P], code[B,code_dim] -> [B,din,P,P]
        h = self.proj(gt); m = self.mod(code).view(-1, self.nblk, 3, self.hid)
        for i in range(self.nblk):
            sh, sc, ga = m[:, i, 0], m[:, i, 1], m[:, i, 2]
            hn = self.gn[i](h) * (1 + sc[:, :, None, None]) + sh[:, :, None, None]
            h = h + ga[:, :, None, None] * self.blk[i](hn)
        return self.out(h)


class InverseEnc(nn.Module):  # teacher: (g_t,g_f)->code (可选 distill)
    def __init__(self, din, code_dim, hid=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2 * din, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU(),
            nn.Conv2d(hid, hid, 3, 2, 1), nn.GroupNorm(8, hid), nn.GELU())
        self.head = nn.Linear(hid, code_dim); self.ln = nn.LayerNorm(code_dim)

    def forward(self, gt, gf):
        return self.ln(self.head(self.conv(torch.cat([gt, gf], 1)).mean((2, 3))))


class LMWMDecoder(nn.Module):
    """契约 C2 的 decoder 替身: (features[B,K,D], actions[B,1,code_dim]) -> [B,1,K,D]。
    内部 reshape 到 grid 喂 MilestoneGenerator(预测 next-milestone 特征)。"""
    def __init__(self, gen: MilestoneGenerator, grid_hw=16):
        super().__init__()
        self.gen = gen; self.P = grid_hw

    def forward(self, features, actions):
        B, K, D = features.shape                       # [B,256,768]
        grid = features.transpose(1, 2).reshape(B, D, self.P, self.P)   # [B,768,16,16]
        code = actions.squeeze(1)                       # [B,1,32] -> [B,32]
        out = self.gen(grid, code)                      # [B,768,16,16]
        out = out.reshape(B, D, K).transpose(1, 2)      # [B,256,768]
        return out.unsqueeze(1)                         # [B,1,256,768]


def load_lmwm_parts(lmwm_ckpt, code_dim=32, din=768, grid_hw=16):
    """[V8 dual-scale] 只加载 LMWM 模块, **不 swap** base_lam(局部通道保 LaWM 原样)。
    回 (lmwm_dec, inv): lmwm_dec=LMWMDecoder(生成器, 可训), inv=InverseEnc(teacher, 冻)。
    与 make_lmwm_lam 的区别: 那个原地替换 lam.decoder/get_latent_action(单通道);
    这个把两模块交给调用方(lawam.py)作独立全局通道, 与 LaWM 局部通道并联。"""
    sd = torch.load(lmwm_ckpt, map_location="cpu", weights_only=False)
    gen = MilestoneGenerator(din, code_dim); gen.load_state_dict(sd["gen"])
    lmwm_dec = LMWMDecoder(gen, grid_hw)
    for p in lmwm_dec.parameters(): p.requires_grad = True   # 生成器随 VLA 微调(同单通道 swap)
    inv = InverseEnc(din, code_dim); inv.load_state_dict(sd["inv"])
    for p in inv.parameters(): p.requires_grad = False        # teacher 冻结
    inv.eval()
    return lmwm_dec, inv


def make_lmwm_lam(base_lam, lmwm_ckpt, code_dim=32, din=768, grid_hw=16, swap_teacher=False):
    """接收原 LAM(已由 load_latent_action_model 加载), 换 decoder(+可选 teacher)为 LMWM。
    保留原 extract_vision_features + 属性(code_dim/input_dim/encoder.grid_h/w/num_frames)。"""
    sd = torch.load(lmwm_ckpt, map_location="cpu", weights_only=False)
    # code_dim 校验(BUG_AUDIT MAJOR-2 附加): 原 LAM code 维必须与 LMWM 一致, 否则 vlm_to_lam 输出与 gen.mod 形状冲突
    _base_cd = getattr(base_lam, "code_dim", code_dim)
    assert int(_base_cd) == int(code_dim), f"LAM code_dim {_base_cd} != LMWM code_dim {code_dim}"
    gen = MilestoneGenerator(din, code_dim); gen.load_state_dict(sd["gen"])
    dev = next(base_lam.parameters()).device
    base_lam.decoder = LMWMDecoder(gen, grid_hw).to(dev).eval()
    if swap_teacher:
        inv = InverseEnc(din, code_dim); inv.load_state_dict(sd["inv"])
        # 用 LMWM teacher 产 quantized: (g_t,g_f)->code[B,1,code_dim]
        _orig_gla = base_lam.get_latent_action
        def _gla(videos=None, states=None, dec_videos=None, predict_future_frame=False, embodiment_ids=None, **kw):
            feats = base_lam.extract_vision_features(videos, n=-2)   # [B,T,256,768]
            gt = feats[:, 0].transpose(1, 2).reshape(feats.shape[0], din, grid_hw, grid_hw)
            gf = feats[:, -1].transpose(1, 2).reshape(feats.shape[0], din, grid_hw, grid_hw)
            code = inv(gt, gf).unsqueeze(1)                          # [B,1,code_dim]
            return {"quantized": code}
        base_lam.get_latent_action = _gla
        base_lam._lmwm_inv = inv.to(dev).eval()
    for p in base_lam.parameters(): p.requires_grad = False   # LAM 冻(decoder 由 unfreeze_lam_decoder 解冻)
    for p in base_lam.decoder.parameters(): p.requires_grad = True
    return base_lam
