# -*- coding: utf-8 -*-
"""MagWorld v30 = v25 + 一步到位 JEPA（单阶段联合训练，无预训练/微调两阶段）.

JEPA 组件（吸取 s515/s516 两轮"真空 loss"教训后的第三版）：

  1. EMA y-encoder（V-JEPA 核心）:
     y 路径 = context_encoder 的 EMA 影子，对真实扰动后表达
     x_p = x_ctrl + Δx_true 编码出目标潜表示 z_y。
     EMA 影子是 requires_grad=False 的 deepcopy 子模块：
       · .to(device) 自动跟随
       · state_dict 包含（体积小，无害），v4 init 加载时缺失由 ema_sync 兜底
       · AdamW 不优化（requires_grad=False）
     每 optimizer.step 后 ema_update()：q ← m·q + (1−m)·p。

  2. 潜空间预测 loss（JEPA 主损失，z 空间，d_z=64）:
     L_jepa = 1 - cos( z(x_ctrl+Δx̂),  sg[z_EMA(x_ctrl+Δx_true)] ), per-sample
     stop-grad 在 EMA 目标上 → 不对称表示学习，防塌缩。

     【为什么前两版真空】v25 的 per-gene 状态 s = state_gene(e) + state_ctx(z)
     是「每基因项 + 每样本项」的直和，无交互：per-(样本,基因) cosine 被
     常量项主导（s515）；基因维中心化把样本项精确减掉了（s516）。
     z 空间（context 潜变量）才是有真实样本交互的表示空间。

  3. 潜空间 Δx 对齐（基因嵌入空间 cosine）:
     L_latent = 1 - cos( Δx̂·E, Δx_true·E ), E = 单位化基因嵌入。

正确用法：去掉 --freeze-base，让 context_encoder 参与训练（lr 建议 5e-5），
JEPA loss 才有梯度路径。配 --freeze-base 时 L_jepa 对可训练参数无梯度
（z_p 不依赖任何可训练参数），会自动退化为纯 L_latent。

trainer flags: --lambda-jepa 0.1 --lambda-latent 0.1 --jepa-ema 0.996
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_world_h1_v25 import WorldModelH1V25


class WorldModelH1V30(WorldModelH1V25):
    def __init__(
        self,
        *args,
        jepa_ema: float = 0.996,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.jepa_ema = float(jepa_ema)
        # EMA 影子（y-encoder）：deepcopy + 冻结
        self.ema_ctx = copy.deepcopy(self.context_encoder)
        for p in self.ema_ctx.parameters():
            p.requires_grad_(False)
        self._pairs = [
            (p, q)
            for (_, p), (_, q) in zip(
                self.context_encoder.named_parameters(),
                self.ema_ctx.named_parameters(),
            )
        ]

    # ---------------------------------------------------------- EMA helpers
    @torch.no_grad()
    def ema_sync(self) -> None:
        """trainer 在 init_from 加载后调用，对齐 EMA 影子。"""
        for p, q in self._pairs:
            q.data.copy_(p.data)

    @torch.no_grad()
    def ema_update(self) -> None:
        m = self.jepa_ema
        for p, q in self._pairs:
            q.data.mul_(m).add_(p.data, alpha=1.0 - m)

    # ---------------------------------------------------------------- JEPA
    def _encode(self, x: torch.Tensor, ema: bool = False) -> torch.Tensor:
        G = self.gene_emb.weight.shape[0]
        ctx_in = F.layer_norm(x, (G,)) if self.normalize_context else x
        return self.ema_ctx(ctx_in) if ema else self.context_encoder(ctx_in)

    def jepa_loss(self, x_ctrl: torch.Tensor, delta_true: torch.Tensor,
                  prediction: torch.Tensor) -> torch.Tensor:
        """L = 1 - cos( z(x_ctrl+Δx̂), sg[z_EMA(x_ctrl+Δx_true)] ), per-sample."""
        with torch.no_grad():
            x_true = (x_ctrl + delta_true).clamp_min(0.0)
            z_y = self._encode(x_true, ema=True)
        x_hat = (x_ctrl + prediction).clamp_min(0.0)
        z_p = self._encode(x_hat, ema=False)
        return (2.0 - 2.0 * F.cosine_similarity(z_p, z_y, dim=-1)).mean()

    def latent_delta_loss(self, delta_true: torch.Tensor,
                          prediction: torch.Tensor) -> torch.Tensor:
        """L = 1 - cos( Δx̂·E, Δx_true·E ), E = 单位化基因嵌入。"""
        e = F.normalize(self.gene_emb.weight, dim=-1)
        p_hat = F.normalize(prediction @ e, dim=-1)
        p_true = F.normalize(delta_true @ e, dim=-1)
        return (2.0 - 2.0 * (p_hat * p_true).sum(-1)).mean()


WorldModel = WorldModelH1V30
