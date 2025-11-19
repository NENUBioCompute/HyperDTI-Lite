# import torch
# from geoopt import PoincareBall
#
# def batched_weighted_midpoint(x: torch.Tensor, w: torch.Tensor, manifold: PoincareBall):
#     """
#     Batched 版本的 weighted_midpoint，适用于形状：
#     x: [B, D_out, D_in, dim]
#     w: [B, D_out, D_in]
#     作用：对于每个输出维度 D_out，计算 D_in 个向量的加权双曲均值
#     返回: [B, D_out, dim]
#     """
#     logmap = manifold.logmap0(x)                          # [B, D_out, D_in, dim]
#     w = w.unsqueeze(-1)                                   # [B, D_out, D_in, 1]
#     weighted = (w * logmap).sum(dim=2)                    # [B, D_out, dim]
#     return manifold.expmap0(weighted)                     # [B, D_out, dim]
#
#
# import torch.nn as nn
# import geoopt
# from geoopt import ManifoldParameter
#
# class HyperbolicSelfAttention(nn.Module):
#     def __init__(self, dim, c=1.0):
#         super().__init__()
#         self.dim = dim
#         self.manifold = geoopt.PoincareBall(c=c)
#
#         self.q_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
#         self.k_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
#         self.v_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
#
#     def forward(self, x):
#         B, D = x.shape
#         manifold = self.manifold
#         eps = 1e-5
#         x = x.view(B, D, 1)
#         Q = manifold.mobius_matvec(self.q_proj, x)  # [B, D, 1]
#         K = manifold.mobius_matvec(self.k_proj, x)
#         V = manifold.mobius_matvec(self.v_proj, x)
#         dist = manifold.dist(Q.unsqueeze(2), K.unsqueeze(1))  # [B, D, D]
#         attn_weights = 1.0 / (dist + eps)                     # [B, D, D]
#         # attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
#
#         # V: [B, 1, D] → [B, D, D]
#         V = V.transpose(1, 2).expand(B, D, D)                 # [B, D_out, D_in]
#         V = V.unsqueeze(-1)                                   # [B, D_out, D_in, 1]
#         V = manifold.projx(V)                                 # 投影到球内
#         output = batched_weighted_midpoint(V, attn_weights, manifold)  # [B, D, 1]
#         return output.squeeze(-1)  # [B, D]
#
# import torch
# from geoopt import PoincareBall
#
# # 参数定义
# batch_size = 4
# dim = 16
# curvature = 1.0
#
# # 创建示例输入：双曲空间中点，shape: [B, D]
# manifold = PoincareBall(c=curvature)
# x = torch.randn(batch_size, dim)
# x = manifold.projx(x)  # 投影到庞加莱球，确保合法性
#
# # 初始化双曲自注意力模块
# hyper_attn = HyperbolicSelfAttention(dim=dim, c=curvature)
#
# # 前向传播
# output = hyper_attn(x)  # shape: [B, D]
#
# # 打印结果
# print("Input x shape:", x.shape)
# print("Output shape:", output.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义模块
class EuclideanCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.q1_proj = nn.Linear(dim, dim)
        self.k2_proj = nn.Linear(dim, dim)
        self.v2_proj = nn.Linear(dim, dim)

        self.q2_proj = nn.Linear(dim, dim)
        self.k1_proj = nn.Linear(dim, dim)
        self.v1_proj = nn.Linear(dim, dim)

        self.scale = dim ** 0.5

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)  # [B, 1, D]
        x2 = x2.unsqueeze(1)  # [B, 1, D]

        # x1 attends to x2
        Q1 = self.q1_proj(x1)
        K2 = self.k2_proj(x2)
        V2 = self.v2_proj(x2)

        attn1_weights = torch.matmul(Q1, K2.transpose(-2, -1)) / self.scale
        attn1 = F.softmax(attn1_weights, dim=-1)
        y1 = torch.matmul(attn1, V2).squeeze(1)

        # x2 attends to x1
        Q2 = self.q2_proj(x2)
        K1 = self.k1_proj(x1)
        V1 = self.v1_proj(x1)

        attn2_weights = torch.matmul(Q2, K1.transpose(-2, -1)) / self.scale
        attn2 = F.softmax(attn2_weights, dim=-1)
        y2 = torch.matmul(attn2, V1).squeeze(1)

        return y1, y2

# === 创建输入并测试 ===
dim = 128
x1 = torch.randn(5, dim)  # [B=1, D=128]
x2 = torch.randn(5, dim)

model = EuclideanCrossAttention(dim)
y1, y2 = model(x1, x2)

print("y1 shape:", y1.shape)  # Expected: [1, 128]
print("y2 shape:", y2.shape)  # Expected: [1, 128]
print("y1 sample:", y1[0, :5])  # 打印前5维示例
print("y2 sample:", y2[0, :5])
