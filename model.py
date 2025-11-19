import json
import torch.nn as nn
import torch
import geoopt
from geoopt import ManifoldParameter,PoincareBall

def zscore_normalize(x, eps=1e-8):
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    return (x - mean) / (std + eps)


def batched_weighted_midpoint(x: torch.Tensor, w: torch.Tensor, manifold: PoincareBall):
    """
    Batched 版本的 weighted_midpoint，适用于形状：
    x: [B, D_out, D_in, dim]
    w: [B, D_out, D_in]
    作用：对于每个输出维度 D_out，计算 D_in 个向量的加权双曲均值
    返回: [B, D_out, dim]
    """
    logmap = manifold.logmap0(x)                          # [B, D_out, D_in, dim]
    w = w.unsqueeze(-1)                                   # [B, D_out, D_in, 1]
    weighted = (w * logmap).sum(dim=2)                    # [B, D_out, dim]
    return manifold.expmap0(weighted)                     # [B, D_out, dim]


# 打开并读取 JSON 文件
with open(f'./pre_files/DrugBank35022_drug_ChemBert_feature_dict.json', 'r') as f1:
    drug_ChemBert_feature_dict = json.load(f1)
with open(f'./pre_files/DrugBank35022_drug_ECFP_feature_dict.json', 'r') as f2:
    drug_ECFP_feature_dict = json.load(f2)
with open(f'./pre_files/DrugBank35022_drug_MACCS_feature_dict.json', 'r') as f3:
    drug_MACCS_feature_dict = json.load(f3)
# with open(f'./pre_files/DrugBank35022_drug_Morgan_feature_dict.json', 'r') as f4:
#    drug_Morgan_feature_dict = json.load(f4)
with open(f'./pre_files/DrugBank35022_drug_PCproperties_feature_dict.json', 'r') as f5:
    drug_PC_feature_dict = json.load(f5)

with open(f'./pre_files/DrugBank35022_target_ESM_feature_dict.json', 'r') as f6:
    target_ESM_feature_dict = json.load(f6)
with open(f'./pre_files/DrugBank35022_target_mer_feature_dict.json', 'r') as f7:
    target_mer_feature_dict = json.load(f7)
with open(f'./pre_files/DrugBank35022_target_PCproperties_feature_dict.json', 'r') as f8:
    target_PC_feature_dict = json.load(f8)


class MobiusTanh(nn.Module):
    def __init__(self, manifold: geoopt.PoincareBall):
        super().__init__()
        self.manifold = manifold

    def forward(self, x):
        x_tangent = self.manifold.logmap0(x)
        x_activated = torch.tanh(x_tangent)
        x_hyper = self.manifold.expmap0(x_activated)
        return self.manifold.projx(x_hyper)


class HyperbolicLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, c=1.0, bias=True, init_std=1e-2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.ball = geoopt.PoincareBall(c=self.c)
        self.init_std = init_std

        # 权重可以直接用 random_normal（因为是二维）
        self.weight = ManifoldParameter(torch.randn((self.in_features, self.out_features)), manifold=self.ball)
        #  偏置不能用 random_normal（因为是一维），改用 expmap0 初始化
        if bias:
            bias_tangent = torch.randn(out_features) * self.init_std
            bias_init = self.ball.expmap0(bias_tangent)
            self.bias = ManifoldParameter(bias_init, manifold=self.ball)
        else:
            self.bias = None

    def forward(self, x):
        assert x.shape[-1] == self.in_features, f"Expected input features {self.in_features}, got {x.shape[-1]}"
        # 转置 weight：从 (out_features, in_features) -> (in_features, out_features)
        out = self.ball.mobius_matvec(self.weight.transpose(0, 1), x)
        if self.bias is not None:
            out = self.ball.mobius_add(out, self.bias)
        return out


# class HyperbolicSelfAttention(nn.Module):
#     def __init__(self, dim, c=1.0):
#         super().__init__()
#         self.dim = dim
#         self.manifold = geoopt.PoincareBall(c=c)
#
#         # 用于映射特征维度内部 Q, K, V
#         self.q_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
#         self.k_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
#         self.v_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
#
#     def forward(self, x):
#         # x: [B, D]，双曲空间中的点
#         B, D = x.shape
#         manifold = self.manifold
#         eps = 1e-5
#         # 先 reshape 成每个样本 [D, 1]，方便处理维度内部注意力
#         x = x.view(B, D, 1)  # [B, D, 1]
#         # 对每个特征维度做 Möbius 投影（intra-feature attention）
#         Q = manifold.mobius_matvec(self.q_proj, x)  # [B, D, 1]
#         K = manifold.mobius_matvec(self.k_proj, x)  # [B, D, 1]
#         V = manifold.mobius_matvec(self.v_proj, x)  # [B, D, 1]
#         # 构造 D × D 的注意力权重（每个样本独立）
#         dist = manifold.dist(Q.unsqueeze(2), K.unsqueeze(1))  # [B, D, D]
#         attn_weights = 1.0 / (dist + eps)  # 反距离权重，无 softmax
#         # 双曲空间下的数乘
#         V = V.transpose(1, 2)  # [B, 1, D]
#         attn_weights = attn_weights  # [B, D, D]
#         weighted_V = manifold.mobius_scalar_mul(attn_weights, V)  # [B, D, D]
#         # 用 Möbius 加法聚合每行（即对每个输出维度）
#         output = weighted_V[:, :, 0]  # 初始化为第一个列
#         for j in range(1, D):
#             output = manifold.mobius_add(output, weighted_V[:, :, j])  # [B, D]
#         return output  # 输出仍在双曲空间中

class HyperbolicSelfAttention(nn.Module):
    def __init__(self, dim, c=1.0):
        super().__init__()
        self.dim = dim
        self.manifold = geoopt.PoincareBall(c=c)

        self.q_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
        self.k_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
        self.v_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)

    def forward(self, x):
        B, D = x.shape
        manifold = self.manifold
        eps = 1e-5
        x = x.view(B, D, 1)
        Q = manifold.mobius_matvec(self.q_proj, x)  # [B, D, 1]
        K = manifold.mobius_matvec(self.k_proj, x)
        V = manifold.mobius_matvec(self.v_proj, x)
        dist = manifold.dist(Q.unsqueeze(2), K.unsqueeze(1))  # [B, D, D]
        attn_weights = 1.0 / (dist + eps)                     # [B, D, D]
        # attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)

        # V: [B, 1, D] → [B, D, D]
        V = V.transpose(1, 2).expand(B, D, D)                 # [B, D_out, D_in]
        V = V.unsqueeze(-1)                                   # [B, D_out, D_in, 1]
        V = manifold.projx(V)                                 # 投影到球内
        output = batched_weighted_midpoint(V, attn_weights, manifold)  # [B, D, 1]
        return output.squeeze(-1)  # [B, D]


# class HyperbolicCrossAttention(nn.Module):
#     def __init__(self, dim, c=1.0):
#         super().__init__()
#         self.dim = dim
#         self.manifold = geoopt.PoincareBall(c=c)
#         # Q, K, V for x1 attending to x2
#         self.q1_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
#         self.k2_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
#         self.v2_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
#         # Q, K, V for x2 attending to x1
#         self.q2_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
#         self.k1_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
#         self.v1_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
#
#     def forward(self, x1, x2):
#         """
#         x1: [B, D] - 双曲张量
#         x2: [B, D]
#         return: y1, y2 - 双曲张量，shape: [B, D]
#         """
#         manifold = self.manifold
#         B, D = x1.shape
#         eps = 1e-5
#         # ========== Cross Attention 1: x1 attends to x2 ==========
#         # reshape: [B, D] → [B, D, 1] for matvec
#         x1_ = x1.view(B, D, 1)
#         x2_ = x2.view(B, D, 1)
#         Q1 = manifold.mobius_matvec(self.q1_proj, x1_)  # [B, D, 1]
#         K2 = manifold.mobius_matvec(self.k2_proj, x2_)  # [B, D, 1]
#         V2 = manifold.mobius_matvec(self.v2_proj, x2_)  # [B, D, 1]
#         # compute cross attention weights: Q1 (from x1) × K2 (from x2)
#         dist1 = manifold.dist(Q1.unsqueeze(2), K2.unsqueeze(1))  # [B, D, D]
#         attn1 = 1.0 / (dist1 + eps)  # [B, D, D]
#         # Möbius 加权 value（来自 x2）
#         V2 = V2.transpose(1, 2)  # [B, 1, D]
#         weighted_V2 = manifold.mobius_scalar_mul(attn1, V2)  # [B, D, D]
#         # Möbius sum over D 维度
#         y1 = weighted_V2[:, :, 0]
#         for j in range(1, D):
#             y1 = manifold.mobius_add(y1, weighted_V2[:, :, j])  # [B, D]
#         # ========== Cross Attention 2: x2 attends to x1 ==========
#         Q2 = manifold.mobius_matvec(self.q2_proj, x2_)  # [B, D, 1]
#         K1 = manifold.mobius_matvec(self.k1_proj, x1_)  # [B, D, 1]
#         V1 = manifold.mobius_matvec(self.v1_proj, x1_)  # [B, D, 1]
#         dist2 = manifold.dist(Q2.unsqueeze(2), K1.unsqueeze(1))  # [B, D, D]
#         attn2 = 1.0 / (dist2 + eps)
#         V1 = V1.transpose(1, 2)  # [B, 1, D]
#         weighted_V1 = manifold.mobius_scalar_mul(attn2, V1)  # [B, D, D]
#         y2 = weighted_V1[:, :, 0]
#         for j in range(1, D):
#             y2 = manifold.mobius_add(y2, weighted_V1[:, :, j])  # [B, D]
#
#         return y1, y2  # still in hyperbolic space


class HyperbolicCrossAttention(nn.Module):
    def __init__(self, dim, c=1.0):
        super().__init__()
        self.dim = dim
        self.manifold = geoopt.PoincareBall(c=c)
        # Q, K, V for x1 attending to x2
        self.q1_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
        self.k2_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
        self.v2_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
        # Q, K, V for x2 attending to x1
        self.q2_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
        self.k1_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)
        self.v1_proj = ManifoldParameter(torch.randn(dim, dim), manifold=self.manifold)

    def forward(self, x1, x2):
        """
        x1: [B, D] - 双曲张量
        x2: [B, D]
        return: y1, y2 - 双曲张量，shape: [B, D]
        """
        manifold = self.manifold
        B, D = x1.shape
        eps = 1e-5

        # reshape: [B, D] → [B, D, 1]
        x1_ = x1.view(B, D, 1)
        x2_ = x2.view(B, D, 1)

        # ===== Cross Attention 1: x1 attends to x2 =====
        Q1 = manifold.mobius_matvec(self.q1_proj, x1_)
        K2 = manifold.mobius_matvec(self.k2_proj, x2_)
        V2 = manifold.mobius_matvec(self.v2_proj, x2_)

        dist1 = manifold.dist(Q1.unsqueeze(2), K2.unsqueeze(1))  # [B, D, D]
        attn1 = 1.0 / (dist1 + eps)
        attn1 = attn1 / attn1.sum(dim=-1, keepdim=True)          # softmax 类归一化

        V2 = V2.transpose(1, 2).expand(B, D, D).unsqueeze(-1)    # [B, D_out, D_in, 1]
        y1 = batched_weighted_midpoint(V2, attn1, manifold)      # [B, D,1]

        # ===== Cross Attention 2: x2 attends to x1 =====
        Q2 = manifold.mobius_matvec(self.q2_proj, x2_)
        K1 = manifold.mobius_matvec(self.k1_proj, x1_)
        V1 = manifold.mobius_matvec(self.v1_proj, x1_)

        dist2 = manifold.dist(Q2.unsqueeze(2), K1.unsqueeze(1))  # [B, D, D]
        attn2 = 1.0 / (dist2 + eps)
        attn2 = attn2 / attn2.sum(dim=-1, keepdim=True)

        V1 = V1.transpose(1, 2).expand(B, D, D).unsqueeze(-1)    # [B, D_out, D_in, 1]
        y2 = batched_weighted_midpoint(V1, attn2, manifold)      # [B, D,1]

        return y1.squeeze(-1), y2.squeeze(-1)  # still in hyperbolic space


class HyperTriplet(nn.Module):
    def __init__(self, hp):
        super(HyperTriplet, self).__init__()

        self.poincare_ball = hp.poincare_ball
        self.char_embedding_dim = hp.char_embedding_dim
        self.conv_base_size = hp.conv_base_size
        self.drug_kernel = hp.drug_kernel
        self.drug_MAX_LENGH = hp.drug_MAX_LENGH
        self.target_kernel = hp.target_kernel
        self.target_MAX_LENGH = hp.target_MAX_LENGH

        self.hyperbolic_dim = hp.hyperbolic_dim
        self.curvature = hp.curvature
        self.dropout1 = nn.Dropout(hp.FC_Dropout)
        self.dropout2 = nn.Dropout(hp.FC_Dropout)
        self.dropout3 = nn.Dropout(hp.FC_Dropout)

        self.hyper_act = MobiusTanh(geoopt.PoincareBall(self.curvature))

        self.act = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.drug_reduce1 = HyperbolicLinear(1711, 1024)
        self.drug_reduce2 = HyperbolicLinear(1024, 512)
        self.drug_reduce3 = HyperbolicLinear(512, self.hyperbolic_dim)

        self.target_reduce1 = HyperbolicLinear(1711, 1024)
        self.target_reduce2 = HyperbolicLinear(1024, 512)
        self.target_reduce3= HyperbolicLinear(512, self.hyperbolic_dim)

        # self.drug_att = HyperbolicSelfAttention(self.hyperbolic_dim,self.curvature)
        # self.target_att = HyperbolicSelfAttention(self.hyperbolic_dim,self.curvature)
        # self.hyper_cross_att = HyperbolicCrossAttention(self.hyperbolic_dim, self.curvature)

        self.hyper_mlp1 = HyperbolicLinear(self.hyperbolic_dim, 64)
        self.hyper_mlp2 = HyperbolicLinear(64, 32)
        self.hyper_mlp3 = HyperbolicLinear(32, 2)


    def forward(self, drugIDs, proIDs, SMILESs, FASTAs, vectors_SMILES, vectors_FASTA):
        drug_ChemBert_feature = torch.zeros((len(drugIDs), 768)).cuda()
        drug_PC_feature = torch.zeros((len(drugIDs), 11)).cuda()
        drug_MACCS_feature = torch.zeros((len(drugIDs), 167)).cuda()
        drug_ECFP_feature = torch.zeros((len(drugIDs), 765)).cuda()

        target_ESM_feature = torch.zeros((len(proIDs), 1280)).cuda()
        target_mer_feature = torch.zeros((len(proIDs), 420)).cuda()
        target_PC_feature = torch.zeros((len(proIDs), 11)).cuda()

        for index in range(len(drugIDs)):
            # drug_ChemBert_feature[index] = torch.tensor(drug_ChemBert_feature_dict[drugIDs[index]]).unsqueeze(0)
            drug_PC_feature[index] = torch.tensor(drug_PC_feature_dict[drugIDs[index]]).unsqueeze(0)
            drug_MACCS_feature[index] = torch.tensor(drug_MACCS_feature_dict[drugIDs[index]]).unsqueeze(0)
            drug_ECFP_feature[index] = torch.tensor(drug_ECFP_feature_dict[drugIDs[index]]).unsqueeze(0)

            # target_ESM_feature[index] = torch.tensor(target_ESM_feature_dict[proIDs[index]]).unsqueeze(0)
            target_PC_feature[index] = torch.tensor(target_PC_feature_dict[proIDs[index]]).unsqueeze(0)
            target_mer_feature[index] = torch.tensor(target_mer_feature_dict[proIDs[index]]).unsqueeze(0)

        drug_feature = torch.cat([zscore_normalize(drug_ChemBert_feature), zscore_normalize(drug_PC_feature), zscore_normalize(drug_MACCS_feature), zscore_normalize(drug_ECFP_feature)], dim=1)
        target_feature = torch.cat([zscore_normalize(target_ESM_feature), zscore_normalize(target_PC_feature), zscore_normalize(target_mer_feature)], dim=1)

        reduce_drug_feature = self.drug_reduce1(self.poincare_ball.expmap0(drug_feature))
        reduce_drug_feature = self.hyper_act(reduce_drug_feature)
        reduce_drug_feature = self.drug_reduce2(reduce_drug_feature)
        reduce_drug_feature = self.hyper_act(reduce_drug_feature)
        head_oringal = self.drug_reduce3(reduce_drug_feature)
        # head_att = self.drug_att(head)
        # head = self.poincare_ball.mobius_add(head, head_att)

        reduce_target_feature = self.target_reduce1(self.poincare_ball.expmap0(target_feature))
        reduce_target_feature = self.hyper_act(reduce_target_feature)
        reduce_target_feature = self.target_reduce2(reduce_target_feature)
        reduce_target_feature = self.hyper_act(reduce_target_feature)
        tail_oringal = self.target_reduce3(reduce_target_feature)
        # tail_att = self.target_att(tail)
        # tail = self.poincare_ball.mobius_add(tail, tail_att)

        #交叉注意力机制
        # head_att1, tail_att1 = self.hyper_cross_att(head_oringal, tail_oringal)
        #残差连接
        # head1 = self.poincare_ball.mobius_add(head_oringal, head_att1)
        # tail1 = self.poincare_ball.mobius_add(tail_oringal, tail_att1)

        final_head = self.poincare_ball.mobius_scalar_mul(torch.tensor(0.27), head_oringal)
        # final_head = self.poincare_ball.mobius_pointwise_mul(torch.tensor(0.27), head1)
        final_tail = self.poincare_ball.mobius_scalar_mul(torch.tensor(0.73), tail_oringal)
        # final_tail = self.poincare_ball.mobius_pointwise_mul(torch.tensor(0.73), tail1)
        mid_result = self.poincare_ball.mobius_add(final_head, final_tail)

        reduce_result1 =self.hyper_mlp1(mid_result)
        act_reduce_result1 = self.hyper_act(reduce_result1)
        reduce_result2 = self.hyper_mlp2(act_reduce_result1)
        act_reduce_result2 = self.hyper_act(reduce_result2)
        result = self.hyper_mlp3(act_reduce_result2)

        return result