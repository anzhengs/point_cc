# models/vn_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from extensions.Pointnet2.pointnet2.pointnet2_utils import furthest_point_sample, grouping_operation, gather_operation
from .model_utils import query_knn_point

# ==============================================================================
# Vector Neurons 核心算子（无陷阱、严格等变）
# ==============================================================================

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return torch.einsum('bind,oi->bond', x, self.weight)

class VNBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        norm_sq = (x ** 2).sum(dim=-1)
        norm = torch.sqrt(norm_sq + 1e-8)
        norm_bn = self.bn(norm)
        factor = norm_bn / norm
        return x * factor.unsqueeze(-1)

class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope
        self.v_gen = VNLinear(in_channels, in_channels)

    def forward(self, x):
        v = self.v_gen(x)

        v_norm_sq = (v ** 2).sum(dim=-1, keepdim=True)
        v_norm = torch.sqrt(v_norm_sq + 1e-8)
        v = v / v_norm

        dot = (x * v).sum(dim=-1, keepdim=True)
        
        proj = dot * v
        res = x - proj
        mask = (dot >= 0).float()
        return mask * x + (1 - mask) * self.negative_slope * res

class VNMaxPool(nn.Module):
    def __init__(self, dim=-2):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # 计算 L2 范数，保持维度
        norm = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-8
        # 在指定维度（通常是邻居维度 S，dim=-2）求最大值的索引
        idx = torch.argmax(norm, dim=self.dim, keepdim=True)
        
        # ====================== 修复：动态维度匹配 ======================
        # 使用 *idx.shape[:-1] 动态获取前面所有的维度，仅把最后一维展开为 3
        # 这样无论是 4D 还是 5D，都能完美兼容！
        idx_expanded = idx.expand(*idx.shape[:-1], 3)
        
        return torch.gather(x, self.dim, idx_expanded)

# ==============================================================================
# 等变 SA 模块（平移不变 + 维度安全 + 无陷阱）
# ==============================================================================

class VN_PointNet_SA_Module_KNN(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, group_all=False, if_bn=True, if_idx=True):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.if_idx = if_idx
        self.group_all = group_all

        layers = []
        cin = in_channel  # ✅ 现在 in_channel 真正生效
        for cout in mlp:
            layers.append(VNLinear(cin, cout))
            if if_bn:
                layers.append(VNBatchNorm(cout))
            layers.append(VNLeakyReLU(cout))
            cin = cout
        self.mlp = nn.Sequential(*layers)
        self.pool = VNMaxPool(dim=-2)

    def forward(self, xyz, points, prev_feat=None, fixed_idx=None):
        """
        xyz: (B, 3, N) 坐标
        points: (B, 1, N, 1) 或 None，仅用于兼容旧接口
        prev_feat: (B, C_prev, N, 3) 上一级的VN特征，None表示没有
        """
        B, _, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()

        if fixed_idx is None:
            fps_idx = furthest_point_sample(xyz_t, self.npoint)
        else:
            fps_idx = fixed_idx

        new_xyz = gather_operation(xyz, fps_idx)
        new_xyz_t = new_xyz.permute(0, 2, 1).contiguous()

        # KNN 与相对坐标
        idx = query_knn_point(self.nsample, xyz_t, new_xyz_t)
        grouped_xyz = grouping_operation(xyz, idx.int())
        grouped_xyz -= new_xyz.unsqueeze(-1)  # (B, 3, G*S)
        grouped_xyz = grouped_xyz.permute(0, 2, 3, 1).contiguous()  # (B, G, S, 3)

        # 核心：拼接上一级特征
        if prev_feat is not None:
            C_prev = prev_feat.shape[1]  # 128
        
            # Step 1: permute + reshape，保证 c' = c*3 + d 的正确映射
            prev_feat_flat = prev_feat.permute(0, 1, 3, 2).contiguous().reshape(B, C_prev * 3, N)
        
            # Step 2: grouping_operation 对展平特征做邻域聚合
            grouped_prev_flat = grouping_operation(prev_feat_flat, idx.int())  # (B, C_prev*3, G*S)
        
            # Step 3: 逆 reshape，恢复 VN 特征的 3D 向量结构
            grouped_prev = grouped_prev_flat.reshape(B, C_prev, 3, self.npoint * self.nsample)
            grouped_prev = grouped_prev.permute(0, 1, 3, 2).contiguous()  # (B, C_prev, G*S, 3)
            grouped_prev = grouped_prev.reshape(B, C_prev, self.npoint, self.nsample, 3)
        
            # Step 4: 拼接相对坐标(1通道) + 上一级特征(C_prev通道)
            v_feat = torch.cat([grouped_xyz.unsqueeze(1), grouped_prev], dim=1)  # (B, 1+C_prev, G, S, 3)
        else:
            v_feat = grouped_xyz.unsqueeze(1)  # (B, 1, G, S, 3)

        B, C, G, S, D = v_feat.shape
        v_feat = v_feat.reshape(B, C, G * S, 3)
        v_feat = self.mlp(v_feat)
        v_feat = v_feat.view(B, -1, G, S, 3)

        new_feat = self.pool(v_feat).squeeze(-2)

        if self.if_idx:
            return new_xyz, new_feat, fps_idx
        return new_xyz, new_feat

# ==============================================================================
# 等变 Transformer（数值稳定 + 无陷阱）
# ==============================================================================

class VN_Transformer(nn.Module):
    def __init__(self, in_channel, dim=64, num_heads=4):
        super().__init__()
        self.h = num_heads
        self.d = dim
        self.q = VNLinear(in_channel, num_heads * dim)
        self.k = VNLinear(in_channel, num_heads * dim)
        self.v = VNLinear(in_channel, num_heads * dim)
        self.out = VNLinear(num_heads * dim, in_channel)

    def forward(self, feat, xyz):
        B, C, N, _ = feat.shape
        q = self.q(feat).view(B, self.h, self.d, N, 3)
        k = self.k(feat).view(B, self.h, self.d, N, 3)
        v = self.v(feat).view(B, self.h, self.d, N, 3)

        attn = torch.einsum('bhdfn,bhdfm->bhnm', q, k)
        # ====================== 专业缩放（保留） ======================
        attn = F.softmax(attn / ((self.d * 3) ** 0.5), dim=-1)
        
        out = torch.einsum('bhnm,bhdfm->bhdfn', attn, v)
        out = out.reshape(B, self.h * self.d, N, 3)
        return self.out(out) + feat
