# models/SymmCompletion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from extensions.chamfer_dist import ChamferDistanceL1
from .model_utils import MLP_CONV, Transformer, PointNet_SA_Module_KNN, query_knn_point
from extensions.Pointnet2.pointnet2.pointnet2_utils import grouping_operation
from .build import MODELS
from .vn_utils import VN_PointNet_SA_Module_KNN, VNLinear, VNMaxPool

# ==============================================================================
# LEFC：等变特征协方差规范化前端
# 核心改动：6D 连续旋转表示 (Gram-Schmidt) 替代 SVD
# 优势：
#   1. 梯度处处连续，无 SVD 奇异矩阵梯度爆炸风险
#   2. 叉乘天然保证 det(R)=+1，无需行列式惩罚
#   3. 硬正交化免去正交/范数辅助 Loss，消除软约束与硬投影的优化冲突
# ==============================================================================

class VNFrameEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一级：从原始坐标提取逐点等变特征
        self.vn_sa1 = VN_PointNet_SA_Module_KNN(
            npoint=256, nsample=32, in_channel=1,
            mlp=[64, 128], group_all=False, if_bn=True, if_idx=False
        )
        # 第二级：在第一级特征基础上进一步抽象
        # in_channel = 1(相对坐标) + 128(feat1通道) = 129
        self.vn_sa2 = VN_PointNet_SA_Module_KNN(
            npoint=128, nsample=16, in_channel=129,
            mlp=[128, 256], group_all=False, if_bn=True, if_idx=False
        )
        # 全局等变池化
        self.vn_global_pool = VNMaxPool(dim=-2)
        # 预测 2 个等变向量 → Gram-Schmidt 正交化 → 叉乘得第三轴
        self.vn_predict = VNLinear(256, 2)

    def forward(self, x):
        """
        x: (B, 3, N) 输入残缺点云
        返回:
            R_align (B, 3, 3) 严格 SO(3) 旋转矩阵
            - 列向量 [u1, u2, u3] 构成规范坐标系基
            - p_canonical = p_world @ R_align 将世界坐标变换到规范坐标
        """
        b, _, n = x.shape
        x_points = x.transpose(1, 2).unsqueeze(1).contiguous()  # (B, N, 1, 3) -> (B, 1, N, 3) for compat

        # ========== 第一级编码 ==========
        if self.training:
            xyz1, feat1 = checkpoint(self.vn_sa1, x, x_points, use_reentrant=False)
        else:
            xyz1, feat1 = self.vn_sa1(x, x_points)

        # ========== 第二级编码 ==========
        xyz1_t = xyz1.transpose(1, 2).unsqueeze(1).contiguous()
        if self.training:
            xyz2, v_feat = checkpoint(self._sa2_wrap, xyz1, xyz1_t, feat1, use_reentrant=False)
        else:
            xyz2, v_feat = self.vn_sa2(xyz1, xyz1_t, prev_feat=feat1)


        # ========== 保存 LGP 所需的逐点特征（全局池化之前） ==========
        vn_per_feat = v_feat      # (B, 256, 128, 3) 第二级逐点特征
        vn_keypoints = xyz2       # (B, 3, 128) 第二级关键点
        vn_per_feat_1 = feat1     # (B, 128, 256, 3) 第一级逐点特征
        vn_keypoints_1 = xyz1     # (B, 3, 256) 第一级关键点

        # ========== 全局池化 + 预测 2 个等变向量 ==========
        v_feat = self.vn_global_pool(v_feat).squeeze(-2)  # (B, 256, 3)
        v_feat = v_feat.unsqueeze(2)                        # (B, 256, 1, 3)
        M = self.vn_predict(v_feat).squeeze(2)              # (B, 2, 3)

        v1 = M[:, 0, :]  # (B, 3)
        v2 = M[:, 1, :]  # (B, 3)

        # ========== 6D 连续旋转：Gram-Schmidt 正交化 ==========
        u1 = F.normalize(v1, p=2, dim=-1, eps=1e-6)
        u2 = v2 - (u1 * v2).sum(-1, keepdim=True) * u1
        u2 = F.normalize(u2, p=2, dim=-1, eps=1e-6)
        u3 = torch.cross(u1, u2, dim=-1)

        R_align = torch.stack([u1, u2, u3], dim=-1)

        # ========== 返回 5 个值 ==========
        return R_align, vn_per_feat, vn_keypoints, vn_per_feat_1, vn_keypoints_1

    def _sa2_wrap(self, xyz1, xyz1_t, feat1):
        return self.vn_sa2(xyz1, xyz1_t, prev_feat=feat1)

# ==============================================================================
# LGPExtractor：轻量级几何先验提取器
#
# 关键修正：
#   原版对规范空间特征取 L2 范数 → 等价于对原始特征取范数（旋转不变性陷阱）
#   修正后直接展平 (C, 3) → 3C 维规范坐标分量
#
# 数学保证：
#   VN 特征 v_i 经 R_align 旋转到规范空间后，各分量 (v_x, v_y, v_z) 严格旋转不变
#   且完整保留了相对于全局对称轴的方向信息（法向、曲率切向、边缘朝向等）
#   3C 维 >> C 维，信息量提升 3 倍
# ==============================================================================

class LGPExtractor(nn.Module):
    def __init__(self, vn_feat_dim=256, out_dim=128, k_interp=3):
        super().__init__()
        self.k_interp = k_interp
        # 输入 3*vn_feat_dim（规范坐标分量展平），输出 out_dim
        self.proj = MLP_CONV(
            in_channel=3 * vn_feat_dim,
            layer_dims=[3 * vn_feat_dim // 2, out_dim]
        )

    def forward(self, vn_feat, vn_xyz, target_xyz, R_align):
        """
        Args:
            vn_feat: (B, C, M, 3) VN 逐点等变特征（原始空间）
            vn_xyz: (B, 3, M) VN 关键点坐标（原始空间）
            target_xyz: (B, 3, N) 目标关键点坐标（规范空间，来自 LSTNet）
            R_align: (B, 3, 3) 旋转矩阵
        Returns:
            feat_lgp: (B, out_dim, N) 几何先验特征
        """
        B, C, M, _ = vn_feat.shape
        N = target_xyz.shape[-1]

        # Step 1: 将 VN 特征的 3D 向量分量旋转至规范空间
        # (B, C, M, 3) @ (B, 3, 3) → (B, C, M, 3)
        feat_canon = torch.einsum('bcmi,bij->bcmj', vn_feat, R_align)

        # Step 2: 展平 (C, 3) → 3C 维规范坐标分量
        # 这 3C 个标量每个都是旋转不变的，且携带方向信息
        # 相比原版取 L2 范数（C 维、无方向信息），信息量提升 3 倍
        feat_ri = feat_canon.reshape(B, C * 3, M)  # (B, 3C, M)

        # Step 3: 将 VN 关键点旋转至规范空间（与 target_xyz 同坐标系）
        vn_xyz_canon = torch.bmm(
            vn_xyz.permute(0, 2, 1), R_align
        ).permute(0, 2, 1)  # (B, 3, M)

        # Step 4: KNN 逆距离加权插值 (M VN点 → N 目标点)
        vn_xyz_t = vn_xyz_canon.permute(0, 2, 1).contiguous()
        target_xyz_t = target_xyz.permute(0, 2, 1).contiguous()
        idx = query_knn_point(self.k_interp, vn_xyz_t, target_xyz_t)

        grouped_feat = grouping_operation(feat_ri.contiguous(), idx.int())
        grouped_feat = grouped_feat.reshape(B, C * 3, N, self.k_interp)

        # 计算逆距离权重
        grouped_xyz = grouping_operation(vn_xyz_canon.contiguous(), idx.int())
        grouped_xyz = grouped_xyz.reshape(B, 3, N, self.k_interp)
        target_xyz_exp = target_xyz.unsqueeze(-1)
        dist = torch.norm(grouped_xyz - target_xyz_exp, p=2, dim=1)
        dist = dist + 1e-8
        weight = 1.0 / dist
        weight = weight / weight.sum(dim=-1, keepdim=True)
        weight = weight.unsqueeze(1)  # (B, 1, N, K)

        feat_interp = (grouped_feat * weight).sum(dim=-1)  # (B, 3C, N)

        # Step 5: MLP 降维投影
        feat_lgp = self.proj(feat_interp)  # (B, out_dim, N)

        return feat_lgp


# ==============================================================================
# 原汁原味的 Baseline 类定义 (保持原版不变)
# ==============================================================================

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        _, NK, _ = y.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(y).reshape(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossFormer(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.1):
        super().__init__()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Identity()
        self.bn3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x, y):
        short_cut = x
        x = self.bn1(x)
        y = self.bn2(y)
        x = self.attn(query=x, key=y, value=y)[0]
        x = short_cut + self.drop_path(x)
        x = x + self.drop_path(self.ffn(self.bn3(x)))
        return x


# ==============================================================================
# LSTNet：核心重构 —— 显式对称面预测替代逐点 R+T
#
# 原版：每个关键点独立预测 3×3 旋转 + 3 平移 = 12 自由度/点 → 自由度过高
# 新版：全局预测对称超平面 n(3) + d(1) = 4 自由度 → 物理约束强、优化简单
#
# 镜像反射公式：p' = p - 2(p·n + d)n
# 在规范空间中，物体对称性本质就是沿特定平面的镜像反射
# ==============================================================================

class LSTNet(nn.Module):
    def __init__(self, out_dim=512):
        super(LSTNet, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.expanding = MLP_CONV(in_channel=128, layer_dims=[256, out_dim])

        # 全局对称面预测：法向量 n(3) + 偏置 d(1)
        self.plane_mlp = nn.Sequential(
            nn.Linear(out_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 4)  # n(3) + d(1)
        )

    def forward(self, point_cloud):
        """
        point_cloud: (B, N, 3) 规范空间中的点云
        返回:
            coarse: (B, 3, 1024) 拼接对称点+关键点
            symmetry_points: (B, 3, 512) 镜像反射生成的对称点
            keyfeatures: (B, 128, 512) 关键点特征
        """
        b = point_cloud.shape[0]
        l0_xyz = point_cloud
        l0_points = point_cloud

        keypoints, keyfeatures, _ = self.sa_module_1(l0_xyz, l0_points)
        keyfeatures = self.transformer_1(keyfeatures, keypoints)

        feat = self.expanding(keyfeatures)          # (B, out_dim, 512)
        feat = feat.transpose(2, 1).contiguous()    # (B, 512, out_dim)

        # 全局特征 → 对称面参数
        gf_feat = feat.max(dim=1, keepdim=True)[0]  # (B, 1, out_dim)
        plane_params = self.plane_mlp(gf_feat)       # (B, 1, 4)

        # 提取法向量并单位化 (eps 防止训练初期零向量)
        n = F.normalize(plane_params[:, :, :3], dim=-1, eps=1e-6)  # (B, 1, 3)
        d = plane_params[:, :, 3:]                                   # (B, 1, 1)

        # ========== 显式镜像反射 ==========
        keypoints_t = keypoints.transpose(2, 1).contiguous()  # (B, 512, 3)
        # 点到对称面的有向距离
        dist_to_plane = torch.sum(keypoints_t * n, dim=-1, keepdim=True) + d  # (B, 512, 1)
        # 沿法向量方向镜像
        symmetry_points_t = keypoints_t - 2 * dist_to_plane * n                # (B, 512, 3)
        symmetry_points = symmetry_points_t.transpose(2, 1).contiguous()       # (B, 3, 512)

        coarse = torch.cat([symmetry_points, keypoints], dim=-1)  # (B, 3, 1024)
        return coarse, symmetry_points, keyfeatures, keypoints


class Fusion(nn.Module):
    def __init__(self, in_channel=512):
        super(Fusion, self).__init__()
        self.corssformer_1 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)
        self.corssformer_2 = CrossFormer(in_channel, in_channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)

    def forward(self, feat_x, feat_y):
        feat = self.corssformer_1(feat_x, feat_y)
        feat = self.corssformer_2(feat, feat)
        return feat


class SGFormer(nn.Module):
    def __init__(self, gf_dim=512, up_factor=2):
        super(SGFormer, self).__init__()
        self.up_factor = up_factor
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_gf = MLP_CONV(in_channel=gf_dim, layer_dims=[256, 128])
        self.mlp_2 = MLP_CONV(in_channel=256, layer_dims=[256, 128])
        self.transformer = Transformer(in_channel=128, dim=64)

        self.expand_dim_1 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_2 = MLP_CONV(in_channel=128, layer_dims=[128, 256])
        self.expand_dim_3 = MLP_CONV(in_channel=128, layer_dims=[128, 256])

        self.fusion_1 = Fusion(in_channel=256)
        self.fusion_2 = Fusion(in_channel=256)

        self.mlp_fusion = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512)
        )
        self.fusion_3 = Fusion(in_channel=512)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 3 * self.up_factor)
        )

    def forward(self, coarse, symmetry_feat, partial_feat):
        b, _, n = coarse.shape
        feat = self.mlp_1(coarse)
        feat_max = feat.max(dim=-1, keepdim=True)[0]
        feat = torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, coarse)

        feat = self.expand_dim_1(feat)
        partial_feat = self.expand_dim_2(partial_feat)
        symmetry_feat = self.expand_dim_3(symmetry_feat)

        feat = feat.transpose(2, 1).contiguous()
        partial_feat = partial_feat.transpose(2, 1).contiguous()
        symmetry_feat = symmetry_feat.transpose(2, 1).contiguous()

        feat_p = self.fusion_1(feat, partial_feat)
        feat_s = self.fusion_2(feat, symmetry_feat)
        feat = torch.cat([feat_p, feat_s], dim=-1)
        feat = self.mlp_fusion(feat)

        feat = self.fusion_3(feat, feat)
        offset = self.fc(feat).view(b, -1, 3)
        pcd_up = coarse.transpose(2, 1).contiguous().unsqueeze(dim=2).repeat(1, 1, self.up_factor, 1).view(b, -1, 3) + offset
        return pcd_up


class local_encoder(nn.Module):
    def __init__(self, out_channel=128):
        super(local_encoder, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2, layer_dims=[256, out_channel])
        self.transformer = Transformer(out_channel, dim=64)

    def forward(self, input):
        feat = self.mlp_1(input)
        feat = torch.cat([feat, torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))], 1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat, input)
        return feat


# ==============================================================================
# SymmCompletion：注册模块与前向闭环逻辑
# ==============================================================================

# models/SymmCompletion.py — 修改后的 SymmCompletion 类

@MODELS.register_module()
class SymmCompletion(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.up_factors = [int(i) for i in config.up_factors.split(',')]

        self.vn_frame_estimator = VNFrameEstimator()
        self.lgp_extractor = LGPExtractor(
            vn_feat_dim=256, out_dim=128, k_interp=3
        )

        # LGP 预热参数
        self.lgp_start_epoch = getattr(config, 'lgp_start_epoch', 50)
        self.lgp_warmup_epochs = getattr(config, 'lgp_warmup_epochs', 50)

        self.lstnet = LSTNet(out_dim=512)
        self.local_encoder = local_encoder(out_channel=128)
        self.sgformer_1 = SGFormer(gf_dim=512, up_factor=self.up_factors[0])
        self.sgformer_2 = SGFormer(gf_dim=512, up_factor=self.up_factors[1])
        self.include_input = config.include_input
        self.loss_func = ChamferDistanceL1()
        self._current_epoch = 0

    def set_epoch(self, epoch):
        """由 runner.py 在每个 epoch 开始时调用"""
        self._current_epoch = epoch

    def _lgp_weight(self):
        """LGP 门控权重：从 0 线性增长到 1"""
        if self._current_epoch < self.lgp_start_epoch:
            return 0.0
        elif self._current_epoch >= self.lgp_start_epoch + self.lgp_warmup_epochs:
            return 1.0
        else:
            return (self._current_epoch - self.lgp_start_epoch) / self.lgp_warmup_epochs

    def get_loss(self, rets, gt):
        loss_list = []
        loss_total = 0.0
        for pcd in rets:
            loss = self.loss_func(pcd, gt)
            loss_list.append(loss)
            loss_total += loss
        dummy_penalty = torch.tensor(0.0, device=gt.device)
        return loss_total, loss_list[0], loss_list[-1], dummy_penalty, dummy_penalty

    def forward(self, point_cloud):
        pc_t = point_cloud.transpose(1, 2).contiguous()
        R_align, vn_per_feat, vn_keypoints, vn_per_feat_1, vn_keypoints_1 = \
            self.vn_frame_estimator(pc_t)

        pc_canonical = torch.bmm(point_cloud, R_align)

        coarse, symmetry_points, keyfeatures, keypoints = \
            self.lstnet(pc_canonical.transpose(2, 1).contiguous())

        # ========== LGP 门控注入 ==========
        feat_lgp = self.lgp_extractor(
            vn_per_feat, vn_keypoints, keypoints, R_align
        )
        lgp_w = self._lgp_weight()
        if lgp_w > 0:
            feat_partial = keyfeatures + lgp_w * feat_lgp
        else:
            feat_partial = keyfeatures
        # =====================================

        feat_symmetry = self.local_encoder(symmetry_points)

        fine1 = self.sgformer_1(coarse, feat_symmetry, feat_partial)
        fine2 = self.sgformer_2(
            fine1.transpose(2, 1).contiguous(), feat_symmetry, feat_partial
        )

        if self.include_input:
            fine2 = torch.cat([fine2, pc_canonical], dim=1).contiguous()

        coarse_points = coarse.transpose(2, 1).contiguous()

        R_inv = R_align.transpose(1, 2).contiguous()
        coarse_orig = torch.bmm(coarse_points, R_inv)
        fine1_orig = torch.bmm(fine1, R_inv)
        fine2_orig = torch.bmm(fine2, R_inv)

        rets = [coarse_orig.contiguous(), fine1_orig.contiguous(), fine2_orig.contiguous()]
        return rets