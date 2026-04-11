# models/SymmCompletion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from extensions.chamfer_dist import ChamferDistanceL1
from .model_utils import MLP_CONV, Transformer, PointNet_SA_Module_KNN
from .build import MODELS

# ======================================
# 导入 LEFC 前端所需的 VN 算子
# ======================================
from .vn_utils import VN_PointNet_SA_Module_KNN, VNLinear, VNMaxPool

# ==============================================================================
# LEFC：等变特征协方差规范化前端(完美解决轴突变 Axis Swapping)
# ==============================================================================

class VNFrameEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        # 两层SA + 更宽的通道
        self.vn_sa1 = VN_PointNet_SA_Module_KNN(
            npoint=512, nsample=32, in_channel=1,
            mlp=[64, 128], group_all=False, if_bn=True, if_idx=False
        )
        self.vn_sa2 = VN_PointNet_SA_Module_KNN(
            npoint=256, nsample=32, in_channel=128,
            mlp=[128, 256], group_all=False, if_bn=True, if_idx=False
        )
        self.vn_global_pool = VNMaxPool(dim=-2)
        self.vn_predict = VNLinear(256, 2)

    def forward(self, x):
        b, _, n = x.shape
        x_points = x.transpose(1, 2).unsqueeze(1).contiguous()

        # 两级编码
        xyz1, feat1 = self.vn_sa1(x, x_points)
        xyz1_t = xyz1.transpose(1, 2).unsqueeze(1).contiguous()
        _, v_feat = self.vn_sa2(xyz1, xyz1_t)  # 需要修改VN SA以接受特征

        v_feat = self.vn_global_pool(v_feat).squeeze(-2)  # (B, 256, 3)
        v_feat = v_feat.unsqueeze(2)  # (B, 256, 1, 3)
        basis = self.vn_predict(v_feat).squeeze(2)  # (B, 2, 3)

        v1 = basis[:, 0, :]
        v2 = basis[:, 1, :]

        # Gram-Schmidt + det修正
        u1 = F.normalize(v1, p=2, dim=-1, eps=1e-8)
        u2_raw = v2 - (v2 * u1).sum(dim=-1, keepdim=True) * u1
        u2_norm = torch.norm(u2_raw, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        u2 = u2_raw / u2_norm
        u3 = torch.cross(u1, u2, dim=-1)

        R_align = torch.stack([u1, u2, u3], dim=-1)  # (B, 3, 3)
        det = torch.det(R_align)
        sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)
        r_col1 = R_align[:, :, 0:1]  # 保留前两列
        r_col2 = R_align[:, :, 1:2]
        r_col3 = R_align[:, :, 2:3] * sign  # 第三列乘符号（非原地）
        R_align = torch.cat([r_col1, r_col2, r_col3], dim=-1)  # 重新拼接
        # 只翻转第三列

        return R_align, v1, v2

# ==============================================================================
# 原汁原味的 Baseline 类定义 (100% 保持原版不变，保证最高生成精度)
# ==============================================================================
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
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

class LSTNet(nn.Module):
    def __init__(self, out_dim=512):
        super(LSTNet, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.expanding = MLP_CONV(in_channel=128, layer_dims=[256, out_dim])     
        
        self.mlp = nn.Sequential(
                nn.Linear(512*2, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 9+3)
        )

    def forward(self, point_cloud):
        b = point_cloud.shape[0]
        l0_xyz = point_cloud
        l0_points = point_cloud

        keypoints, keyfeatures, _ = self.sa_module_1(l0_xyz, l0_points)  
        keyfeatures = self.transformer_1(keyfeatures, keypoints) 

        feat = self.expanding(keyfeatures)
        feat = feat.transpose(2, 1).contiguous()
        gf_feat = feat.max(dim=1, keepdim=True)[0]
        feat = torch.cat([feat, gf_feat.repeat(1, feat.size(1), 1)], dim=-1) 

        ret = self.mlp(feat)   
        R = ret[:, :, :9].view(b, 512, 3, 3)
        T = ret[:, :, 9:]
        symmetry_points = torch.matmul(keypoints.transpose(2, 1).contiguous().unsqueeze(2), R).view(b, 512, 3)
        symmetry_points = symmetry_points + T
        symmetry_points = symmetry_points.transpose(2, 1).contiguous()
        coarse = torch.cat([symmetry_points, keypoints], dim=-1) 
        return coarse, symmetry_points, keyfeatures

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
        feat= torch.cat([feat, feat_max.repeat(1, 1, feat.shape[-1])], dim=1)
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
    def __init__(self,out_channel=128):
        super(local_encoder, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2, layer_dims=[256, out_channel])
        self.transformer = Transformer(out_channel, dim=64)

    def forward(self,input):
        feat = self.mlp_1(input)
        feat = torch.cat([feat,torch.max(feat, 2, keepdim=True)[0].repeat((1, 1, feat.size(2)))], 1)
        feat = self.mlp_2(feat)
        feat = self.transformer(feat,input)

        return feat

# ==============================================================================
# LEFC: 注册模块与前向闭环逻辑
# ==============================================================================
@MODELS.register_module()
class SymmCompletion(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.up_factors = [int(i) for i in config.up_factors.split(',')]
        
        self.vn_frame_estimator = VNFrameEstimator()
        
        self.lstnet = LSTNet(out_dim=512)
        self.local_encoder = local_encoder(out_channel=128)
        self.sgformer_1 = SGFormer(gf_dim=512, up_factor=self.up_factors[0])
        self.sgformer_2 = SGFormer(gf_dim=512, up_factor=self.up_factors[1])
        self.include_input = config.include_input
        self.loss_func = ChamferDistanceL1()
        
        # 正交损失权重
        self.ortho_weight = getattr(config, 'ortho_weight', 0.1)
        # 范数正则化权重
        self.norm_weight = getattr(config, 'norm_weight', 0.01)

    def get_loss(self, rets, gt):
        """
        rets: list of predicted point clouds
        gt: ground truth point cloud
        v1, v2: stored in self during forward
        """
        loss_list = []
        loss_total = 0
        for pcd in rets:
            loss = self.loss_func(pcd, gt)
            loss_list.append(loss)
            loss_total += loss

        # 正交惩罚：鼓励 v1 ⊥ v2（作为软约束辅助Gram-Schmidt）
        v1 = self.v1  # (B, 3)
        v2 = self.v2  # (B, 3)
        v1_norm = F.normalize(v1, p=2, dim=-1)
        v2_norm = F.normalize(v2, p=2, dim=-1)
        loss_ortho = torch.mean((torch.sum(v1_norm * v2_norm, dim=-1)) ** 2)
        
        # 范数正则化：防止 v1, v2 退化为零向量
        # 指数形式，自然有界
        loss_norm = torch.mean(
            torch.exp(-torch.norm(v1, p=2, dim=-1)) + 
            torch.exp(-torch.norm(v2, p=2, dim=-1))
        )


        #  修正：将正交损失和范数正则加入总损失
        total_loss = loss_total + self.ortho_weight * loss_ortho + self.norm_weight * loss_norm

        return total_loss, loss_list[0], loss_list[-1], loss_ortho, loss_norm

    def forward(self, point_cloud):
        """
        point_cloud: (B, N, 3)
        """
        # Phase 1: 规范化对齐
        pc_t = point_cloud.transpose(1, 2).contiguous()  # (B, 3, N)
        R_align, v1, v2 = self.vn_frame_estimator(pc_t)  # (B, 3, 3)

        # 将输入点云旋转到规范空间
        pc_canonical = torch.bmm(point_cloud, R_align)  # (B, N, 3) × (B, 3, 3)

        # Phase 2: 在规范空间中处理
        coarse, symmetry_points, keyfeatures = self.lstnet(pc_canonical.transpose(2, 1).contiguous())
        feat_symmetry = self.local_encoder(symmetry_points)
        feat_partial = keyfeatures
        fine1 = self.sgformer_1(coarse, feat_symmetry, feat_partial)
        fine2 = self.sgformer_2(fine1.transpose(2, 1).contiguous(), feat_symmetry, feat_partial)

        if self.include_input:
            fine2 = torch.cat([fine2, pc_canonical], dim=1).contiguous()

        coarse_points = coarse.transpose(2, 1).contiguous()

        # Phase 3: 逆旋转回原始空间
        R_inv = R_align.transpose(1, 2).contiguous()  # 正交矩阵的逆 = 转置
        coarse_orig = torch.bmm(coarse_points, R_inv)
        fine1_orig = torch.bmm(fine1, R_inv)
        fine2_orig = torch.bmm(fine2, R_inv)

        rets = [coarse_orig.contiguous(), fine1_orig.contiguous(), fine2_orig.contiguous()]

        # ✅ 保存 v1, v2 到 self，供 get_loss 使用
        self.v1 = v1
        self.v2 = v2

        return rets