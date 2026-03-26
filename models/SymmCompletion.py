# models/SymmCompletion.py
import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL1
from .model_utils import MLP_CONV, Transformer, PointNet_SA_Module_KNN
from .build import MODELS

# ======================================
# 导入我们创建的VN算子
# ======================================
from .vn_utils import (
    VNLinear, VNBatchNorm, VNLeakyReLU, VNMaxPool,
    VN_PointNet_SA_Module_KNN, VN_Transformer
)

# ======================================
# 原有Attention、CrossFormer、Fusion、SGFormer、local_encoder类完全保留
# ======================================
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

# ======================================
# 核心替换：VN_LSTNet（你的严格等变性方案）
# ======================================
class VN_LSTNet(nn.Module):
    def __init__(self, out_dim=512, K=8):
        super(VN_LSTNet, self).__init__()
        self.K = K  # 构造A矩阵的基向量维度，可调整
        # 1. VN-SA下采样模块（与原代码参数完全对齐）
        self.sa_module_1 = VN_PointNet_SA_Module_KNN(
            npoint=512, nsample=16, in_channel=1, 
            mlp=[64, 128], group_all=False, if_bn=False, if_idx=True
        )
        # 2. VN-Transformer特征提取
        self.transformer_1 = VN_Transformer(in_channel=128, dim=64)
        # 3. VN特征扩张层
        self.expanding = nn.Sequential(
            VNLinear(128, 256),
            VNBatchNorm(256),
            VNLeakyReLU(256),
            VNLinear(256, out_dim)
        )
        # 4. 局部+全局特征融合VN-MLP
        fuse_dim = out_dim * 2
        self.vn_fuse = nn.Sequential(
            VNLinear(fuse_dim, 256),
            VNBatchNorm(256),
            VNLeakyReLU(256),
            VNLinear(256, 128),
            VNBatchNorm(128),
            VNLeakyReLU(128)
        )
        # 5. 预测A矩阵的两个基向量组V1、V2（核心等变性构造）
        self.vn_predict_V1 = VNLinear(128, self.K)
        self.vn_predict_V2 = VNLinear(128, self.K)
        # 6. 预测平移向量T
        self.vn_predict_T = VNLinear(128, 1)

    def forward(self, point_cloud):
        """
        输入输出与原LSTNet完全一致，零侵入修改后续流程
        输入：point_cloud (B, 3, 2048) → 原代码输入格式
        输出：
            coarse: (B, 3, 1024) → 与原代码完全一致
            symmetry_points: (B, 3, 512) → 与原代码完全一致
            keyfeatures_scalar: (B, 128, 512) → 旋转不变标量特征，直接喂给后续SGFormer
        """
        b = point_cloud.shape[0]
        l0_xyz = point_cloud  # (B, 3, 2048)
        # 初始VN特征：将坐标扩展为向量特征 (B, 1, 2048, 3)
        l0_points = l0_xyz.permute(0, 2, 1).unsqueeze(1)  # (B, 1, N, 3)

        # Step1: VN-SA提取关键点和向量特征
        keypoints, keyfeatures_vec, _ = self.sa_module_1(l0_xyz, l0_points)
        # keypoints: (B, 3, 512), keyfeatures_vec: (B, 128, 512, 3)

        # Step2: VN-Transformer提取等变特征
        keyfeatures_vec = self.transformer_1(keyfeatures_vec, keypoints)

        # Step3: 特征扩张+全局池化
        feat_vec = self.expanding(keyfeatures_vec)  # (B, 512, 512, 3)
        gf_feat_vec = torch.max(feat_vec, dim=2, keepdim=True)[0]  # (B, 512, 1, 3)
        # 拼接局部+全局特征
        feat_vec = torch.cat([feat_vec, gf_feat_vec.expand(-1, -1, 512, -1)], dim=1)  # (B, 1024, 512, 3)

        # Step4: 特征融合
        feat_fused = self.vn_fuse(feat_vec)  # (B, 128, 512, 3)

        # Step5: 构造严格等变的仿射矩阵A（核心方案：A = V1^T @ V2）
        V1 = self.vn_predict_V1(feat_fused)  # (B, K, 512, 3)
        V2 = self.vn_predict_V2(feat_fused)  # (B, K, 512, 3)
        # 维度转换计算内积矩阵：(B, 512, 3, K) @ (B, 512, K, 3) = (B, 512, 3, 3)
        A = torch.matmul(V1.permute(0, 2, 3, 1), V2.permute(0, 2, 1, 3))

        # Step6: 预测严格等变的平移向量T
        T = self.vn_predict_T(feat_fused).permute(0, 2, 3, 1).squeeze(-1)  # (B, 512, 3)

        # Step7: 生成对称点云（与原论文逻辑完全一致）
        pk = keypoints.permute(0, 2, 1).unsqueeze(2)  # (B, 512, 1, 3)
        symmetry_points = torch.matmul(pk, A).squeeze(2) + T  # (B, 512, 3)
        symmetry_points = symmetry_points.permute(0, 2, 1).contiguous()  # (B, 3, 512) 对齐原格式

        # Step8: 生成coarse粗点云，与原代码输出完全一致
        coarse = torch.cat([symmetry_points, keypoints], dim=-1)  # (B, 3, 1024)

        # Step9: 提取旋转不变标量特征（给后续SGFormer使用，无需修改后端）
        keyfeatures_scalar = torch.norm(keyfeatures_vec, p=2, dim=-1)  # (B, 128, 512)

        return coarse, symmetry_points, keyfeatures_scalar

# ======================================
# 主模型注册类，仅替换LSTNet为VN_LSTNet
# ======================================
@MODELS.register_module()
class SymmCompletion(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.up_factors = [int(i) for i in config.up_factors.split(',')]
        # 替换为VN版本的LSTNet
        #self.lstnet = VN_LSTNet(out_dim=512, K=8)
        self.lstnet = LSTNet(out_dim=512, K=8)
        self.local_encoder = local_encoder(out_channel=128)
        self.sgformer_1 = SGFormer(gf_dim=512, up_factor=self.up_factors[0])
        self.sgformer_2 = SGFormer(gf_dim=512, up_factor=self.up_factors[1])
        self.include_input = config.include_input
        self.loss_func = ChamferDistanceL1()
        
    def get_loss(self, rets, gt):
        loss_list = []
        loss_total = 0
        for pcd in rets:
            loss = self.loss_func(pcd, gt)
            loss_list.append(loss)
            loss_total += loss
        return loss_total, loss_list[0], loss_list[-1], loss_list[0], loss_list[-1]

    def forward(self, point_cloud):
        """
        前向流程与原代码100%一致，无任何侵入修改
        """
        # point_cloud: (B, N, 3) → 原代码输入格式
        coarse, symmetry_points, keyfeatures = self.lstnet(point_cloud.transpose(2,1).contiguous())
        feat_symmetry = self.local_encoder(symmetry_points)
        feat_partial = keyfeatures
        fine1 = self.sgformer_1(coarse, feat_symmetry, feat_partial)
        fine2 = self.sgformer_2(fine1.transpose(2,1).contiguous(), feat_symmetry, feat_partial)

        if self.include_input: 
            fine2 = torch.cat([fine2, point_cloud],dim=1).contiguous()

        rets = [coarse.transpose(2,1).contiguous(), fine1, fine2]
        self.pred_dense_point = rets[-1]

        return rets