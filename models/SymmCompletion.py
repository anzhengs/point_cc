# models/SymmCompletion.py
import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL1
from .model_utils import MLP_CONV, Transformer, PointNet_SA_Module_KNN
from .build import MODELS

# ======================================
# 导入 LEFC 前端所需的 VN 算子
# ======================================
from .vn_utils import VN_PointNet_SA_Module_KNN

# ==============================================================================
# LEFC: 等变特征协方差规范化前端 (严格排除数学陷阱版)
# ==============================================================================
class VNFrameEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        # 轻量级 VN 编码器，用于提取等变特征
        self.vn_encoder = VN_PointNet_SA_Module_KNN(
            npoint=128, nsample=16, in_channel=1, 
            mlp=[64, 128], group_all=True, if_bn=True, if_idx=False
        )

    def forward(self, x):
        """
        x: (B, 3, N) 输入的残缺点云
        返回: R_align (B, 3, 3) 严格的 SO(3) 规范化旋转矩阵
        """
        b, _, n = x.shape
        x_points = x.transpose(1, 2).unsqueeze(1).contiguous() # (B, 1, N, 3)
        
        # 1. 提取等变特征 (B, 128, 128, 3)
        _, v_feat = self.vn_encoder(x, x_points)
        
        if v_feat.dim() == 4:
            v_feat = v_feat.mean(dim=2) # 降维后变成 (B, 128, 3)
        
        # 2. 计算协方差矩阵 M = V^T * V -> (B, 3, 3)
        v_feat = torch.nn.functional.normalize(v_feat, p=2, dim=-1) # 将等变向量长度归一化
        M = torch.bmm(v_feat.transpose(1, 2), v_feat)
        
        # 3. 极其关键：加入微小扰动防止特征值重合导致的 NaN 梯度
        noise = torch.diag(torch.tensor([1e-5, 2e-5, 3e-5], device=M.device)).unsqueeze(0)
        M = M + noise
        
        # 4. 特征值分解
        eigenvalues, eigenvectors = torch.linalg.eigh(M)
        
        # 提取主轴 (B, 3)
        v1 = eigenvectors[:, :, 0] 
        v2 = eigenvectors[:, :, 1]
        
        # 5. 解决特征向量的符号多义性
        sign_v1 = torch.sign((v_feat * v1.unsqueeze(1)).sum(dim=1).sum(dim=-1, keepdim=True))
        sign_v2 = torch.sign((v_feat * v2.unsqueeze(1)).sum(dim=1).sum(dim=-1, keepdim=True))
        
        sign_v1 = sign_v1 + (sign_v1 == 0).float()
        sign_v2 = sign_v2 + (sign_v2 == 0).float()
        
        v1 = v1 * sign_v1
        v2 = v2 * sign_v2
        
        # 6. 叉乘保证生成符合右手定则的第三个轴，强制 det(R) = +1
        v3 = torch.cross(v1, v2, dim=-1)
        
        # 7. 组合成纯旋转矩阵 (B, 3, 3)
        R_align = torch.stack([v1, v2, v3], dim=-1)
        
        return R_align

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
        
        # [新增] 引入 LEFC 前端
        self.vn_frame_estimator = VNFrameEstimator()
        
        self.lstnet = LSTNet(out_dim=512)
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
        Args:
            point_cloud: (B, N, 3) 带有任意残缺和旋转的输入点云
        """
        # ==========================================
        # Phase 1: 规范化对齐 (Canonicalization)
        # ==========================================
        # 1. 提取等变规范坐标系 R_align
        pc_t = point_cloud.transpose(1, 2).contiguous() # 转为 (B, 3, N)
        R_align = self.vn_frame_estimator(pc_t) # 得到 (B, 3, 3) 旋转矩阵
        
        # 2. 将输入点云“摆正”至绝对规范空间
        pc_canonical = torch.bmm(point_cloud, R_align) # (B, N, 3)

        # ==========================================
        # Phase 2: 高精度处理 (Processing in Canonical Space)
        # ==========================================
        # 所有的 Baseline 操作都在被摆正的数据上执行，无惧外界旋转干扰
        coarse, symmetry_points, keyfeatures = self.lstnet(pc_canonical.transpose(2,1).contiguous())
        feat_symmetry = self.local_encoder(symmetry_points) # B,128,512
        feat_partial = keyfeatures # B,128,512
        fine1 = self.sgformer_1(coarse, feat_symmetry, feat_partial)
        fine2 = self.sgformer_2(fine1.transpose(2,1).contiguous(), feat_symmetry, feat_partial)

        if self.include_input: 
            fine2 = torch.cat([fine2, pc_canonical], dim=1).contiguous()

        coarse_points = coarse.transpose(2, 1).contiguous() # 提取为 (B, 1024, 3) 格式

        # ==========================================
        # Phase 3: 逆向复原 (De-canonicalization)
        # ==========================================
        # 3. 将补全后的点云旋转回真实的观测视角 (R_align 是正交阵，转置即为逆矩阵)
        R_inv = R_align.transpose(1, 2).contiguous()
        
        coarse_orig = torch.bmm(coarse_points, R_inv)
        fine1_orig = torch.bmm(fine1, R_inv)
        fine2_orig = torch.bmm(fine2, R_inv)

        rets = [coarse_orig.contiguous(), fine1_orig.contiguous(), fine2_orig.contiguous()]
        self.pred_dense_point = rets[-1]

        return rets