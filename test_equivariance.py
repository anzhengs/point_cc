import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from models.SymmCompletion import SymmCompletion
import yaml
from easydict import EasyDict

def load_model(cfg_path, ckpt_path):
    with open(cfg_path, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    model = SymmCompletion(cfg).cuda().eval()
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, cfg

def apply_random_so3(pcd):
    """对输入点云施加随机 SO(3) 旋转"""
    rot_mat = R.random().as_matrix().astype(np.float32)
    rot_mat_torch = torch.from_numpy(rot_mat).cuda()
    pcd_rotated = torch.matmul(pcd, rot_mat_torch)
    return pcd_rotated, rot_mat_torch

def test_equivariance():
    # 1. 加载模型（请修改路径为你的 Overfit 实验路径）
    cfg_path = 'cfgs/PCN_models/SymmCompletion.yaml'
    ckpt_path = 'experiments/Overfit_Sanity_Check/ckpt-best.pth'
    
    model, cfg = load_model(cfg_path, ckpt_path)
    
    # 2. 构造一个虚拟的单 Batch 输入（或从 DataLoader 取真实的一个 Batch）
    dummy_partial = torch.randn(1, 2048, 3).cuda()  # [B, N, 3]
    
    # 3. 前向传播两次：一次原始，一次旋转后
    with torch.no_grad():
        # 原始输入
        _, complete_original = model(dummy_partial)
        
        # 旋转输入
        dummy_partial_rot, rot_mat = apply_random_so3(dummy_partial)
        _, complete_rotated_pred = model(dummy_partial_rot)
        
        # 理论上：旋转原始输出 应该等于 旋转输入的输出
        complete_rotated_gt = torch.matmul(complete_original, rot_mat)
        
        # 计算误差
        error = torch.mean(torch.norm(complete_rotated_pred - complete_rotated_gt, dim=-1))
        print(f"[Equivariance Check] Rotation Error (CD-like): {error.item():.8f}")
        
        if error.item() < 1e-4:
            print("✅ 等变性验证通过！误差在数值噪声范围内。")
        else:
            print("⚠️  警告：等变性可能存在问题，请检查 VNMaxPool 或 VNLinear。")

if __name__ == '__main__':
    test_equivariance()