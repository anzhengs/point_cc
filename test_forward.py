import torch
from models.build import build_model
from omegaconf import OmegaConf

# 构造模拟配置（对齐原代码格式）
config = OmegaConf.create({
    "up_factors": "2,2",
    "include_input": False,
    "loss": {"sparse_loss_weight": 1.0, "dense_loss_weight": 1.0}
})

# 构建模型并测试
model = build_model(config, model_type="SymmCompletion").cuda()
x = torch.randn(2, 2048, 3).cuda()  # 模拟输入：B=2, N=2048, 3
rets = model(x)

# 验证输出维度（与原代码一致）
assert rets[0].shape == (2, 1024, 3), "Coarse维度错误"
assert rets[-1].shape == (2, 4096, 3), "Fine2维度错误"
print("✅ 前向传播无错误，维度匹配！")