import torch

# 替换成你的文件路径
file_path = 'd2v_256d.pt'

# 加载 .pt 文件
data = torch.load(file_path, map_location='cpu')

# 打印类型
print(f"类型: {type(data)}")

# 如果是 dict，打印所有 key
if isinstance(data, dict):
    print(f"包含的 keys: {list(data.keys())}")

# 如果是 tensor，打印形状
elif isinstance(data, torch.Tensor):
    print(f"张量形状: {data.shape}")
