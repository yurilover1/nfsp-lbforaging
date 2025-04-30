import torch

# 替换为你的路径
model_path = './partners/agents_for_5*5/agent_0_1.pt'

# 尝试加载
model_data = torch.load(model_path, map_location='cpu')  # 防止自动加载到 GPU

# 判断是完整模型还是 state_dict
if isinstance(model_data, dict):
    if 'state_dict' in model_data or any(isinstance(v, torch.Tensor) for v in model_data.values()):
        print("这是 state_dict（模型参数）")
    else:
        print("这是一个普通字典，可能包含其他内容")
else:
    print("这是完整模型对象")