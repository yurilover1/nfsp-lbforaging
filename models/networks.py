import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    """
    密集连接的神经网络,用于策略和Q值估计
    """
    def __init__(self, input_size, output_size, hidden_units=256, num_layers=5):
        super(DenseNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        
        # 输入层
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_units)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_units)])
        
        # 隐藏层
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
            self.batch_norms.append(nn.BatchNorm1d(hidden_units))
            
        # 输出层
        self.output = nn.Linear(hidden_units, output_size)
        
    def forward(self, x):
        """前向传播"""
        # 确保输入是2D张量
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        # 通过所有隐藏层
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            
        # 输出层
        x = self.output(x)
        
        return x
        
    def act(self, state, epsilon=0.0):
        """根据状态选择动作"""
        if torch.rand(1) < epsilon:
            # 随机探索
            return torch.randint(self.output_size, (1,))
        else:
            # 贪婪选择
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax(dim=1) 