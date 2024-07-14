import torch
import torch.nn as nn
import torch.optim as optim

class MultimodalNetwork(nn.Module):
    def __init__(self):
        super(MultimodalNetwork, self).__init__()
        # 定义针对各种输入的编码器
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.ft_sensor_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.ee_pose_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # 将不同模态的信息融合
        self.fusion = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 解码器，用于预测任务相关的输出（例如：操作条件的光流）
        self.decoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 假设输出维度为3，例如xyz位置
        )

    def forward(self, rgb, depth, ft_readings, ee_pose):
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth)
        ft_features = self.ft_sensor_encoder(ft_readings)
        ee_features = self.ee_pose_encoder(ee_pose)

        # 合并所有特征
        combined_features = torch.cat(
            [rgb_features.flatten(start_dim=1),
             depth_features.flatten(start_dim=1),
             ft_features, ee_features], dim=1)

        fused_features = self.fusion(combined_features)
        output = self.decoder(fused_features)
        return output

# 实例化模型
model = MultimodalNetwork()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 示例输入数据
rgb_input = torch.randn(1, 3, 64, 64)  # 假设输入RGB图片大小为64x64
depth_input = torch.randn(1, 1, 64, 64)  # 假设输入深度图大小为64x64
ft_input = torch.randn(1, 6)  # 力/扭矩传感器读数
ee_input = torch.randn(1, 6)  # 末端执行器位置和方向

# 模型训练的一个简单迭代
model.train()
optimizer.zero_grad()
output = model(rgb_input, depth_input, ft_input, ee_input)
loss = criterion(output, torch.randn(1, 3))  # 随机生成一个目标输出作为示例
loss.backward()
optimizer.step()

print("Training loss:", loss.item())



import torch
import torch.nn as nn
import torch.optim as optim

# 改进的多头注意力机制
class MultiHeadAttentionModule(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttentionModule, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
    
    def forward(self, query, key, value):
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        return attn_output, attn_weights

# 在变分自编码器中添加这个模块
class VariationalAutoencoderWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, heads):
        super(VariationalAutoencoderWithAttention, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            MultiHeadAttentionModule(hidden_dim, heads),
            nn.Linear(hidden_dim, latent_dim * 2) # 输出均值和对数方差
        )
        
        self.decoder = nn.Sequential(
            MultiHeadAttentionModule(latent_dim, heads),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.unsqueeze(0)  # 增加一个批处理维度
        encoded, encoder_attn_weights = self.encoder[2](x, x, x)
        params = self.encoder[3](encoded)
        mu, log_var = params.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)
        decoded, decoder_attn_weights = self.decoder[0](z, z, z)
        recon_x = self.decoder[3](self.decoder[2](self.decoder[1](decoded)))
        return recon_x.squeeze(0), mu, log_var, encoder_attn_weights, decoder_attn_weights

# 参数设定
input_dim = 1024
hidden_dim = 400
latent_dim = 20
heads = 8  # 多头注意力的头数

# 模型初始化
model = VariationalAutoencoderWithAttention(input_dim, hidden_dim, latent_dim, heads)
optimizer = optim.Adam(model.parameters(), lr=1e-3)