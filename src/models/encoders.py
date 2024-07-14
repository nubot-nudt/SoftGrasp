from torchvision.models import resnet18
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
import torch
from torch import nn

# from perceiver_pytorch import Perceiver
import torch.nn.functional as F
import torchaudio
from torch.nn.modules.activation import MultiheadAttention

class CoordConv(nn.Module):
    """Add coordinates in [0,1] to an image, like CoordConv paper."""

    def forward(self, x):
        # needs N,C,H,W inputs
        assert x.ndim == 4
        h, w = x.shape[2:]
        ones_h = x.new_ones((h, 1))
        type_dev = dict(dtype=x.dtype, device=x.device)
        lin_h = torch.linspace(-1, 1, h, **type_dev)[:, None]
        ones_w = x.new_ones((1, w))
        lin_w = torch.linspace(-1, 1, w, **type_dev)[None, :]
        new_maps_2d = torch.stack((lin_h * ones_w, lin_w * ones_h), dim=0)
        new_maps_4d = new_maps_2d[None]
        assert new_maps_4d.shape == (1, 2, h, w), (x.shape, new_maps_4d.shape)
        batch_size = x.size(0)
        new_maps_4d_batch = new_maps_4d.repeat(batch_size, 1, 1, 1)
        result = torch.cat((x, new_maps_4d_batch), dim=1)
        return result


class Encoder(nn.Module):
    def __init__(self, feature_extractor, out_dim=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.downsample = nn.MaxPool2d(2, 2)
        self.coord_conv = CoordConv()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if out_dim is not None:
            self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.coord_conv(x)
        x = self.feature_extractor(x)
        assert len(x.values()) == 1
        x = list(x.values())[0]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.fc is not None:
            x = self.fc(x)
        return x


def make_image_encoder(out_dim=None):
    image_extractor = resnet18(pretrained=True)
    image_extractor.conv1 = nn.Conv2d(
        5, 64, kernel_size=7, stride=1, padding=3, bias=False
    )
    image_extractor = create_feature_extractor(image_extractor, ["layer4.1.relu_1"])
    return Encoder(image_extractor, out_dim)

    
class AngleEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AngleEncoder, self).__init__()
        # 定义编码器的层和参数
        self.fc = nn.Linear(input_dim, output_dim)
        # self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        # 定义编码器的前向传播逻辑
        x = self.fc(x)
        x = self.relu(x)
        return x

class Angle_ProprioceptionEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super(Angle_ProprioceptionEncoder, self).__init__()
        # 利用多层感知机处理本体感觉信息
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)  # 确保输出尺寸和类型匹配
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class TorqueEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TorqueEncoder, self).__init__()
        # 定义编码器的层和参数
        self.fc = nn.Linear(input_dim, output_dim)
        # self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x = x.float()
        # 定义编码器的前向传播逻辑
        x = self.fc(x)
        x = self.relu(x)
        return x
    
class Torque_ProprioceptionEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super(Torque_ProprioceptionEncoder, self).__init__()
        # 利用多层感知机处理本体感觉信息
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)  # 确保输出尺寸和类型匹配
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
    
    
class share_Encoder_pos(nn.Module):
    def __init__(self, input_dim=256, output_dim=2):
        super(share_Encoder_pos, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.mlp(x)  
        return x
    
class share_Encoder(nn.Module):
    def __init__(self, input_dim=256, output_dim=2):
        super(share_Encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # print('x.shape',x.shape)
        x = self.mlp(x)  # 预期形状为 [batch, some_dimension, 256]
        # print('x.shape',x.shape)
        pooled_x_list = []

        for i in range(x.shape[0]):
            # 应用池化到 (some_dimension, 256) 维度
            pooled_x = F.adaptive_avg_pool1d(x[i:i+1].permute(0, 2, 1), 1).permute(0, 2, 1)
            pooled_x_list.append(pooled_x)

        # 拼接池化后的输出
        x = torch.cat(pooled_x_list, dim=0)  # [batch, 1, 256]
        x = x.squeeze(dim=1)  # 调整形状为 [batch, 256]
        return x

    
def make_angle_encoder(in_dim,out_dim):
    encoder = AngleEncoder(in_dim,out_dim).to('cuda')
    return encoder

def make_torque_encoder(in_dim,out_dim):
    encoder = TorqueEncoder(in_dim,out_dim).to('cuda')
    return encoder

def make_angle_Proprioceptionencoder(in_dim,out_dim):
    encoder = Angle_ProprioceptionEncoder(in_dim,out_dim).to('cuda')
    return encoder

def make_torque_Proprioceptionencoder(in_dim,out_dim):
    encoder = Torque_ProprioceptionEncoder(in_dim,out_dim).to('cuda')
    return encoder

def make_torque_encoder(in_dim,out_dim):
    encoder = Torque_ProprioceptionEncoder(in_dim,out_dim).to('cuda')
    return encoder

def make_share_POS_encoder(in_dim,out_dim):
    encoder = share_Encoder_pos(in_dim,out_dim).to('cuda')
    return encoder

def make_share_encoder(in_dim,out_dim):
    encoder = share_Encoder(in_dim,out_dim).to('cuda')
    return encoder



if __name__ == "__main__":
    inp = torch.zeros((1, 3, 480, 640))
    encoder = make_image_encoder(64, 1280)
    print(encoder(inp).shape)

 