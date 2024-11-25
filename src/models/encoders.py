from torchvision.models import resnet18
from torchvision.models.feature_extraction import (
    create_feature_extractor
)
import torch
from torch import nn

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


class Angle_ProprioceptionEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super(Angle_ProprioceptionEncoder, self).__init__()
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

class Torque_ProprioceptionEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super(Torque_ProprioceptionEncoder, self).__init__()
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




def make_angle_Proprioceptionencoder(in_dim,out_dim):
    encoder = Angle_ProprioceptionEncoder(in_dim,out_dim).to('cuda')
    return encoder

def make_torque_Proprioceptionencoder(in_dim,out_dim):
    encoder = Torque_ProprioceptionEncoder(in_dim,out_dim).to('cuda')
    return encoder



if __name__ == "__main__":
    inp = torch.zeros((1, 3, 480, 640))
    encoder = make_image_encoder(64, 1280)
    print(encoder(inp).shape)

 