import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Resize


class ResidualConvolutionalAttention(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_feats,
                out_channels=out_feats,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=out_feats),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_feats,
                out_channels=out_feats,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=out_feats),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(
                in_channels=out_feats,
                out_channels=out_feats,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            nn.BatchNorm2d(num_features=out_feats),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_feats, out_channels=out_feats, kernel_size=1, stride=2
            ),
            nn.BatchNorm2d(num_features=out_feats),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.input_downsampler = nn.Conv2d(
            in_channels=in_feats, out_channels=out_feats, kernel_size=1, stride=2
        )

    def forward(self, x):
        identity = x
        identity = self.input_downsampler(identity)

        output = self.conv(x)
        output = F.sigmoid(output)

        attention_map = output * identity
        final_output = identity + output + attention_map

        return final_output, attention_map


class Network(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.RCA1 = ResidualConvolutionalAttention(in_channels, 64)
        self.RCA2 = ResidualConvolutionalAttention(64, 128)
        self.RCA3 = ResidualConvolutionalAttention(128, 256)
        self.Resizer = Resize((32, 32))
        self.attention_pooler = nn.Conv2d(
            in_channels=448, out_channels=256, kernel_size=1, stride=2
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=out_channels),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x, att1 = self.RCA1(x)
        x, att2 = self.RCA2(x)
        x, att3 = self.RCA3(x)

        att1 = self.Resizer(att1)
        att2 = self.Resizer(att2)
        att3 = self.Resizer(att3)

        pooled_attention = torch.cat((att1, att2, att3), dim=1)
        pooled_attention = self.attention_pooler(pooled_attention)
        pooled_attention = F.sigmoid(pooled_attention)

        final_x = pooled_attention * x
        final_x = self.final_conv(final_x)

        return final_x
