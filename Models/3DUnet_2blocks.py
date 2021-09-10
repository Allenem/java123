import torch
from torch import nn


class UnetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(UnetDownBlock, self).__init__()
        self.poolingOrNot = pooling
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        out_before_pooling = self.convs(x)
        if self.poolingOrNot:
            out = self.maxpool(out_before_pooling)
            return out_before_pooling, out
        else:
            # the last downblock without pooling
            return out_before_pooling


class UnetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels * 2,
                                           out_channels,
                                           kernel_size=2,
                                           stride=2)
        self.convs = nn.Sequential(
            nn.Conv3d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x, x_bridge):
        x_up = self.upsample(x)
        x_concat = torch.cat([x_up, x_bridge], dim=1)
        out = self.convs(x_concat)

        return out


class Unet3D(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, n_base_channels=64):
        super(Unet3D, self).__init__()
        self.down_blocks = nn.ModuleList([
            UnetDownBlock(in_ch, n_base_channels),
            UnetDownBlock(n_base_channels * 1, n_base_channels * 2),
            UnetDownBlock(n_base_channels * 2, n_base_channels * 4),
            UnetDownBlock(n_base_channels * 4, n_base_channels * 8),
            UnetDownBlock(n_base_channels * 8, n_base_channels * 16),
        ])
        self.up_blocks = nn.ModuleList([
            UnetUpBlock(n_base_channels * 8, n_base_channels * 8),
            UnetUpBlock(n_base_channels * 4, n_base_channels * 4),
            UnetUpBlock(n_base_channels * 2, n_base_channels * 2),
            UnetUpBlock(n_base_channels * 1, n_base_channels * 1),
        ])
        self.final_block = nn.Sequential(
            nn.Conv3d(n_base_channels, out_ch, kernel_size=1))

    def forward(self, x):
        out = x
        outputs_before_pooling = []

        for i, block in enumerate(self.down_blocks):
            out_before_pooling, out = block(out)
            # outputs_before_pooling = [block1_out_before_pooling, ..., block4_out_before_pooling]
            outputs_before_pooling.append(out_before_pooling)
        # last downblock without pooling
        out = out_before_pooling

        for i, block in enumerate(self.up_blocks):
            # concat [i, -i-2], eg. [0, -2], [1, -3], [2, -4], [3, -5]
            out = block(out, outputs_before_pooling[-i - 2])
        out = self.final_block(out)

        return out.sigmoid()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = [128, 128, 128]
x = torch.Tensor(2, 3, image_size[0], image_size[1],
                 image_size[2])  # batch_size, channel, w, h, frames
x.to(device)
print("x size: {}".format(x.size()))

model = Unet3D()
print(model)

out = model(x)
print("out size: {}".format(out.size()))
