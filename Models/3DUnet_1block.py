import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.dbconv = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, input):
        return self.dbconv(input)


class UNet3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = DoubleConv(in_ch, 32, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv2 = DoubleConv(64, 64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv3 = DoubleConv(128, 128, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv4 = DoubleConv(256, 256, 512)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.conv5 = DoubleConv(512, 512, 512)

        self.up6 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(256 + 512, 256, 256)
        self.up7 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(128 + 256, 128, 128)
        self.up8 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(64 + 128, 64, 64)
        self.up9 = nn.ConvTranspose3d(64, 48, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(48 + 64, 64, 32)
        self.conv10 = nn.Conv3d(32, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        print("c1 size: {}".format(c1.size()))
        p1 = self.pool1(c1)
        print("p1 size: {}".format(p1.size()))

        c2 = self.conv2(p1)
        print("c2 size: {}".format(c2.size()))
        p2 = self.pool2(c2)
        print("p2 size: {}".format(p2.size()))

        c3 = self.conv3(p2)
        print("c3 size: {}".format(c3.size()))
        p3 = self.pool3(c3)
        print("p3 size: {}".format(p3.size()))

        c4 = self.conv4(p3)
        print("c4 size: {}".format(c4.size()))
        p4 = self.pool4(c4)
        print("p4 size: {}".format(p4.size()))

        c5 = self.conv5(p4)
        print("c5 size: {}".format(c5.size()))

        up_6 = self.up6(c5)
        print("up_6 size: {}".format(up_6.size()))
        cat6 = torch.cat([up_6, c4], dim=1)
        print("cat6 size: {}".format(cat6.size()))
        c6 = self.conv6(cat6)
        print("c6 size: {}".format(c6.size()))

        up_7 = self.up7(c6)
        print("up_7 size: {}".format(up_7.size()))
        cat7 = torch.cat([up_7, c3], dim=1)
        print("cat7 size: {}".format(cat7.size()))
        c7 = self.conv7(cat7)
        print("c7 size: {}".format(c7.size()))

        up_8 = self.up8(c7)
        print("up_8 size: {}".format(up_8.size()))
        cat8 = torch.cat([up_8, c2], dim=1)
        print("cat8 size: {}".format(cat8.size()))
        c8 = self.conv8(cat8)
        print("c8 size: {}".format(c8.size()))

        up_9 = self.up9(c8)
        print("up_9 size: {}".format(up_9.size()))
        cat9 = torch.cat([up_9, c1], dim=1)
        print("cat9 size: {}".format(cat9.size()))
        c9 = self.conv9(cat9)
        print("c9 size: {}".format(c9.size()))

        c10 = self.conv10(c9)
        print("c10 size: {}".format(c10.size()))
        sigmoidout = nn.Sigmoid()(c10)
        print("sigmoidout size: {}".format(sigmoidout.size()))

        # return out
        return c10


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = [128, 128, 128]
x = torch.Tensor(2, 3, image_size[0], image_size[1],
                 image_size[2])  # batch_size, channel, w, h, frames
x.to(device)
print("x size: {}".format(x.size()))

model = UNet3D(3, 1)
# print(model)

out = model(x)
print("out size: {}".format(out.size()))