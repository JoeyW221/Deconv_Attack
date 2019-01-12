import torch
import torch.nn as nn
import torchvision.models as models
import sys


class _vgg16Deconv(nn.Module):
    def __init__(self):
        super(_vgg16Deconv, self).__init__()

        self.features = nn.Sequential(
            # deconv1
            nn.AvgPool2d(kernel_size=1, stride=1),
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1),
            # deconv2
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 3, padding=1),
            # deconv3
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 3, padding=1),
            # deconv4
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            # deconv5
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, 3, padding=1)
        )

        # conv idx : deconv idx
        self.conv2deconv_indices = {0: 44, 3: 41, 7: 37, 10: 34, 14: 30,
                                    17: 27, 20: 24, 24: 20, 27: 17,
                                    30: 14, 34: 10, 37: 7, 40: 4}
        # conv bias idx : deconv bias idx;
        self.conv2deconv_bias_indices = {
            0: 41, 3: 37, 7: 34, 10: 30, 14: 27, 17: 24, 20: 20, 24: 17,
            27: 14, 30: 10, 34: 7, 37: 4}
        # relu idx : de-relu idx
        self.relu2relu_indices = {2: 42, 5: 39, 9: 35, 12: 32, 16: 28,
                                  19: 25, 22: 22, 26: 18, 29: 15,
                                  32: 12, 36: 8, 39: 5, 42: 2}
        # unpool idx : pool idx
        self.unpool2pool_indices = {38: 6, 31: 13, 21: 23, 11: 33, 1: 43}

    def forward(self, x, layer, pool_locs):
        if layer in self.conv2deconv_indices:
            start_idx = self.conv2deconv_indices[layer]
        elif layer in self.relu2relu_indices:
            start_idx = self.relu2relu_indices[layer]
        elif layer == 44 or layer == 43:
            start_idx = 1
            # print('start_idx: ', start_idx)
        else:
            print('No such Conv2d or RelU layer!')
            sys.exit(0)

        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                x = self.features[idx](
                    x, pool_locs[self.unpool2pool_indices[idx]])
            else:
                x = self.features[idx](x)

        return x

# vgg16_deconv = _vgg16Deconv()
# print(vgg16_deconv)
