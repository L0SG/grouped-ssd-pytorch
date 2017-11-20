import torch.nn as nn
import torch.nn.functional as F
import torch

# unet implementation
# base code borrowed from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/unet.py


class unet(nn.Module):
    # unet architecture class
    # uses unetDown and unetUp class for skip connections in the model
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        """
        :param feature_scale: scale factor of feature (1: original, 4: 4x smaller # of conv feature map)
        :param n_classes: number of classes
        :param is_deconv: switch for deconv in upsampling
        :param in_channels: number of input channels
        :param is_batchnorm: switch for batchnorm in downsampling
        """
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # filter size definition
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # softmax
        self.softmax = nn.Softmax2d()

    def forward(self, inputs):
        # forward pass of inputs
        # downsampling pass: unetConv2 then maxpool
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        # upsampling with skip connection from downsampling layers
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)
        softmax = self.softmax(final)

        return softmax


class unetUp(nn.Module):
    # unet upsampling class
    # composed of upsampling and concat with corresponding downsampling layer
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        # batchnorm is explicitly false
        self.conv = unetConv2(in_size, out_size, False)
        # if using deconv, apply transposed conv
        # else, apply bilinear upsampling
        if is_deconv:
            # fix: add stride parameter of 2 (which is upsample by 2)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        # inputs1: tensor output from conv layer for crop and concat
        # inputs2: tensor for upsampling

        # upsample the input
        outputs2 = self.up(inputs2)

        # crop the output from conv layer of downsampling pass to match the size
        offset = outputs2.size()[2] - inputs1.size()[2]
        # fix: remove +1 of the second element (since feature image is always a square)
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)

        # concatenate the outputs then apply conv
        return self.conv(torch.cat([outputs1, outputs2], 1))


class unetConv2(nn.Module):
    # 2d conv class for unet
    # (conv2d -> BN(optional) -> ReLU) x 2
    # the only difference is conditional application of batchnorm (downsampling only)
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()
        # apply batchnorm only if the flag is true
        # else, apply conv only

        # fix: replace padding from 1 to 0 (as in the paper), which fixes the concat dim mismatch
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 0),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 0),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 0),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 0),
                                       nn.ReLU(),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs