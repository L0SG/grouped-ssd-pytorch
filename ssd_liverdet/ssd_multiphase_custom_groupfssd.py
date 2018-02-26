import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v2
import os

GROUPS_VGG = 4
GROUPS_EXTRA = 4
use_fuseconv = True
feature_scale = 1

def xavier(param):
    nn.init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head, num_classes, batch_norm):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 300

        # SSD network
        self.vgg = nn.ModuleList(base)

        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

        # FSSD extra layers before fusion
        self.conv81 = nn.Conv2d(1024, 256, kernel_size=1, groups=GROUPS_EXTRA)
        self.conv81.apply(weights_init)
        self.bn_conv81 = nn.BatchNorm2d(256)
        self.conv82 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, groups=GROUPS_EXTRA)
        self.conv82.apply(weights_init)
        self.bn_conv82 = nn.BatchNorm2d(512)
        self.bn_fused = nn.BatchNorm2d(768)
        # FSSD extra layers after fusion
        self.conv91 = nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1, groups=GROUPS_EXTRA)
        self.conv91.apply(weights_init)
        self.bn_conv91 = nn.BatchNorm2d(512)
        self.conv101 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=GROUPS_EXTRA)
        self.conv101.apply(weights_init)
        self.bn_conv101 = nn.BatchNorm2d(512)
        self.conv111 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1, groups=GROUPS_EXTRA)
        self.conv111.apply(weights_init)
        self.bn_conv111 = nn.BatchNorm2d(256)
        self.conv121 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=GROUPS_EXTRA)
        self.conv121.apply(weights_init)
        self.bn_conv121 = nn.BatchNorm2d(256)
        self.conv131 = nn.Conv2d(256, 256, kernel_size=3, stride=1, groups=GROUPS_EXTRA)
        self.conv131.apply(weights_init)
        self.bn_conv131 = nn.BatchNorm2d(256)
        self.conv141 = nn.Conv2d(256, 256, kernel_size=3, stride=1, groups=GROUPS_EXTRA)
        self.conv141.apply(weights_init)
        self.bn_conv141 = nn.BatchNorm2d(256)


        # FSSD fuse layers
        self.fuse_conv43 = nn.Conv2d(512, 256, kernel_size=1, groups=GROUPS_VGG)
        self.fuse_conv43.apply(weights_init)
        self.fuse_fc7 = nn.Conv2d(1024, 256, kernel_size=1, groups=GROUPS_VGG)
        self.fuse_fc7.apply(weights_init)
        self.fuse_fc7_bilinear = nn.UpsamplingBilinear2d(size=(38, 38))
        self.fuse_conv82 = nn.Conv2d(512, 256, kernel_size=1, groups=GROUPS_EXTRA)
        self.fuse_conv82.apply(weights_init)
        self.fuse_conv82_bilienar = nn.UpsamplingBilinear2d(size=(38, 38))

        # groupfuse layers
        if use_fuseconv is True:
            # fuse conv layers for mixing grouped feature map
            # it adds 2 extra 1x1 conv just before attaching to the sources

            self.fuse_11 = nn.Conv2d(512*feature_scale, 512*feature_scale, kernel_size=1)
            self.fuse_11.apply(weights_init)
            if batch_norm:
                self.bn_fuse_11 = nn.BatchNorm2d(512*feature_scale)
            # self.fuse_12 = nn.Conv2d(512*feature_scale, 512*feature_scale, kernel_size=1)
            # self.fuse_12.apply(weights_init)
            # if batch_norm:
            #     self.bn_fuse_12 = nn.BatchNorm2d(512*feature_scale)

            self.fuse_21 = nn.Conv2d(512*feature_scale, 512*feature_scale, kernel_size=1)
            self.fuse_21.apply(weights_init)
            if batch_norm:
                self.bn_fuse_21 = nn.BatchNorm2d(512*feature_scale)
            # self.fuse_22 = nn.Conv2d(1024*feature_scale, 1024*feature_scale, kernel_size=1)
            # self.fuse_22.apply(weights_init)
            # if batch_norm:
            #     self.bn_fuse_22 = nn.BatchNorm2d(1024*feature_scale)

            self.fuse_31 = nn.Conv2d(256*feature_scale, 256*feature_scale, kernel_size=1)
            self.fuse_31.apply(weights_init)
            if batch_norm:
                self.bn_fuse_31 = nn.BatchNorm2d(256*feature_scale)
            # self.fuse_32 = nn.Conv2d(512*feature_scale, 512*feature_scale, kernel_size=1)
            # self.fuse_32.apply(weights_init)
            # if batch_norm:
            #     self.bn_fuse_32 = nn.BatchNorm2d(512*feature_scale)

            self.fuse_41 = nn.Conv2d(256*feature_scale, 256*feature_scale, kernel_size=1)
            self.fuse_41.apply(weights_init)
            if batch_norm:
                self.bn_fuse_41 = nn.BatchNorm2d(256*feature_scale)
            # self.fuse_42 = nn.Conv2d(256*feature_scale, 256*feature_scale, kernel_size=1)
            # self.fuse_42.apply(weights_init)
            # if batch_norm:
            #     self.bn_fuse_42 = nn.BatchNorm2d(256*feature_scale)

            self.fuse_51 = nn.Conv2d(256*feature_scale, 256*feature_scale, kernel_size=1)
            self.fuse_51.apply(weights_init)
            if batch_norm:
                self.bn_fuse_51 = nn.BatchNorm2d(256*feature_scale)
            # self.fuse_52 = nn.Conv2d(256*feature_scale, 256*feature_scale, kernel_size=1)
            # self.fuse_52.apply(weights_init)
            # if batch_norm:
            #     self.bn_fuse_52 = nn.BatchNorm2d(256*feature_scale)

            self.fuse_61 = nn.Conv2d(256*feature_scale, 256*feature_scale, kernel_size=1)
            self.fuse_61.apply(weights_init)
            if batch_norm:
                self.bn_fuse_61 = nn.BatchNorm2d(256*feature_scale)
            # self.fuse_62 = nn.Conv2d(256*feature_scale, 256*feature_scale, kernel_size=1)
            # self.fuse_62.apply(weights_init)
            # if batch_norm:
            #     self.bn_fuse_62 = nn.BatchNorm2d(256*feature_scale)

            self.fuse_list1 = nn.ModuleList([self.fuse_31, self.fuse_41, self.fuse_51, self.fuse_61])
            # self.fuse_list2 = nn.ModuleList([self.fuse_32, self.fuse_42, self.fuse_52, self.fuse_62])
            if batch_norm:
                self.bn_fuse_list1 = nn.ModuleList([self.bn_fuse_31, self.bn_fuse_41, self.bn_fuse_51, self.bn_fuse_61])
                # self.bn_fuse_list2 = nn.ModuleList([self.bn_fuse_32, self.bn_fuse_42, self.bn_fuse_52, self.bn_fuse_62])



    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        # TODO: change hardcoding 23 for BN case
        if self.batch_norm is False:
            idx_until_conv4_3 = 23
        elif self.batch_norm is True:
            idx_until_conv4_3 = 33
        for k in range(idx_until_conv4_3):
            x = self.vgg[k](x)
        x_conv43 = x


        #s = self.L2Norm(x)
        # TODO: append lower level features
        #sources.append(s)

        # apply vgg up to fc7
        for k in range(idx_until_conv4_3, len(self.vgg)):
            x = self.vgg[k](x)
        #sources.append(x)
        x_fc7 = x

        # apply extra up to conv82
        conv81 = self.conv81(x_fc7)
        if self.batch_norm:
            conv81 = self.bn_conv81(conv81)
        conv81 = F.relu(conv81)
        conv82 = self.conv82(conv81)
        if self.batch_norm:
            conv82 = self.bn_conv82(conv82)
        conv82 = F.relu(conv82)

        x_conv82 = conv82

        # fuse 3 layers: conv43, fc7, conv82
        fuse_conv43 = self.fuse_conv43(x_conv43)
        fuse_fc7 = self.fuse_fc7(x_fc7)
        fuse_fc7 = self.fuse_fc7_bilinear(fuse_fc7)
        fuse_conv82 = self.fuse_conv82(x_conv82)
        fuse_conv82 = self.fuse_conv82_bilienar(fuse_conv82)

        # concat
        fused = torch.cat([fuse_conv43, fuse_fc7, fuse_conv82], dim=1)
        if self.batch_norm:
            fused = self.bn_fused(fused)

        # apply extras for feature map
        conv91 = self.conv91(fused)
        if self.batch_norm:
            conv91 = self.bn_conv91(conv91)
        conv91 = F.relu(conv91)

        conv101 = self.conv101(conv91)
        if self.batch_norm:
            conv101 = self.bn_conv101(conv101)
        conv101 = F.relu(conv101)

        conv111 = self.conv111(conv101)
        if self.batch_norm:
            conv111 = self.bn_conv111(conv111)
        conv111 = F.relu(conv111)

        conv121 = self.conv121(conv111)
        if self.batch_norm:
            conv121 = self.bn_conv121(conv121)
        conv121 = F.relu(conv121)

        conv131 = self.conv131(conv121)
        if self.batch_norm:
            conv131 = self.bn_conv131(conv131)
        conv131 = F.relu(conv131)

        conv141 = self.conv141(conv131)
        if self.batch_norm:
            conv141 = self.bn_conv141(conv141)
        conv141 = F.relu(conv141)

        # apply group fusion layer
        if use_fuseconv:
            conv91 = self.fuse_11(conv91)
            conv101 = self.fuse_21(conv101)
            conv111 = self.fuse_31(conv111)
            conv121 = self.fuse_41(conv121)
            conv131 = self.fuse_51(conv131)
            conv141 = self.fuse_61(conv141)
            if self.batch_norm:
                conv91 = self.bn_fuse_11(conv91)
                conv101 = self.bn_fuse_21(conv101)
                conv111 = self.bn_fuse_31(conv111)
                conv121 = self.bn_fuse_41(conv121)
                conv131 = self.bn_fuse_51(conv131)
                conv141 = self.bn_fuse_61(conv141)
        sources.append(conv91)
        sources.append(conv101)
        sources.append(conv111)
        sources.append(conv121)
        sources.append(conv131)
        sources.append(conv141)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                #self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            weight_pretrained = torch.load(base_file, map_location=lambda storage, loc: storage)
            self.load_state_dict(weight_pretrained)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            # depthwise separable conv: add groups=4 (4 phases)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, groups=GROUPS_VGG)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6, groups=GROUPS_VGG)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1, groups=GROUPS_VGG)
    if batch_norm:
        layers += [pool5,
                   conv6, nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
                   conv7, nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]
    else:
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if batch_norm is False:
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1, groups=GROUPS_EXTRA)]
                else:
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                         kernel_size=(1, 3)[flag], stride=2, padding=1, groups=GROUPS_EXTRA),
                               nn.BatchNorm2d(cfg[k + 1])]

            else:
                if batch_norm is False:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag], groups=GROUPS_EXTRA)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag], groups=GROUPS_EXTRA),
                               nn.BatchNorm2d(v)]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes, batch_norm):
    loc_layers = []
    conf_layers = []
    # hard-coded

    # TODO: make this generic
    """
    if batch_norm is False:
        vgg_source = [21, -2]
    elif batch_norm is True:
        vgg_source = [30, -3]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    # hard-coded
    if batch_norm is False:
        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]
    elif batch_norm is True:
        for k, v in enumerate(extra_layers[2::4], 2):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]
    """
    source_output = [512, 512, 256, 256, 256, 256]
    for idx in range(len(source_output)):
        loc_layers += [nn.Conv2d(source_output[idx], cfg[idx] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(source_output[idx], cfg[idx] * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    512, 512, 512],
    #'300': [128, 128, 'M', 256, 256, 'M', 512, 512, 512, 'C', 1024, 1024, 1024, 'M',
    #1024, 1024, 1024],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    #'300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    # for v2_custom cfg: use 6 for lowest layer
    '300': [4, 6, 6, 6, 4, 4],
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21, batch_norm=False):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
        return

    # change the input channel from i=3 to 12
    return SSD(phase, *multibox(vgg(base[str(size)], i=12, batch_norm=batch_norm),
                                add_extras(extras[str(size)], 512, batch_norm),
                                mbox[str(size)], num_classes, batch_norm), num_classes, batch_norm)
