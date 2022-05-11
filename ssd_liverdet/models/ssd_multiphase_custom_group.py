import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from layers import self_attn
#from dcn_v2 import DCN
from layers.dcn_v2_custom import DCN
from utils.show_offset import show_dconv_offset
from data import v2
import os
import numpy as np

def xavier(param):
    nn.init.xavier_uniform_(param)


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

    def __init__(self, phase, base, extras, head,
                 num_classes, batch_norm, groups_vgg, groups_extra, feature_scale, use_fuseconv, use_self_attention, use_self_attention_base, num_dcn_layers, groups_dcn, dcn_cat_sab, detach_sab,
                 max_pool_factor):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.batch_norm = batch_norm
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward())
        self.priors.requires_grad = False
        self.size = 300

        self.groups_vgg = groups_vgg
        self.groups_extra = groups_extra
        self.feature_scale = feature_scale
        self.use_fuseconv = use_fuseconv
        self.use_self_attention = use_self_attention
        self.use_self_attention_base = use_self_attention_base
        self.num_dcn_layers = num_dcn_layers

        # SSD network
        self.vgg = nn.ModuleList(base)

        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512*feature_scale, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            # PyTorch1.5.0 support new-style autograd function
            # self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            self.detect = Detect()

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

            self.fuse_21 = nn.Conv2d(1024*feature_scale, 1024*feature_scale, kernel_size=1)
            self.fuse_21.apply(weights_init)
            if batch_norm:
                self.bn_fuse_21 = nn.BatchNorm2d(1024*feature_scale)
            # self.fuse_22 = nn.Conv2d(1024*feature_scale, 1024*feature_scale, kernel_size=1)
            # self.fuse_22.apply(weights_init)
            # if batch_norm:
            #     self.bn_fuse_22 = nn.BatchNorm2d(1024*feature_scale)

            self.fuse_31 = nn.Conv2d(512*feature_scale, 512*feature_scale, kernel_size=1)
            self.fuse_31.apply(weights_init)
            if batch_norm:
                self.bn_fuse_31 = nn.BatchNorm2d(512*feature_scale)
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

        self.max_pool_factor = max_pool_factor
        if self.use_self_attention:
            self.self_attn_list = nn.ModuleList([])
            self.self_attn_in_channel_list = [512, 1024, 512, 256, 256, 256]
            self.self_attn_in_channel_list = [i * feature_scale for i in self.self_attn_in_channel_list]
            for i in range(len(self.self_attn_in_channel_list)):
                self.self_attn_list.append(self_attn.Self_Attn(in_channels=self.self_attn_in_channel_list[i], max_pool_factor=max_pool_factor))

        if self.use_self_attention_base:
            self.self_attn_base_list = nn.ModuleList([])
            self.self_attn_base_in_channel_list = [512, 1024, 512, 256, 256, 256]
            self.self_attn_base_in_channel_list = [i * feature_scale for i in self.self_attn_base_in_channel_list]
            for i in range(len(self.self_attn_base_in_channel_list)):
                self.self_attn_base_list.append(self_attn.Self_Attn(in_channels=self.self_attn_base_in_channel_list[i], max_pool_factor=max_pool_factor))

        if self.num_dcn_layers > 0:
            self.use_dcn = True
            self.groups_dcn = groups_dcn
            self.dcn_list = nn.ModuleList([])
            # we try to match the offset of each phase after vgg conv7 stage
            self.dcn_in_channel_list = [512]
            self.dcn_in_channel_list = [i * feature_scale for i in self.dcn_in_channel_list]
            self.dcn_cat_sab = dcn_cat_sab  # concat sab (self_attention_base) and the original x before dcn input
            self.detach_sab = detach_sab
            if self.detach_sab:
                assert self.dcn_cat_sab is True, "deatch_sab requires --dcn_cat_sab=True"
            for i in range(len(self.dcn_in_channel_list)):
                if self.dcn_cat_sab:
                    assert self.use_self_attention_base is True, "dcn_cat_sab requires use_self_attention_base=True"
                    self.dcn_list.append(
                        DCN(in_channels=self.dcn_in_channel_list[i] * 2, out_channels=self.dcn_in_channel_list[i],
                            kernel_size=3, stride=1, padding=1, deformable_groups=self.groups_dcn))
                else:
                    self.dcn_list.append(
                        DCN(in_channels=self.dcn_in_channel_list[i], out_channels=self.dcn_in_channel_list[i],
                            kernel_size=3, stride=1, padding=1, deformable_groups=self.groups_dcn))
                for j in range(self.num_dcn_layers-1):
                    self.dcn_list.append(DCN(in_channels=self.dcn_in_channel_list[i], out_channels=self.dcn_in_channel_list[i],
                                             kernel_size=3, stride=1, padding=1, deformable_groups=self.groups_dcn))
        else:
            self.use_dcn = False
            self.dcn_cat_sab = False
            self.detach_sab = False

    def slice_and_cat(self, a, b):
        # slice each tensor by 4 (4-phase), concat a & b for each phase, then merge together
        # why?: we want to keep the "grouped" context of base convnet feature before feeding to next grouped conv
        a = torch.split(a, int(a.size(1)/self.groups_vgg), dim=1)
        b = torch.split(b, int(b.size(1)/self.groups_vgg), dim=1)
        ab = [torch.cat([a[i], b[i]], dim=1) for i in range(len(a))]
        ab = torch.cat(ab, dim=1)
        return ab

    def visualize_offset(self, offset, x_orig):
        import matplotlib
        matplotlib.use('module://backend_interagg')
        # try to visualize offset predicted by DCN

        o1, o2 = torch.chunk(offset, 2, dim=1)
        o1_grp = torch.chunk(o1, self.groups_dcn, dim=1)
        o2_grp = torch.chunk(o2, self.groups_dcn, dim=1)

        x_orig_grp = x_orig.view(1, 4, 3, x_orig.shape[2], x_orig.shape[3])

        for i in range(x_orig_grp.shape[2]):
            img = x_orig_grp[0, i, 1].cpu().unsqueeze(-1).numpy()
            # make rgb uint8
            img = (np.repeat(img, 3, axis=2) * 255).astype(np.uint8)
            o1_ = o1_grp[i]
            o2_ = o2_grp[i]
            offset_ = torch.cat((o1_, o2_), dim=1)
            show_dconv_offset(img, [offset_])

        print("test")
        return

    def forward(self, x, visualize=False):
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
        if visualize:
            x_orig = x.clone()  # save for later
            all_offset = []
            all_attnb = []
            all_attn = []

        sources = list()
        loc = list()
        conf = list()
        if self.use_self_attention:
            sa_counter = 0
        if self.use_self_attention_base:
            sa_base_counter = 0
        if self.use_dcn:
            dcn_counter = 0

        # apply vgg up to conv4_3 relu
        # TODO: change hardcoding 23 for BN case
        if self.batch_norm is False:
            idx_until_conv4_3 = 23
        elif self.batch_norm is True:
            idx_until_conv4_3 = 33
        for k in range(idx_until_conv4_3):
            x = self.vgg[k](x)

        if self.use_self_attention_base:
            x, attn_g, attnb = self.self_attn_base_list[sa_base_counter](x, True)
            if visualize:
                all_attnb.append(attnb)
            sa_base_counter += 1

        if self.dcn_cat_sab:
            if self.detach_sab:
                x = self.slice_and_cat(x, attn_g.detach())
            else:
                x = self.slice_and_cat(x, attn_g)

        if self.use_dcn:
            for i_dcn in range(self.num_dcn_layers):
                x, offset = self.dcn_list[dcn_counter](x)
                if visualize:
                    all_offset.append(offset)
                dcn_counter += 1

        # TODO: l2normed x or just x?
        s = self.L2Norm(x)
        #s = x

        if self.use_self_attention:
            s, attn_g, attn = self.self_attn_list[sa_counter](s, True)
            if visualize:
                all_attn.append(attn)
            sa_counter += 1

        if self.use_fuseconv:
            if self.batch_norm:
                s = F.relu(self.bn_fuse_11(self.fuse_11(s)), inplace=True)
                # s = F.relu(self.bn_fuse_12(self.fuse_12(s)), inplace=True)
            else:
                s = F.relu(self.fuse_11(s), inplace=True)
                # s = F.relu(self.fuse_12(s), inplace=True)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(idx_until_conv4_3, len(self.vgg)):
            x = self.vgg[k](x)

        if self.use_self_attention_base:
            x, attn_g, attnb = self.self_attn_base_list[sa_base_counter](x, True)
            if visualize:
                all_attnb.append(attnb)
            sa_base_counter += 1

        s2 = x

        if self.use_self_attention:
            s2, attn_g, attn = self.self_attn_list[sa_counter](s2, True)
            if visualize:
                all_attn.append(attn)
            sa_counter += 1

        if self.use_fuseconv:
            if self.batch_norm:
                s2 = F.relu(self.bn_fuse_21(self.fuse_21(s2)), inplace=True)
                # s2 = F.relu(self.bn_fuse_22(self.fuse_22(s2)), inplace=True)
            else:
                s2 = F.relu(self.fuse_21(s2), inplace=True)
                # s2 = F.relu(self.fuse_22(s2), inplace=True)

        sources.append(s2)

        # apply extra layers and cache source layer outputs
        # hard-coded for BN case
        if self.batch_norm is False:
            fuse_counter = 0
            for k, v in enumerate(self.extras):
                x = F.relu(v(x), inplace=True)
                if k % 2 == 1:
                    if self.use_self_attention_base:
                        x, attn_g, attnb = self.self_attn_base_list[sa_base_counter](x, True)
                        if visualize:
                            all_attnb.append(attnb)
                        sa_base_counter += 1
                    s_extra = x
                    if self.use_self_attention:
                        s_extra, attn_g, attn = self.self_attn_list[sa_counter](s_extra, True)
                        if visualize:
                            all_attn.append(attn)
                        sa_counter += 1
                    if self.use_fuseconv:
                        s_extra = F.relu(self.fuse_list1[fuse_counter](s_extra), inplace=True)
                        # s_extra = F.relu(self.fuse_list2[fuse_counter](s_extra), inplace=True)
                        fuse_counter += 1
                    sources.append(s_extra)
        elif self.batch_norm is True:
            fuse_counter = 0
            for k, v in enumerate(self.extras):
                x = v(x)
                if k % 2 == 1:
                    x = F.relu(x, inplace=True)
                if k % 4 == 3:
                    if self.use_self_attention_base:
                        x, attn_g, attnb = self.self_attn_base_list[sa_base_counter](x, True)
                        if visualize:
                            all_attnb.append(attnb)
                        sa_base_counter += 1
                    s_extra = x
                    if self.use_self_attention:
                        s_extra, attn_g, attn = self.self_attn_list[sa_counter](s_extra, True)
                        if visualize:
                            all_attn.append(attn)
                        sa_counter += 1
                    if self.use_fuseconv:
                        s_extra = F.relu(self.bn_fuse_list1[fuse_counter](self.fuse_list1[fuse_counter](s_extra)), inplace=True)
                        # s_extra = F.relu(self.bn_fuse_list2[fuse_counter](self.fuse_list2[fuse_counter](s_extra)), inplace=True)
                        fuse_counter += 1
                    sources.append(s_extra)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            # PyTorch1.5.0 support new-style autograd function
            # output = self.detect(
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                # PyTorch1.5.0 support new-style autograd function
                loc.view(loc.size(0), -1, 4),  # loc preds
                # self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        if visualize:
            return output, all_offset, all_attnb, all_attn
        else:
            return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            weight_pretrained = torch.load(base_file, map_location=lambda storage, loc: storage)
            # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/32
            from collections import OrderedDict
            pretrained_dict = OrderedDict()
            model_dict = self.state_dict()

            for k, v in weight_pretrained.items():
                if k.startswith("module."):  # DataParallel case
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                pretrained_dict[name] = v
                if pretrained_dict[name].shape != model_dict[name].shape:
                    print(
                        "WARNING: shape of pretrained {} {} does not match the current model {}. this weight will be ignored.".format(
                            name, pretrained_dict[name].shape, model_dict[name].shape))

            filtered_dict = {k: v for k, v in pretrained_dict.items() if
                             (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            # overwite model_dict entries in the existing filtered_dict and update it
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict)
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False, feature_scale=1, groups_vgg=4):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            # depthwise separable conv: add groups=4 (4 phases)
            conv2d = nn.Conv2d(in_channels, v * feature_scale, kernel_size=3, padding=1, groups=groups_vgg)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v * feature_scale), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v * feature_scale
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512*feature_scale, 1024*feature_scale, kernel_size=3, padding=6, dilation=6, groups=groups_vgg)
    conv7 = nn.Conv2d(1024*feature_scale, 1024*feature_scale, kernel_size=1, groups=groups_vgg)
    if batch_norm:
        layers += [pool5,
                   conv6, nn.BatchNorm2d(1024*feature_scale), nn.ReLU(inplace=True),
                   conv7, nn.BatchNorm2d(1024*feature_scale), nn.ReLU(inplace=True)]
    else:
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False, feature_scale=1, groups_extra=4):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if batch_norm is False:
                    layers += [nn.Conv2d(in_channels, (cfg[k + 1]) * feature_scale,
                           kernel_size=(1, 3)[flag], stride=2, padding=1, groups=groups_extra)]
                else:
                    layers += [nn.Conv2d(in_channels, (cfg[k + 1]) * feature_scale,
                                         kernel_size=(1, 3)[flag], stride=2, padding=1, groups=groups_extra),
                               nn.BatchNorm2d((cfg[k + 1]) * feature_scale)]

            else:
                if batch_norm is False:
                    layers += [nn.Conv2d(in_channels, v * feature_scale, kernel_size=(1, 3)[flag], groups=groups_extra)]
                else:
                    layers += [nn.Conv2d(in_channels, v * feature_scale, kernel_size=(1, 3)[flag], groups=groups_extra),
                               nn.BatchNorm2d(v * feature_scale)]
            flag = not flag
        if v == 'S':
            in_channels = v
        else:
            in_channels = v * feature_scale
    return layers


def multibox(vgg, extra_layers, cfg, num_classes, batch_norm):
    loc_layers = []
    conf_layers = []
    # hard-coded
    # TODO: make this generic
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


def build_ssd(phase, size=300, num_classes=21, batch_norm=False,
              groups_vgg=4, groups_extra=4, feature_scale=1, use_fuseconv=True, use_self_attention=False, use_self_attention_base=False, num_dcn_layers=0, groups_dcn=1, dcn_cat_sab=False, detach_sab=False,
              max_pool_factor=1):
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300:
        print("Error: Sorry only SSD300 is supported currently!")
        return

    # change the input channel from i=3 to 12
    return SSD(phase, *multibox(vgg(base[str(size)], i=12, batch_norm=batch_norm, feature_scale=feature_scale, groups_vgg=groups_vgg),
                                add_extras(extras[str(size)], 1024 * feature_scale, batch_norm, feature_scale, groups_extra),
                                mbox[str(size)], num_classes, batch_norm),
               num_classes, batch_norm, groups_vgg, groups_extra, feature_scale, use_fuseconv, use_self_attention, use_self_attention_base, num_dcn_layers, groups_dcn, dcn_cat_sab, detach_sab,
               max_pool_factor)
