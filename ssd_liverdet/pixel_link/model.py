import torch.nn as nn
import pixel_link.pixel_link_config as config
import numpy as np
from pixel_link.pixel_link_decode import *
from layers.dcn_v2_custom import DCN
from layers import self_attn
import os
import torch
from torch.utils.checkpoint import checkpoint

def xavier(param):
    nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

class PixelLink(nn.Module):
    def __init__(self, cascade_fuse, use_fuseconv, batch_norm, use_self_attention, use_self_attention_base, num_dcn_layers, groups_dcn, dcn_cat_sab, detach_sab,
                 max_pool_factor=1):
        super(PixelLink, self).__init__()

        self.vgg_groups = config.vgg_groups
        self.scale = config.feature_scale
        self.cascade_fuse = cascade_fuse
        self.use_self_attention = use_self_attention
        self.use_self_attention_base = use_self_attention_base
        self.num_dcn_layers = num_dcn_layers
        self.use_fuseconv = use_fuseconv
        self.batch_norm = batch_norm

        # TODO: modify padding
        self.conv1_1 = nn.Conv2d(12, int(64*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(int(64*self.scale), int(64*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(int(64*self.scale), int(128*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(int(128*self.scale), int(128*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(int(128*self.scale), int(256*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(int(256*self.scale), int(256*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(int(256*self.scale), int(256*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(int(256*self.scale), int(512*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(int(512*self.scale), int(512*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(int(512*self.scale), int(512*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(int(512*self.scale), int(512*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(int(512*self.scale), int(512*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(int(512*self.scale), int(512*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=[3, 3], stride=1, padding=1, ceil_mode=True)
        if config.dilation:
            self.conv6 = nn.Conv2d(int(512*self.scale), int(1024*self.scale), 3, stride=1, padding=6, dilation=6, groups=self.vgg_groups)
        else:
            self.conv6 = nn.Conv2d(int(512*self.scale), int(1024*self.scale), 3, stride=1, padding=1, groups=self.vgg_groups)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(int(1024*self.scale), int(1024*self.scale), 1, stride=1, padding=0, groups=self.vgg_groups)
        self.relu7 = nn.ReLU()

        self.modules_except_dcn = nn.ModuleList([self.conv1_1, self.relu1_1, self.conv1_2, self.relu1_2, self.pool1,
                                                 self.conv2_1, self.relu2_1, self.conv2_2, self.relu2_2, self.pool2,
                                                 self.conv3_1, self.relu3_1, self.conv3_2, self.relu3_2, self.conv3_3, self.relu3_3, self.pool3,
                                                 self.conv4_1, self.relu4_1, self.conv4_2, self.relu4_2, self.conv4_3, self.relu4_3, self.pool4,
                                                 self.conv5_1, self.relu5_1, self.conv5_2, self.relu5_2, self.conv5_3, self.relu5_3, self.pool5,
                                                 self.conv6, self.relu6, self.conv7, self.relu7])

        if config.version == "2s":
            self.out1_1 = nn.Conv2d(int(128*self.scale), 2, 1, stride=1, padding=0) #conv2_2
            self.out1_2 = nn.Conv2d(int(128*self.scale), 16, 1, stride=1, padding=0)
            self.modules_except_dcn.extend([self.out1_1, self.out1_2])
        self.out2_1 = nn.Conv2d(int(256*self.scale), 2, 1, stride=1, padding=0) #conv3_3
        self.out2_2 = nn.Conv2d(int(256*self.scale), 16, 1, stride=1, padding=0)
        self.out3_1 = nn.Conv2d(int(512*self.scale), 2, 1, stride=1, padding=0) #conv4_3
        self.out3_2 = nn.Conv2d(int(512*self.scale), 16, 1, stride=1, padding=0)
        self.out4_1 = nn.Conv2d(int(512*self.scale), 2, 1, stride=1, padding=0) #conv5_3
        self.out4_2 = nn.Conv2d(int(512*self.scale), 16, 1, stride=1, padding=0)
        self.out5_1 = nn.Conv2d(int(1024*self.scale), 2, 1, stride=1, padding=0) #fc7
        self.out5_2 = nn.Conv2d(int(1024*self.scale), 16, 1, stride=1, padding=0)
        self.modules_except_dcn.extend([self.out2_1, self.out2_2, self.out3_1, self.out3_2, self.out4_1, self.out4_2, self.out5_1, self.out5_2])

        if self.use_fuseconv:
            if config.version == "2s":
                self.fuse1 = nn.Conv2d(int(128*self.scale), int(128*self.scale), kernel_size=1)
                self.modules_except_dcn.append(self.fuse1)
            self.fuse2 = nn.Conv2d(int(256*self.scale), int(256*self.scale), kernel_size=1)
            self.fuse3 = nn.Conv2d(int(512*self.scale), int(512*self.scale), kernel_size=1)
            self.fuse4 = nn.Conv2d(int(512*self.scale), int(512*self.scale), kernel_size=1)
            self.fuse5 = nn.Conv2d(int(1024*self.scale), int(1024*self.scale), kernel_size=1)
            self.modules_except_dcn.extend([self.fuse2, self.fuse3, self.fuse4, self.fuse5])
            if batch_norm:
                if config.version == "2s":
                    self.bn_fuse1 = nn.BatchNorm2d(int(128*self.scale))
                    self.modules_except_dcn.append(self.bn_fuse1)
                self.bn_fuse2 = nn.BatchNorm2d(int(256*self.scale))
                self.bn_fuse3 = nn.BatchNorm2d(int(512*self.scale))
                self.bn_fuse4 = nn.BatchNorm2d(int(512*self.scale))
                self.bn_fuse5 = nn.BatchNorm2d(int(1024*self.scale))
                self.modules_except_dcn.extend([self.bn_fuse2, self.bn_fuse3, self.bn_fuse4, self.bn_fuse5])

        if self.cascade_fuse:
            if config.version == "2s":
                self.final_1 = nn.Conv2d(2 * 5, 2, 1, stride=1, padding=0)
                self.final_2 = nn.Conv2d(16 * 5, 16, 1, stride=1, padding=0)
            else:
                self.final_1 = nn.Conv2d(2 * 4, 2, 1, stride=1, padding=0)
                self.final_2 = nn.Conv2d(16 * 4, 16, 1, stride=1, padding=0)
        else:
            self.final_1 = nn.Conv2d(2, 2, 1, stride=1, padding=0)
            self.final_2 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.modules_except_dcn.extend([self.final_1, self.final_2])

        # new: try adjusting maxpool factor
        self.max_pool_factor = max_pool_factor

        if self.use_self_attention_base:
            self.self_attn_base_list = nn.ModuleList([])
            self.self_attn_base_in_channel_list = [256, 512, 512, 1024]
            if config.version == "2s":
                self.self_attn_base_in_channel_list = [128] + self.self_attn_base_in_channel_list
            self.self_attn_base_in_channel_list = [int(i * self.scale) for i in self.self_attn_base_in_channel_list]
            for i in range(len(self.self_attn_base_in_channel_list)):
                self.self_attn_base_list.append(self_attn.Self_Attn(in_channels=self.self_attn_base_in_channel_list[i],
                                                                    max_pool_factor=max_pool_factor))
        if self.use_self_attention:
            self.self_attn_list = nn.ModuleList([])
            self.self_attn_in_channel_list = [256, 512, 512, 1024]
            if config.version == "2s":
                self.self_attn_in_channel_list = [128] + self.self_attn_in_channel_list
            self.self_attn_in_channel_list = [int(i * self.scale) for i in self.self_attn_in_channel_list]
            for i in range(len(self.self_attn_in_channel_list)):
                self.self_attn_list.append(self_attn.Self_Attn(in_channels=self.self_attn_in_channel_list[i],
                                                               max_pool_factor=max_pool_factor))

        if self.num_dcn_layers > 0:
            self.use_dcn = True
            self.groups_dcn = groups_dcn
            self.dcn_list = nn.ModuleList([])
            # we try to match the offset of each phase after vgg conv7 stage
            self.dcn_in_channel_list = [256]
            self.dcn_in_channel_list = [int(i * self.scale) for i in self.dcn_in_channel_list]
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

        for m in self.modules():
            weights_init(m)

    def slice_and_cat(self, a, b):
        # slice each tensor by 4 (4-phase), concat a & b for each phase, then merge together
        # why?: we want to keep the "grouped" context of base convnet feature before feeding to next grouped conv
        a = torch.split(a, int(a.size(1)/self.vgg_groups), dim=1)
        b = torch.split(b, int(b.size(1)/self.vgg_groups), dim=1)
        ab = [torch.cat([a[i], b[i]], dim=1) for i in range(len(a))]
        ab = torch.cat(ab, dim=1)
        return ab

    def forward(self, x):

        if self.use_self_attention_base:
            sa_base_counter = 0
        if self.use_self_attention:
            sa_counter = 0
        # print("forward1")
        x = self.pool1(self.relu1_2(self.conv1_2(self.relu1_1(self.conv1_1(x)))))
        # print("forward11")
        x = self.relu2_2(self.conv2_2(self.relu2_1(self.conv2_1(x))))
        # print("forward12")
        if config.version == "2s":
            if self.use_self_attention_base:
                # x, attn_g, attnb = self.self_attn_base_list[sa_base_counter](x)
                x, attn_g = checkpoint(self.self_attn_base_list[sa_base_counter], x)
                sa_base_counter += 1
            if self.dcn_cat_sab:
                if self.detach_sab:
                    x = self.slice_and_cat(x, attn_g.detach())
                else:
                    x = self.slice_and_cat(x, attn_g)
            if self.use_dcn:
                for i_dcn in range(self.num_dcn_layers):
                    x, offset = self.dcn_list[i_dcn](x)
            s1 = x
            if self.use_self_attention:
                # s1, attn_g, attn = self.self_attn_list[sa_counter](s1)
                s1, attn_g, = checkpoint(self.self_attn_list[sa_counter], s1)
                sa_counter += 1
            if self.use_fuseconv:
                s1 = self.fuse1(s1)
                if self.batch_norm:
                    s1 = self.bn_fuse1(s1)
            l1_1x = self.out1_1(s1) #conv2_2
            # print("forward13")
            l1_2x = self.out1_2(s1) #conv2_2
            # print("forward14")
        x = self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(self.relu3_1(self.conv3_1(self.pool2(x)))))))
        # print("forward15")
        if self.use_self_attention_base:
            # x, attn_g, attnb = self.self_attn_base_list[sa_base_counter](x)
            x, attn_g = checkpoint(self.self_attn_base_list[sa_base_counter], x)
            sa_base_counter += 1
        if config.version != "2s" and self.use_dcn:
            if self.dcn_cat_sab:
                if self.detach_sab:
                    x = self.slice_and_cat(x, attn_g.detach())
                else:
                    x = self.slice_and_cat(x, attn_g)
            for i_dcn in range(self.num_dcn_layers):
                x, offset = self.dcn_list[i_dcn](x)
        s2 = x
        if self.use_self_attention:
            # s2, attn_g, attn = self.self_attn_list[sa_counter](s2)
            s2, attn_g = checkpoint(self.self_attn_list[sa_counter], s2)
            sa_counter += 1
        if self.use_fuseconv:
            s2 = self.fuse2(s2)
            if self.batch_norm:
                s2 = self.bn_fuse2(s2)
        l2_1x = self.out2_1(s2) #conv3_3
        # print("forward16")
        l2_2x = self.out2_2(s2) #conv3_3
        # print("forward17")

        x = self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(self.relu4_1(self.conv4_1(self.pool3(x)))))))
        if self.use_self_attention_base:
            # x, attn_g, attnb = self.self_attn_base_list[sa_base_counter](x)
            x, attn_g = checkpoint(self.self_attn_base_list[sa_base_counter], x)
            sa_base_counter += 1
        s3 = x
        if self.use_self_attention:
            # s3, attn_g, attn = self.self_attn_list[sa_counter](s3)
            s3, attn_g = checkpoint(self.self_attn_list[sa_counter], s3)
            sa_counter += 1
        if self.use_fuseconv:
            s3 = self.fuse3(s3)
            if self.batch_norm:
                s3 = self.bn_fuse3(s3)
        l3_1x = self.out3_1(s3) #conv4_3
        l3_2x = self.out3_2(s3) #conv4_3

        x = self.relu5_3(self.conv5_3(self.relu5_2(self.conv5_2(self.relu5_1(self.conv5_1(self.pool4(x)))))))
        if self.use_self_attention_base:
            # x, attn_g, attnb = self.self_attn_base_list[sa_base_counter](x)
            x, attn_g = checkpoint(self.self_attn_base_list[sa_base_counter], x)
            sa_base_counter += 1
        s4 = x
        if self.use_self_attention:
            # s4, attn_g, attn = self.self_attn_list[sa_counter](s4)
            s4, attn_g = checkpoint(self.self_attn_list[sa_counter], s4)
            sa_counter += 1
        if self.use_fuseconv:
            s4 = self.fuse4(s4)
            if self.batch_norm:
                s4 = self.bn_fuse4(s4)
        l4_1x = self.out4_1(s4) #conv5_3
        l4_2x = self.out4_2(s4) #conv5_3

        x = self.relu7(self.conv7(self.relu6(self.conv6(self.pool5(x)))))
        if self.use_self_attention_base:
            # x, attn_g, attnb = self.self_attn_base_list[sa_base_counter](x)
            x, attn_g = checkpoint(self.self_attn_base_list[sa_base_counter], x)
            sa_base_counter += 1
        s5 = x
        if self.use_self_attention:
            # s5, attn_g, attn = self.self_attn_list[sa_counter](s5)
            s5, attn_g = checkpoint(self.self_attn_list[sa_counter], s5)
            sa_counter += 1
        if self.use_fuseconv:
            s5 = self.fuse5(s5)
            if self.batch_norm:
                s5 = self.bn_fuse5(s5)
        l5_1x = self.out5_1(s5) #fc7
        l5_2x = self.out5_2(s5) #fc7
        # print("forward3")

        if self.cascade_fuse:
            upsample1_1 = nn.functional.interpolate(l5_1x + l4_1x, size=l3_1x.size()[2:], mode="bilinear",
                                                    align_corners=True)
            upsample2_1 = nn.functional.interpolate(upsample1_1 + l3_1x, size=l2_1x.size()[2:], mode="bilinear",
                                                    align_corners=True)
            if config.version == "2s":
                upsample3_1 = nn.functional.interpolate(upsample2_1 + l2_1x, size=l1_1x.size()[2:], mode="bilinear",
                                                        align_corners=True)
                logit_1 = upsample3_1 + l1_1x
                features = [nn.functional.interpolate(l5_1x, size=logit_1.size()[2:], mode="bilinear", align_corners=True),
                            nn.functional.interpolate(l5_1x + l4_1x, size=logit_1.size()[2:], mode="bilinear", align_corners=True),
                            nn.functional.interpolate(upsample1_1 + l3_1x, size=logit_1.size()[2:], mode="bilinear", align_corners=True),
                            nn.functional.interpolate(upsample2_1 + l2_1x, size=logit_1.size()[2:], mode="bilinear", align_corners=True),
                            logit_1]
                out_1 = self.final_1(torch.cat(features, dim=1))
            else:
                logit_1 = upsample2_1 + l2_1x
                features_1 = [nn.functional.interpolate(l5_1x, size=logit_1.size()[2:], mode="bilinear", align_corners=True),
                            nn.functional.interpolate(l5_1x + l4_1x, size=logit_1.size()[2:], mode="bilinear", align_corners=True),
                            nn.functional.interpolate(upsample1_1 + l3_1x, size=logit_1.size()[2:], mode="bilinear", align_corners=True),
                            logit_1]
                out_1 = self.final_1(torch.cat(features_1, dim=1))

            upsample1_2 = nn.functional.interpolate(l5_2x + l4_2x, size=l3_2x.size()[2:], mode="bilinear",
                                                    align_corners=True)
            upsample2_2 = nn.functional.interpolate(upsample1_2 + l3_2x, size=l2_2x.size()[2:], mode="bilinear",
                                                    align_corners=True)
            if config.version == "2s":
                upsample3_2 = nn.functional.interpolate(upsample2_2 + l2_2x, size=l1_1x.size()[2:], mode="bilinear",
                                                        align_corners=True)
                logit_2 = upsample3_2 + l1_2x
                features_2 = [
                    nn.functional.interpolate(l5_2x, size=logit_2.size()[2:], mode="bilinear", align_corners=True),
                    nn.functional.interpolate(l5_2x + l4_2x, size=logit_2.size()[2:], mode="bilinear",
                                              align_corners=True),
                    nn.functional.interpolate(upsample1_2 + l3_2x, size=logit_2.size()[2:], mode="bilinear",
                                              align_corners=True),
                    nn.functional.interpolate(upsample2_2 + l2_2x, size=logit_2.size()[2:], mode="bilinear",
                                              align_corners=True),
                    logit_2]
                out_2 = self.final_2(torch.cat(features_2, dim=1))
            else:
                logit_2 = upsample2_2 + l2_2x
                features_2 = [
                    nn.functional.interpolate(l5_2x, size=logit_2.size()[2:], mode="bilinear", align_corners=True),
                    nn.functional.interpolate(l5_2x + l4_2x, size=logit_2.size()[2:], mode="bilinear",
                                              align_corners=True),
                    nn.functional.interpolate(upsample1_2 + l3_2x, size=logit_2.size()[2:], mode="bilinear",
                                              align_corners=True),
                    logit_2]
                out_2 = self.final_2(torch.cat(features_2, dim=1))
        else:
            # upsample1_1 = nn.functional.upsample(l5_1x + l4_1x, scale_factor=2, mode="bilinear", align_corners=True)
            upsample1_1 = nn.functional.interpolate(l5_1x + l4_1x, size=l3_1x.size()[2:], mode="bilinear", align_corners=True)
            #upsample2_1 = nn.functional.upsample(upsample1_1 + l3_1x, scale_factor=2, mode="bilinear", align_corners=True)
            upsample2_1 = nn.functional.interpolate(upsample1_1 + l3_1x, size=l2_1x.size()[2:], mode="bilinear", align_corners=True)
            if config.version == "2s":
                # upsample3_1 = nn.functional.upsample(upsample2_1 + l2_1x, scale_factor=2, mode="bilinear", align_corners=True)
                upsample3_1 = nn.functional.interpolate(upsample2_1 + l2_1x, size=l1_1x.size()[2:], mode="bilinear", align_corners=True)
                out_1 = upsample3_1 + l1_1x
            else:
                out_1 = upsample2_1 + l2_1x
            out_1 = self.final_1(out_1)
            # print("forward4")

            # upsample1_2 = nn.functional.upsample(l5_2x + l4_2x, scale_factor=2, mode="bilinear", align_corners=True)
            upsample1_2 = nn.functional.interpolate(l5_2x + l4_2x, size=l3_2x.size()[2:], mode="bilinear", align_corners=True)
            # upsample2_2 = nn.functional.upsample(upsample1_2 + l3_2x, scale_factor=2, mode="bilinear", align_corners=True)
            upsample2_2 = nn.functional.interpolate(upsample1_2 + l3_2x, size=l2_2x.size()[2:], mode="bilinear", align_corners=True)
            if config.version == "2s":
                # upsample3_2 = nn.functional.upsample(upsample2_2 + l2_2x, scale_factor=2, mode="bilinear", align_corners=True)
                upsample3_2 = nn.functional.interpolate(upsample2_2 + l2_2x, size=l1_1x.size()[2:], mode="bilinear", align_corners=True)
                out_2 = upsample3_2 + l1_2x
            else:
                out_2 = upsample2_2 + l2_2x
            out_2 = self.final_2(out_2)
            # print("forward5")

        return [out_1, out_2] # [ [B, 2, H, W], [B, 16, H, W] ]

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

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
                if name in model_dict.keys():
                    if v.shape != model_dict[name].shape:
                        print(
                            "WARNING: shape of pretrained {} {} does not match the current model {}. this weight will be ignored.".format(
                                name, v.shape, model_dict[name].shape))
                    pretrained_dict[name] = v


            filtered_dict = {k: v for k, v in pretrained_dict.items() if
                             (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            # overwite model_dict entries in the existing filtered_dict and update it
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict)
        else:
            print('Sorry only .pth and .pkl files supported.')
