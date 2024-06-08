"""
input_size=(473,473), PA=85.01, mPA=57.52, mIoU=46.92
"""
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable

affine_par = True
# import functools
# import sys, os
# from libs import InPlaceABN, InPlaceABNSync
# BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

from torch.nn import BatchNorm2d as BatchNorm2d

def InPlaceABNSync(in_channel):
    layers = [
        BatchNorm2d(in_channel),
        nn.ReLU(),
    ]
    return nn.Sequential(*layers)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU,nn.AdaptiveAvgPool2d,nn.Softmax,nn.Dropout2d)):
            pass
        else:
            m.initialize()

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(inner_features))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            InPlaceABNSync(inner_features))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            InPlaceABNSync(inner_features))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            InPlaceABNSync(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle


class Edge_Module(nn.Module):

    def __init__(self, in_fea=64, mid_fea=256, out_fea=2):
        super(Edge_Module, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea, mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea, mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea, mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea)
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()

        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)

        return edge, edge_fea


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=3, padding=1, dilation=1,
                      bias=False),
            InPlaceABNSync(out_features),
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class Decoder_Module(nn.Module):

    def __init__(self, num_classes):
        super(Decoder_Module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(48)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
        )
        # self.RC1 = Residual_Covolution(512, 1024, num_classes)
        # self.RC2 = Residual_Covolution(512, 1024, num_classes)
        # self.RC3 = Residual_Covolution(512, 1024, num_classes)
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()

        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x

class SEMblock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(SEMblock, self).__init__()
        self.squeeze1 = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, dilation=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.convg = nn.Conv2d(128, 128, 1)
        self.sftmax = nn.Softmax(dim=1)
        self.convAB = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnAB = nn.BatchNorm2d(64)

    def forward(self, l, h):
        h = self.squeeze1(l) if h is None else self.squeeze1(l+h)
        l = self.squeeze2(l)

        l = torch.cat((l, h), 1)
        y = self.convg(self.gap(l))
        l = torch.mul(self.sftmax(y)*y.shape[1], l)
        l = F.relu(self.bnAB(self.convAB(l)), inplace=True)
        return l, h

    def initialize(self):
        weight_init(self)
        print("AFMblock module init")


class AFMblock(nn.Module):
    def __init__(self):
        super(AFMblock, self).__init__()

        self.convA1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bnA1 = nn.BatchNorm2d(64)

        self.convB1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bnB1 = nn.BatchNorm2d(64)

        self.convAB = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnAB = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.convg = nn.Conv2d(128, 128, 1)
        self.sftmax = nn.Softmax(dim=1)

    #  self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x, y):
        if x.size()[2:] != y.size()[2:]:
            y = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=True)
        fuze = torch.mul(x, y)
        y = F.relu(self.bnB1(self.convB1(fuze + y)), inplace=True)
        x = F.relu(self.bnA1(self.convA1(fuze + x)), inplace=True)
        x = torch.cat((x, y), 1)
        y = self.convg(self.gap(x))
        x = torch.mul(self.sftmax(y) * y.shape[1], x)
        #   x = self.dropout(x)
        x = F.relu(self.bnAB(self.convAB(x)), inplace=True)
        return x

    def initialize(self):
        weight_init(self)
        print("Yblock module init")

class AggregationModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(AggregationModule, self).__init__()
        assert isinstance(kernel_size, int), "kernel_size must be a Integer"
        padding = kernel_size // 2

        # self.layer1 = nn.ConvBnReLU2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # BN层输入特征矩阵深度为out_channel
            nn.ReLU(inplace=True)
        )

        self.fsconvl = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(1, kernel_size),
                      padding=(0, padding),
                      groups=out_channels),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(kernel_size, 1),
                      padding=(padding, 0),
                      groups=out_channels)
        )
        self.fsconvr = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(kernel_size, 1),
                      padding=(padding, 0),
                      groups=out_channels),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(1, kernel_size),
                      padding=(0, padding),
                      groups=out_channels)
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        xl = self.fsconvl(x)
        xr = self.fsconvr(x)
        y = self.layer2(xl + xr)
        return y

class CPMoudle(nn.Module):

    def __init__(self, fea_channels=2048, reduce_channels=256, am_kernel_size=11):
        super(CPMoudle, self).__init__()

        self.aggregation = AggregationModule(fea_channels, reduce_channels, am_kernel_size)
        """
        self.prior_conv = nn.Sequential(
            nn.Conv2d(in_channels=reduce_channels,
                    out_channels=np.prod(self.prior_size),
                    kernel_size=1,
                    groups=groups),
            nn.BatchNorm2d(num_features=np.prod(self.prior_size))
        )
        """

        self.intra_conv = nn.Sequential(
            nn.Conv2d(reduce_channels, reduce_channels, kernel_size=1, padding=0,stride=1),
            nn.BatchNorm2d(reduce_channels),  # BN层输入特征矩阵深度为out_channel
            nn.ReLU(inplace=True)
        )

        self.inter_conv = nn.Sequential(
            nn.Conv2d(reduce_channels, reduce_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(reduce_channels),  # BN层输入特征矩阵深度为out_channel
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(reduce_channels*2+fea_channels, reduce_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduce_channels),  # BN层输入特征矩阵深度为out_channel
            nn.ReLU(inplace=True)#输入通道准确的写应该是feat23+reduce_channels*2
        #nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
        #nn.ReLU(inplace=True),
        #nn.Dropout2d(p=0.5)
        )

    def forward(self, feat):

        cosine_eps = 1e-7

        feat_agg = self.aggregation(feat)  # 大小通道和输入feature4相同
        #feat_agg = feat
        batch_size, channels_size, spatial_size, _ = feat_agg.size()[:]  # 经过融合模块的feature大小，和feat_4相同
        #print(batch_size,channels_size,spatial_size)  # 16,256,64

        # 生成context prior map，这里考虑余弦相似度求prior
        tmp_query1 = feat_agg
        tmp_query1 = tmp_query1.contiguous().view(batch_size, channels_size, -1)  # c*hw 16,256,2048
        tmp_query1_norm = torch.norm(tmp_query1, 2, 1, True)

        tmp_query2 = feat_agg
        tmp_query2 = tmp_query2.contiguous().view(batch_size, channels_size, -1)
        tmp_query2 = tmp_query2.contiguous().permute(0, 2, 1)  # hw*c 16,2048,256
        tmp_query2_norm = torch.norm(tmp_query2, 2, 2, True)

        similarity = torch.bmm(tmp_query2, tmp_query1) / (torch.bmm(tmp_query2_norm, tmp_query1_norm) + cosine_eps)  # 大小为hw*hw
        #similarity = similarity.max(1)[0].view(batch_size, spatial_size * spatial_size)
        #similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        #corr_query = similarity.view(batch_size, 1, spatial_size, spatial_size)
        #corr_query = F.interpolate(corr_query, size=(self.prior_size, self.prior_size), mode='bilinear',align_corners=True)

        #需不需要把prior图转换为二值图
        context_prior_map = similarity  #hw*hw  2048*2048
        """
        context_prior_map = self.prior_conv(xt)#调整xt的channel数
        context_prior_map = context_prior_map.reshape((batch_size, np.prod(self.prior_size), -1))
        context_prior_map = context_prior_map.transpose((0, 2, 1))
        context_prior_map = F.sigmoid(context_prior_map)
        """
        #feat_agg的reshape过程（ reshape to BxCxN -> BxNxC）

        feat_agg = feat_agg.contiguous().view(batch_size, channels_size, -1)
        feat_agg = feat_agg.contiguous().permute(0, 2, 1) #hw*c  2048,256

        # intra-class context
        intra_context = torch.bmm(context_prior_map, feat_agg) #hw*c 2048,256
        intra_context = intra_context.contiguous().permute(0, 2, 1)  #256,2048
        intra_context = intra_context.reshape((batch_size, channels_size, spatial_size, -1))#或者用.view()
        intra_context = self.intra_conv(intra_context)
        #intra_context = F.interpolate(intra_context, size=(feat1.size()[2], feat1.size()[3]), mode='bilinear',align_corners=True)

        # inter-class context
        inter_context_prior_map = 1 - context_prior_map
        inter_context = torch.bmm(inter_context_prior_map, feat_agg)#hw*c
        inter_context = inter_context.contiguous().permute(0, 2, 1)
        inter_context = inter_context.reshape((batch_size, channels_size, spatial_size, -1))
        inter_context = self.inter_conv(inter_context)
        #inter_context = F.interpolate(inter_context, size=(feat1.size()[2], feat1.size()[3]), mode='bilinear',align_corners=True)
        # concat
        concat_x = torch.cat([feat, intra_context, inter_context], 1)

        final_x = self.bottleneck(concat_x)

        return final_x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=False)
        return self.conv(p)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

        self.psp = PSPModule(2048, 512)
        self.decoder = Decoder_Module(num_classes)
        # self.dual_attn = DANetHead(256, 128)
        self.cpm = CPMoudle(256, 128, 11)

        self.squeeze3 = SEMblock(1024, 64)
        self.squeeze2 = SEMblock(512, 64)
        self.squeeze1 = SEMblock(256, 64)

        self.ca_conv3 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1)
        self.ca_bn3 = nn.BatchNorm2d(512)

        self.ca_conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.ca_bn2 = nn.BatchNorm2d(256)

        # self.conv_edge = nn.Conv2d(64, 2, kernel_size=1, padding=0, dilation=1, bias=True)
        #
        # self.afm3 = AFMblock()
        # self.afm2 = AFMblock()
        # self.afm1 = AFMblock()

        self.edge_layer = Edge_Module()

        self.up_1 = Upsample(896, 512)
        self.up_2 = Upsample(512, 256)
        self.drop = nn.Dropout2d(p=0.15)

        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x2 = self.layer1(x)  # 256
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.psp(x5)  # 512

        # edge, edge_fea = self.edge_layer(x2, x3, x4)
        seg1, x = self.decoder(x, x2)
        # x = self.dual_attn(x)
        x = self.cpm(x)  # 256

        f4, h = self.squeeze3(x4, None)
        h = F.interpolate(h, size=x3.size()[2:], mode='bilinear', align_corners=True)
        h = F.relu(self.ca_bn3(self.ca_conv3(h)))
        f3, h = self.squeeze2(x3, h)
        h = F.interpolate(h, size=x2.size()[2:], mode='bilinear', align_corners=True)
        h = F.relu(self.ca_bn2(self.ca_conv2(h)))
        f2, h = self.squeeze1(x2, h)
        edge, edge_fea = self.edge_layer(f2, f3, f4)

        # f34 = self.afm3(f3, f4)
        # f23 = self.afm2(f2, f3)
        # f1 = self.afm1(f23, f34)  # [16,64,64,32]
        #
        # edge_feat = self.conv_edge(f1)

        x = torch.cat([x, edge_fea], dim=1)
        x = self.drop(self.up_2(self.drop(self.up_1(x))))  # seg_feat:[64,256,128]
        seg2 = self.layer7(x)

        return [[seg1, seg2], [edge]]


def Res_Deeplab(num_classes=20):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model
