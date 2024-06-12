#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:08:19 2020
@author: Ethel   School:ECNU   Email:52181214003@stu.ecnu.edu.cn
"""
from efficientnet_pytorch import model as enet
from torch.autograd import Variable
from functools import partial
import torch.nn as nn
import torch
from torch.utils import model_zoo
import torch.nn.functional as F
import math
import numpy as np
from torch.nn import Parameter
from collections import OrderedDict
from torchsummary import summary
import pretrainedmodels
import numpy
import os
import argparse
import matplotlib.pyplot as plt
sigmoid = torch.nn.Sigmoid()
import torchvision.models as models

spectral_path = "/home/ubuntu/Data1/zq/Project/TAJ-Net/data_curve"

def spectral_curve(batch_data): # batch_data:[16,40,224,224]
    curve = np.zeros((16,40,40), np.float32)
    for i in range(batch_data.shape[0]):
        for j in range(batch_data.shape[1]):
            for p in range(batch_data.shape[2]):
                curve[0,i,j] = batch_data[i,j,111,111]
    return curve

def spectral_parameter(spectral_path, x_numpy, trained_model):
    # print(trained_model)
    if trained_model == 'ConvNet1d':
        spectral_tumor = np.load(os.path.join(spectral_path, 'original_tumor.npy'))
        spectral_hyper = np.load(os.path.join(spectral_path, 'original_hyper.npy'))
        spectral_normal = np.load(os.path.join(spectral_path, 'original_normal.npy'))
    else:
        spectral_tumor = np.load(os.path.join(spectral_path, 'se_tumor.npy'))
        spectral_hyper = np.load(os.path.join(spectral_path, 'se_hyper.npy'))
        spectral_normal = np.load(os.path.join(spectral_path, 'se_normal.npy'))
    distance_tumor = np.zeros(x_numpy.shape[1], np.float32)
    distance_hyper = np.zeros(x_numpy.shape[1], np.float32)
    distance_normal = np.zeros(x_numpy.shape[1], np.float32)
    distance_mean = np.zeros(x_numpy.shape[0], np.float32)
    # print('x_numpy.shape: ', x_numpy.shape)
    # print('spectral_tumor.shape: ', spectral_tumor.shape)
    for j in range(x_numpy.shape[0]):
        for i in range(x_numpy.shape[1]):
            if trained_model == 'se_resnet_2d_20':
                distance_tumor[i] = abs(
                    x_numpy[j, i, 111, 111] - (spectral_tumor[2 * i] + spectral_tumor[2 * i + 1]) / 2)
                distance_hyper[i] = abs(
                    x_numpy[j, i, 111, 111] - (spectral_hyper[2 * i] + spectral_hyper[2 * i + 1]) / 2)
                distance_normal[i] = abs(
                    x_numpy[j, i, 111, 111] - (spectral_normal[2 * i] + spectral_hyper[2 * i + 1]) / 2)
            elif trained_model == 'ConvNet1d' or trained_model == 'ConvNet1d_se':  # ConvNet1d or ConvNet1d_se
                distance_tumor[i] = abs(x_numpy[j, i, :] - spectral_tumor[i])
                distance_hyper[i] = abs(x_numpy[j, i, :] - spectral_hyper[i])
                distance_normal[i] = abs(x_numpy[j, i, :] - spectral_normal[i])
            else:
                distance_tumor[i] = abs(x_numpy[j, i, 111, 111] - spectral_tumor[i])
                distance_hyper[i] = abs(x_numpy[j, i, 111, 111] - spectral_hyper[i])
                distance_normal[i] = abs(x_numpy[j, i, 111, 111] - spectral_normal[i])

        distance_mean_tumor = distance_tumor.mean()
        distance_mean_hyper = distance_hyper.mean()
        distance_mean_normal = distance_normal.mean()
        # print(distance_mean_tumor, distance_mean_hyper, distance_mean_tumor)
        distance_mean[j] = min([distance_mean_tumor, distance_mean_hyper, distance_mean_normal])
    distance_mean_batch = distance_mean.mean()
    return distance_mean_batch

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     dilation=2,
                     bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     dilation=2,
                     bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     dilation=2,
                     bias=False)
class BasicBlock_2d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

swish = Swish.apply

class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)
    
class DenseCrossEntropy(nn.Module):
    def forward(self, x, target, reduction='mean'):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss

class Flatten(nn.Module):
    def forward(self, x):
        N, C, L = x.size()  # read in N, C, L
        z = x.view(N, -1)
#        print(C, L)
        return z  # "flatten" the C * L values into a single vector per image


class ConvNet1d(nn.Module):
    def __init__(self, channels, reduction, conv1d_trained_model):
        super(ConvNet1d, self).__init__()
        self.se_module = SEModule(channels=channels, reduction=reduction)
        self.conv1 = nn.Conv1d(1, 8, 3)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.maxpool1 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fn = nn.Linear(16 * 18, 3)
        self.sigmoid = nn.Sigmoid()
        self.activate = nn.ReLU() # sigmoid, softmax, Tanh
        self.trained_model = conv1d_trained_model
    def forward(self, x):
        if True in torch.isnan(x):
            print('there is nan in input!')
        # print('original x: ', x.shape())
        if self.trained_model == 'ConvNet1d_se':
            x = self.se_module(x)

        # turn the image into spectral curve
        x_tensor = x
        x_numpy = x_tensor.cpu().detach().numpy()
        # print('x_numpy: ', x_numpy.size)
        x_numpy = x_numpy[:, :, 111, 111]
        x = torch.from_numpy(x_numpy).cuda()
        x = x.unsqueeze(1)

        # start: acquire spectral difference after senet
        x_tensor = x
        x_numpy = x_tensor.cpu().detach().numpy()
        x_numpy = x_numpy.transpose(0,2,1)
        distance_mean_batch = spectral_parameter(spectral_path=spectral_path, x_numpy=x_numpy, trained_model=self.trained_model)
        # distance_mean_batch = spectral_parameter(spectral_path=spectral_path, x_numpy=x_numpy, trained_model='ConvNet1d_se')
        # end: acquire spectral difference after senet

        x = self.conv1(x)
        if True in torch.isnan(x):
            print('there is nan in output of conv1!')
        #     print(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.flatten(x)
        x = self.fn(x)
        # x = self.activate(x)
        return x, distance_mean_batch

class ConvNet1d_old(nn.Module):
    def __init__(self, out_dim_1=3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(40, 11, kernel_size=3),# 40: channel
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10))
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Sequential(
            # nn.Linear(768,100),
            nn.Linear(2560,1000),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            # nn.Linear(100,6),
            nn.Linear(1000,out_dim_1),
            nn.Softmax())

    def forward(self, x):
        # x_tensor = x
        # x_numpy = x_tensor.cpu().detach().numpy()
        # curve = spectral_curve(x_numpy)
        # x = torch.from_numpy(curve)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,stride=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,stride=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print("x: ", x.shape)
        module_input = x
        # print("after se x: ", x.shape)
        # start: acquire spectral curve before senet
        x_tensor = module_input
        x_numpy = x_tensor.cpu().detach().numpy()
        spectral = np.zeros(x_numpy.shape[1], np.float32)
        spec_mean = np.zeros(x_numpy.shape[0], np.float32)
        for j in range(x_numpy.shape[0]):
            for i in range(x_numpy.shape[1]):
                spectral[i] = x_numpy[j, i, 111, 111]

        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
  
class SEModule_3D(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule_3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        print("avg_pool: {}".format(x.std()))
        x = self.fc1(x)
        print("fc1: {}".format(x.std()))
        x = self.relu(x)
        # print("relu: {}".format(x.std()))
        x = self.fc2(x)
        # print("fc2: {}".format(x.std()))
        x = self.sigmoid(x)
        # print("sigmoid: {}".format(x.std()))
        return module_input * x

class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=30.0, m=0.5, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.s = s
        self.m = m
        
        self.cos_m = math.cos(m)             #  0.87758
        self.sin_m = math.sin(m)             #  0.47943
        self.th = math.cos(math.pi - m)      # -0.87758
        self.mm = math.sin(math.pi - m) * m  #  0.23971
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        # logits = torch.from_numpy(logits)
        logits = logits.float()  # float16 to float32 (if used float16)
        # logits = logits.astype(np.float32)
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # equals to **2
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # print('one_hot:', one_hot)
        # exit()
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss = self.criterion(output,labels)
        if True in torch.isnan(loss):
            print('there is nan in the AFLoss!')
        return loss
    
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine

class enet_arcface(nn.Module):

    def __init__(self,out_dim_1=512, out_dim_2 = 2,backbone='efficientnet-b0'):
        super(enet_arcface, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        self.dropout = nn.Dropout(0.4)

        self.gfc = nn.Linear(self.enet._fc.out_features, out_dim_1)
        self.metric_classify = ArcMarginProduct(out_dim_1, out_dim_2)

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = Swish_module()(self.gfc(x))
        
#        out_2 = self.myfc_2_1(self.dropout(x))
#
#        out_2 = self.myfc_2_2(Swish_module()(out_2))
        metric_output = self.metric_classify(x)
        return metric_output

class se_resnet_3d(nn.Module):
    def __init__(self, block,
                 channels=40,
                 reduction=16,
                 layers = [3,4,6,3],
                 block_inplanes = [8,16,32,64], #[64,128,256,512]
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=3):
        super(se_resnet_3d, self).__init__()
        # self.se_module = SEModule_3D(n_input_channels, reduction=reduction)
        self.se_module = SEModule(channels=channels, reduction=reduction)
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.n_input_channels = n_input_channels

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(7, 7, 7), #(conv1_t_size, 7, 7)
                               stride=(2, 2, 2),#conv1_t_stride
                               padding=(3, 3, 3),# (conv1_t_size // 2, 3, 3)
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))# 1,1,1
        self.last_linear = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.sigmoid = nn.Sigmoid()
        self.metric_classify = ArcMarginProduct(self.last_linear.in_features, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.se_module(x)

        # start: acquire spectral difference after senet
        x_tensor = x
        x_numpy = x_tensor.cpu().detach().numpy()
        distance_mean_batch = spectral_parameter(spectral_path=spectral_path, x_numpy=x_numpy, trained_model='se_resnet_3d')
        # end: acquire spectral difference after senet

        x = x.unsqueeze(-4)
        x = x.repeat(1,self.n_input_channels,1,1,1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        metric_output = self.metric_classify(x)
        return metric_output, distance_mean_batch
        # x = x.view(x.size(0), -1)
        # x = self.last_linear(x)
        # x = self.sigmoid(x)
        # return x

class se_resnet_2d_noPre(nn.Module):
    def __init__(self, block,
                 channels=40,
                 reduction=16,
                 layers = [3,4,6,3],
                 block_inplanes = [8,16,32,64], #[64,128,256,512]
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=3):
        super(se_resnet_2d_noPre, self).__init__()
        # self.se_module = SEModule_3D(n_input_channels, reduction=reduction)
        self.se_module = SEModule(channels=channels, reduction=reduction)
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.n_input_channels = n_input_channels

        self.conv1 = nn.Conv2d(n_input_channels,
                               self.in_planes,
                               kernel_size=(7, 7), #(conv1_t_size, 7, 7)
                               stride=(2, 2),#conv1_t_stride
                               padding=(3, 3),# (conv1_t_size // 2, 3, 3)
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))# 1,1,1
        self.last_linear = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.sigmoid = nn.Sigmoid()
        self.metric_classify = ArcMarginProduct(self.last_linear.in_features, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool2d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.se_module(x)

        # start: acquire spectral curve after senet
        x_tensor = x
        x_numpy = x_tensor.cpu().detach().numpy()
        distance_mean_batch = spectral_parameter(spectral_path=spectral_path, x_numpy=x_numpy, trained_model='se_resnet_2d_noPre')
        # end: acquire spectral curve after senet

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        metric_output = self.metric_classify(x)
        return metric_output, distance_mean_batch

class se_resnet_2d(nn.Module):
    def __init__(self,out_dim_1=2,model_name='resnext50_32x4d',pretrained='imagenet',reduction=16,channels=40,inplanes=64):
        super(se_resnet_2d, self).__init__()
        self.se_module = SEModule(channels, reduction=reduction)
        layer0_modules = [
                ('conv1', nn.Conv2d(channels, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True)),
        ]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        
        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        # model = models.resnext50_32x4d(pretrained=None)
        # self.base_model = load_state_dict(torch.load('/home/zq/Documents/TSC-Net-2D-3D/se_resnext50_32x4d-a260b3a4.pth'))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        self.metric_classify = ArcMarginProduct(self.base_model.last_linear.in_features, out_dim_1)
    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
  
    def forward(self, x):
        x = self.se_module(x)

        # start: acquire spectral difference after senet
        x_tensor = x
        x_numpy = x_tensor.cpu().detach().numpy()
        # if True in np.isnan(x_numpy):
        #     print('there is nan in the array!!!!')
        if x_numpy.shape[1] == 20:
            distance_mean_batch = spectral_parameter(spectral_path=spectral_path, x_numpy=x_numpy, trained_model = 'se_resnet_2d_20')
        else:
            distance_mean_batch = spectral_parameter(spectral_path=spectral_path, x_numpy=x_numpy, trained_model = 'se_resnet_2d')
        # end: acquire spectral difference after senet

        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.size(0), -1)
        #x = Swish_module()(self.gfc(x))
        metric_output = self.metric_classify(x)
        return metric_output, distance_mean_batch
      
if __name__=="__main__":
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = senet_arcface(pretrained=None).to(device)#enet.EfficientNet.from_name('efficientnet-b0').to(device)
    
    #net = torch.nn.Sequential(*list(net.children())[:-1])
  
    for name,module in net.named_modules():
        print('children module:', name)
    #summary(net, (40, 224, 224),4) # 4-batch_size
