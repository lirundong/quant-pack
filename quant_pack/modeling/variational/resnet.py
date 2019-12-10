import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from ._components import NaiveQuantConv2d, NaiveQuantLinear, \
                         TernaryConv2d, TernaryLinear, NonLocal, \
                         QConv2dDiffBounds, QLinearDiffBounds
from ._quant_backbone import QuantBackbone

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def get_conv(in_channels, out_channels, kernel_size, stride=1, padding=0,
             dilation=1, groups=1, bias=True, quant_mode=None, **kwargs):
    if not quant_mode:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
    elif quant_mode.lower() == "ste":
        conv = NaiveQuantConv2d
    elif quant_mode.lower() == "prob":
        conv = TernaryConv2d
    elif quant_mode.lower() == "opt_bounds":
        conv = QConv2dDiffBounds
    else:
        raise ValueError(f"Invalid quant mode: `{quant_mode}`")

    return conv(in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, bias, **kwargs)


def get_linear(in_features, out_features, bias=True, quant_mode=None, **kwargs):
    if not quant_mode:
        return nn.Linear(in_features, out_features, bias)
    elif quant_mode.lower() == "ste":
        linear = NaiveQuantLinear
    elif quant_mode.lower() == "prob":
        linear = TernaryLinear
    elif quant_mode.lower() == "opt_bounds":
        linear = QLinearDiffBounds
    else:
        raise ValueError(f"Invalid quant mode: `{quant_mode}`")

    return linear(in_features, out_features, bias, **kwargs)


def conv1x1(in_planes, out_planes, stride=1, quant_mode=None, **kwargs):
    return get_conv(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
                    bias=False, quant_mode=quant_mode, **kwargs)


def conv3x3(in_planes, out_planes, stride=1, quant_mode=None, **kwargs):
    return get_conv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                    bias=False, quant_mode=quant_mode, **kwargs)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        quant_conf = kwargs.get("quant", dict())
        denoise_conf = kwargs.get("denoise", dict())
        quant_mode = kwargs.get("quant_mode", None)
        self.enable_denoise = kwargs.get("enable_denoise", False)
        if self.enable_denoise:
            self.denoiser = NonLocal(planes, **denoise_conf)

        self.conv1 = conv3x3(inplanes, planes, stride, quant_mode=quant_mode, **quant_conf)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, quant_mode=quant_mode, **quant_conf)
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

        if self.enable_denoise:
            out = self.denoiser(out)
        else:
            out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__()

        quant_conf = kwargs.get("quant", dict())
        denoise_conf = kwargs.get("denoise", dict())
        quant_mode = kwargs.get("quant_mode", None)
        self.enable_denoise = kwargs.get("enable_denoise", False)
        if self.enable_denoise:
            self.denoiser = NonLocal(planes * 4, **denoise_conf)

        self.conv1 = conv1x1(inplanes, planes, quant_mode=quant_mode, **quant_conf)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, quant_mode=quant_mode, **quant_conf)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4, quant_mode=quant_mode, **kwargs)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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

        out += residual

        if self.enable_denoise:
            out = self.denoiser(out)
        else:
            out = self.relu(out)

        return out


class ResNet(QuantBackbone):

    def __init__(self, block, layers, num_classes=1000, **kwargs):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.quant_mode = kwargs.get("quant_mode")
        self.quant_conv = self.quant_mode["conv"]
        self.quant_linear = self.quant_mode["linear"]
        self.quant_conf = kwargs.get("quant_conf", dict())
        self.denoise_mode = kwargs.get("denoise_mode")
        self.denoise_conf = kwargs.get("denoise_conf", dict())
        self.distillation = kwargs.get("distillation", False)

        self.conv1 = get_conv(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False, quant_mode=self.quant_conv, **self.quant_conf)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = get_linear(512 * block.expansion, num_classes,
                             quant_mode=self.quant_linear, **self.quant_conf)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride,
                        quant_mode=self.quant_conv, **self.quant_conf),
                nn.BatchNorm2d(planes * block.expansion),
            )

        block_conf = {
            "quant": self.quant_conf,
            "denoise": self.denoise_conf,
            "enable_denoise": bool(self.denoise_mode),
            "quant_mode": self.quant_conv,
        }
        layers = [block(self.inplanes, planes, stride, downsample, **block_conf)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, **block_conf))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p4 = self.avgpool(c4)
        p4 = p4.view(p4.size(0), -1)
        logits = self.fc(p4)

        if self.training and self.distillation:
            return c2, c3, c4, logits
        else:
            return logits


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
