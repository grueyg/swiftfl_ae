"""Model for Fjord."""

import torch.nn.functional as F
from torch import nn
from torch.nn import Module

from torch.nn import ReLU
from torch.nn import Flatten
from torch.nn import MaxPool2d

from .od.models.utils import (
    SequentialWithSampler,
    create_bn_layer,
    create_conv_layer,
    create_linear_layer,
    create_lstm_layer
)
from .od.samplers import BaseSampler, ODSampler


class BasicBlock(nn.Module):
    """Basic Block for resnet."""

    expansion = 1

    def __init__(
        self, od, p_s, in_planes, planes, stride=1
    ):  # pylint: disable=too-many-arguments
        super().__init__()
        self.od = od
        self.conv1 = create_conv_layer(
            od,
            True,
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = create_bn_layer(od=od, p_s=p_s, num_features=planes)
        self.conv2 = create_conv_layer(
            od, True, planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = create_bn_layer(od=od, p_s=p_s, num_features=planes)

        self.shortcut = SequentialWithSampler()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = SequentialWithSampler(
                create_conv_layer(
                    od,
                    True,
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                create_bn_layer(od=od, p_s=p_s, num_features=self.expansion * planes),
            )

    def forward(self, x, sampler):
        """Forward method for basic block.

        Args:
        :param x: input
        :param sampler: sampler
        :return: Output of forward pass
        """
        if sampler is None:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
        else:
            out = F.relu(self.bn1(self.conv1(x, p=sampler())))
            out = self.bn2(self.conv2(out, p=sampler()))
            shortcut = self.shortcut(x, sampler=sampler)
            assert (
                shortcut.shape == out.shape
            ), f"Shortcut shape: {shortcut.shape} out.shape: {out.shape}"
            out += shortcut
            # out += self.shortcut(x, sampler=sampler)
            out = F.relu(out)
        return out


# Adapted from:
#   https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class ResNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """ResNet in PyTorch.

    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385
    """

    def __init__(
        self, od, p_s, block, num_blocks, num_classes=10
    ):  # pylint: disable=too-many-arguments
        super().__init__()
        self.od = od
        self.in_planes = 64

        self.conv1 = create_conv_layer(
            od, True, 3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = create_bn_layer(od=od, p_s=p_s, num_features=64)
        self.layer1 = self._make_layer(od, p_s, block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(od, p_s, block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(od, p_s, block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(od, p_s, block, 512, num_blocks[3], stride=2)
        self.linear = create_linear_layer(od, False, 512 * block.expansion, num_classes)

    def _make_layer(
        self, od, p_s, block, planes, num_blocks, stride
    ):  # pylint: disable=too-many-arguments
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(od, p_s, self.in_planes, planes, strd))
            self.in_planes = planes * block.expansion
        return SequentialWithSampler(*layers)

    def forward(self, x, sampler=None):
        """Forward method for ResNet.

        Args:
        :param x: input
        :param sampler: sampler
        :return: Output of forward pass
        """
        if self.od:
            if sampler is None:
                sampler = BaseSampler(self)
            out = F.relu(self.bn1(self.conv1(x, p=sampler())))
            out = self.layer1(out, sampler=sampler)
            out = self.layer2(out, sampler=sampler)
            out = self.layer3(out, sampler=sampler)
            out = self.layer4(out, sampler=sampler)
            out = F.avg_pool2d(out, 4)  # pylint: disable=not-callable
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)  # pylint: disable=not-callable
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out

class ConvNet2(Module):
    def __init__(self,
                 od,
                 p_s,
                 in_channels,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 use_bn=True,
                 dropout=.0):
        super(ConvNet2, self).__init__()
        self.od = od

        self.conv1 = create_conv_layer(od, True, in_channels=in_channels, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = create_conv_layer(od, True, in_channels=32, out_channels=64, kernel_size=5, padding=2)

        self.use_bn = use_bn
        if use_bn:
            self.bn1 = create_bn_layer(od=od, p_s=p_s, num_features=32)
            self.bn2 = create_bn_layer(od=od, p_s=p_s, num_features=64)

        self.fc1 = create_linear_layer(od, False, (h // 2 // 2) * (w // 2 // 2) * 64, hidden)
        self.fc2 = create_linear_layer(od, False, hidden, class_num)

        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(2)
        self.dropout = dropout

    def forward(self, x, sampler=None):
        if self.od:
            if sampler is None:
                sampler = BaseSampler(self)
            x = self.conv1(x, p=sampler())
            if self.use_bn:
                x = self.bn1(x)
            x = self.maxpool(self.relu(x))
            x = self.conv2(x, p=sampler())
            if self.use_bn:
                x = self.bn2(x)
            x = self.maxpool(self.relu(x))
            x = Flatten()(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc1(x)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc2(x)
        else:
            x = self.bn1(self.conv1(x)) if self.use_bn else self.conv1(x)
            x = self.maxpool(self.relu(x))
            x = self.bn2(self.conv2(x)) if self.use_bn else self.conv2(x)
            x = self.maxpool(self.relu(x))
            x = Flatten()(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.relu(self.fc1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc2(x)

        return x

class LSTM(Module):
    def __init__(self, od, p_s, 
                 in_channels,
                 hidden,
                 out_channels,
                 n_layers=1,
                 embed_size=8,
                 dropout=.0):
        super(LSTM, self).__init__()
        self.od = od

        self.in_channels = in_channels
        self.hidden = hidden
        self.embed_size = embed_size
        self.out_channels = out_channels
        self.n_layers = n_layers

        self.encoder = nn.Embedding(in_channels, embed_size)

        self.rnn1 = create_lstm_layer(od, 
                                     input_size=embed_size if embed_size else in_channels, 
                                     hidden_size=hidden, 
                                     num_layers=n_layers, 
                                     batch_first=True, 
                                     dropout=dropout)
        self.rnn2 = create_lstm_layer(od,
                                     input_size=hidden, 
                                     hidden_size=hidden, 
                                     num_layers=n_layers, 
                                     batch_first=True, 
                                     dropout=dropout)

        self.decoder = nn.Linear(hidden, out_channels)

    def forward(self, input_, sampler=None):
        if self.embed_size:
            input_ = self.encoder(input_)
        if self.od:
            output, hx = self.rnn1(input_, p=sampler())
            output, _ = self.rnn2(output, hx, p=sampler())
        else:
            output, _ = self.rnn1(input_)
            output, _ = self.rnn2(output, hx)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        final_word = output[:, :, -1]
        return final_word
    

def ResNet18(od=False, p_s=(1.0,)):
    """Construct a ResNet-18 model.

    Args:
    :param od: whether to create OD (Ordered Dropout) layer
    :param p_s: list of p-values
    """
    return ResNet(od, p_s, BasicBlock, [2, 2, 2, 2])



