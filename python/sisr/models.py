"""PyTorch models for training and testing SiSR super-resolution network
"""

import logging
import torch

from torch import nn
from torchvision import models


logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """VGG feature extractor module
    """

    def __init__(self, variant='E', pretrained=True):
        super().__init__()

        # Extract torchvision feature modules, and download pretrained ImageNet
        # weights
        if variant == 'A':
            layers = list(models.vgg11(pretrained=pretrained).features)
        elif variant == 'B':
            layers = list(models.vgg13(pretrained=pretrained).features)
        elif variant == 'D':
            layers = list(models.vgg16(pretrained=pretrained).features)
        elif variant == 'E':
            layers = list(models.vgg19(pretrained=pretrained).features)
        elif isinstance(variant, str):
            raise ValueError("Unsupported VGG variant %s" % variant)
        else:
            raise TypeError("Argument variant should be specified as a string")
        self.variant = variant

        # Use TensorFlow VGG layer naming convention
        pool_idx = 1
        conv_idx = 1
        self.names = []
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                self.names.append('conv%d_%d' % (pool_idx, conv_idx))
                conv_idx++

            elif isinstance(layer, nn.ReLU):
                self.names.append('relu%d_%d' % (pool_idx, conv_idx))

            elif isinstance(layer, nn.MaxPool2d):
                self.names.append('pool%d' % (pool_idx))
                pool_idx++

        # Save modules in a list, making sure we do not use ReLU inplace
        self.features = nn.ModuleList(
            nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
            for l in layers)


    def forward(self, x, names=set()):

        # Make sure names are valid
        names = set(names)
        if not names.issubset(self.names):
            raise ValueError(
                "Unknown layer name %s" %', '.join(names - set(self.names)))

        # Make sure greyscale images have three channels
        x = x.expand(-1,3,-1,-1)

        # Extract named feature maps into a dictionary
        f = {}
        for name, feature in zip(self.names, self.features):

            # Stop once we've extracted all features
            if len(f) == len(names):
                break

            # Extract the feature
            x = feature(x)

            # Save features to dictionary if requested
            if name in names:
                f[name] = x

        return f


class ResidualBlock(nn.Module):
    """Residual learning block
    """

    def __init__(self,
            kernel_size=3,
            num_filters=64,
            weight_norm=True,
        ):
        super().__init__()

        conv1 = nn.Conv2d(num_filters, num_filters, kernel_size)
        relu1 = nn.ReLU(True)
        conv2 = nn.Conv2d(num_filters, num_filters, kernel_size)
        if weight_norm:
            conv1 = nn.utils.weight_norm(conv1)
            conv2 = nn.utils.weight_norm(conv2)
        self.body = nn.Sequential(conv1, relu1, conv2)

    def forward(self, x):
        x += self.body(x)
        return x


class Generator(nn.Module):
    """Generator network
    """

    def __init__(self,
            kernel_size=3,
            num_colours=1,
            num_filters=64,
            num_resblocks=8,
            scale_factor=4,
            weight_norm=True,
            upscaling='nearest'
        ):
        # Colour space to feature space
        conv = nn.Conv2d(num_colours, num_filters, kernel_size)
        if weight_norm:
            conv = nn.utils.weight_norm(conv)
        self.head = nn.Sequential(conv)

        # Residual body
        self.body = nn.Sequential(*[
            nn.ResidualBlock(num_filters=num_filters, weight_norm=weight_norm)
            for _ in range(num_resblocks)])

        # Upsample features, then feature space to colour space
        upsample = []
        while scale_factor > 1:

            for sf in 3, 2:
                if scale_factor % sf == 0:

                    if upscaling == 'nearest':
                        nearest = nn.Upsample(scale_factor=sf, mode='nearest')
                        conv = nn.Conv2d(num_filters, num_filters, kernel_size)
                        # Is this ReLU really needed
                        relu = nn.ReLU(True)
                        if weight_norm:
                            conv = nn.utils.weight_norm(conv)

                        upsample.append(nearest)
                        upsample.append(conv)
                        upsample.append(relu)

                    elif upscaling='shuffle':
                        conv = nn.Conv2d(
                            num_filters, num_filters*sf**2, kernel_size)
                        shuffle = nn.PixelShuffle(sf)
                        if weight_norm:
                            conv = nn.utils.weight_norm(conv)

                        upsample.append(conv)
                        upsample.append(shuffle)

                    scale_factor /= sf

        conv = nn.Conv2d(num_filters, num_colours, kernel_size)
        self.tail = nn.Sequential(*upsample, conv)


    def forward(self, x)
        x = self.head(x)
        x += self.body(x)
        x = self.tail(x)
        return x


def discriminator():
    """Constructs a resnet18 network with two classes
    """
    return models.resnet18(num_classes=2)
