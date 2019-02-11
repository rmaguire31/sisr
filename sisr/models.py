"""PyTorch models for training and testing SiSR super-resolution network
"""

import logging

from torch import nn
from torch.nn import functional as F
from torchvision import models


___all___ = 'Sisr', 'FeatureExtractor', 'Discriminator'


logger = logging.getLogger(__name__)


class Sisr(nn.Module):
    """SiSR generator network
    """

    def __init__(
        self,
        kernel_size=3,
        num_channels=1,
        num_features=64,
        num_resblocks=8,
        scale_factor=4,
        weight_norm=True,
        upsample='nearest',
    ):
        super().__init__()

        # Colour space to feature space
        conv = nn.Conv2d(
            num_channels,
            num_features,
            kernel_size,
            padding=kernel_size//2)
        if weight_norm:
            conv = nn.utils.weight_norm(conv)
        self.head = nn.Sequential(conv)

        # Residual body
        self.body = nn.Sequential(*[
            _ResidualBlock(num_features=num_features, weight_norm=weight_norm)
            for _ in range(num_resblocks)])

        # Upsample features, then feature space to colour space
        layers = []
        while scale_factor > 1:

            for sf in 3, 2:
                if scale_factor % sf == 0:

                    if upsample == 'nearest':
                        nearest = nn.Upsample(scale_factor=sf, mode='nearest')
                        conv = nn.Conv2d(
                            num_features,
                            num_features,
                            kernel_size,
                            padding=kernel_size//2)
	                # Is this ReLU really needed?
                        relu = nn.ReLU(inplace=False)
                        if weight_norm:
                            conv = nn.utils.weight_norm(conv)

                        layers.append(nearest)
                        layers.append(conv)
                        layers.append(relu)

                    elif upsample == 'shuffle':
                        conv = nn.Conv2d(
                            num_features,
                            num_features * sf**2,
                            kernel_size,
                            padding=kernel_size//2)
                        shuffle = nn.PixelShuffle(sf)
                        if weight_norm:
                            conv = nn.utils.weight_norm(conv)

                        layers.append(conv)
                        layers.append(shuffle)

                    scale_factor /= sf

        conv = nn.Conv2d(
            num_features,
            num_channels,
            kernel_size,
            padding=kernel_size//2)
        self.tail = nn.Sequential(*layers, conv)

    def forward(self, x):
        x = self.head(x)
        x = x + self.body(x)
        x = self.tail(x)
        return x


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
                conv_idx += 1

            elif isinstance(layer, nn.ReLU):
                self.names.append('relu%d_%d' % (pool_idx, conv_idx))

            elif isinstance(layer, nn.MaxPool2d):
                self.names.append('pool%d' % (pool_idx))
                pool_idx += 1
                conv_idx = 1

        # Save modules in a list, making sure we do not use ReLU inplace so we
        # extract arbitrary features
        self.features = nn.ModuleList(
            nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
            for layer in layers)

    def forward(self, x, names=set()):

        # Make sure names are valid
        names = set(names)
        if not names.issubset(self.names):
            logger.warning("Unknown feature names: %r",
                           names - set(self.names))

        # Make sure greyscale images have three channels
        x = x.expand(-1, 3, -1, -1)

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


class Discriminator(nn.Module):
    """Discriminator for SiSR, borrowed from
    """

    def __init__(self, num_channels=1, kernel_size=3):
        super().__init__()

        # Basic blocks
        blocks = []
        in_channels = num_channels
        for out_channels in 64, 128, 256, 512:
            blocks.append(
                _BasicBlock(in_channels, out_channels, kernel_size))
            in_channels = out_channels

        # Fully convolutional final layer before avg pool
        conv = nn.Conv2d(in_channels, 1, 1)

        self.body = nn.Sequential(*blocks, conv)

    def forward(self, x):
        x = self.body(x)
        # Dynamically sized avg pool
        return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)


class _ResidualBlock(nn.Module):
    """Residual learning block
    """

    def __init__(
        self,
        kernel_size=3,
        num_features=64,
        weight_norm=True
    ):
        super().__init__()

        conv1 = nn.Conv2d(
            num_features,
            num_features,
            kernel_size,
            padding=kernel_size//2)
        relu1 = nn.ReLU(inplace=False)
        conv2 = nn.Conv2d(
            num_features,
            num_features,
            kernel_size,
            padding=kernel_size//2)
        if weight_norm:
            conv1 = nn.utils.weight_norm(conv1)
            conv2 = nn.utils.weight_norm(conv2)
        self.body = nn.Sequential(conv1, relu1, conv2)

    def forward(self, x):
        return x + self.body(x)


class _BasicBlock(nn.Sequential):
    """Downscaling block for discriminator
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        alpha=1e-2,
        stride=2
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size//2)]
        if in_channels > 3:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend((
            nn.LeakyReLU(alpha, inplace=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha, inplace=False)))
        return super().__init__(*layers)
