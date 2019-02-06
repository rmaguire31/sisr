"""PyTorch Dataset utilities for SiSR super-resolution dataset
"""

import os
import glob
import random
import logging

import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset as BaseDataset


__all__ = 'Dataset', 'JointRandomTransform'


logger = logging.getLogger(__name__)


class Dataset(BaseDataset):
    """Paired dataset of input and target images
    """

    FILE_EXTENSIONS = {'png', 'PNG', 'jpg', 'JPG'}

    def __init__(self, data_dir, transform=None):
        self.transform = transform

        filenames = set()
        for file_extension in self.FILE_EXTENSIONS:

            input_glob = os.path.join(
                data_dir,
                'inputs',
                '*.%s' % file_extension)

            target_glob = os.path.join(
                data_dir,
                'targets',
                '*.%s' % file_extension)

            input_filenames = glob.glob(input_glob)
            target_filenames = glob.glob(target_glob)

            input_basenames = {
                os.path.basename(f)
                for f in input_filenames
                if os.path.isfile(f)}
            target_basenames = {
                os.path.basename(f)
                for f in target_filenames
                if os.path.isfile(f)}

            basenames = input_basenames & target_basenames

            input_filenames = sorted(
                f for f in input_filenames
                if os.path.basename(f) in basenames)
            target_filenames = sorted(
                f for f in target_filenames
                if os.path.basename(f) in basenames)

            filenames.update(set(zip(input_filenames, target_filenames)))

        self.filenames = sorted(filenames)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_filename, target_filename = self.filenames[idx]

        # Open PIL Images
        input_image = Image.open(input_filename)
        target_image = Image.open(target_filename)

        if self.transform is not None:
            input_image, target_image = self.transform(input_image, target_image)

        return input_image, target_image


class JointRandomTransform:
    """Apply the same to two input and target images with different scales

    Applies random crop, flip and rotation
    """

    def __init__(self, input_size=None):
        if input_size is None:
            self.crop_width = self.crop_height = None
        else:
            self.crop_width, self.crop_height = input_size

    def __call__(self, input, target):
        # Random patch extraction
        if self.crop_width is not None and self.crop_height is not None:
            
            width, height = input.size
            scaled_width, scaled_height = target.size

            # Determine image scale factor
            scale = scaled_width / width
            if scale != scaled_height / height:
                logger.warning("Input and target image have different aspect "
                               "ratios: %r, %r",
                               input.size, target.size)
            if not scale.is_integer():
                logger.warning("Target image size is not an integer multiple "
                               "of input image size: %r, %r",
                               input.size, target.size)
            scale = int(scale)

            # Random top, left position for patch
            left = random.randrange(0, width - self.crop_width)
            top = random.randrange(0, height - self.crop_height)

            # Crop
            input = TF.crop(input, top, left, self.crop_height, self.crop_width)
            target = TF.crop(target, scale*top, scale*left,
                             scale*self.crop_height, scale*self.crop_width)

        # Random horizontal flip and rotation
        width, height = input.size
        if width == height:
            angle = random.randrange(0, 360, 90)
        else:
            angle = random.randrange(0, 360, 180)
        flip = random.randint(0, 1)

        if angle:
            input = TF.rotate(input, angle)
            target = TF.rotate(target, angle)
        if flip:
            input = TF.hflip(input)
            target = TF.hflip(target)

        # Convert to tensor
        input = TF.to_tensor(input)
        target = TF.to_tensor(target)

        return input, target
