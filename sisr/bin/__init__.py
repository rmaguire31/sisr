"""Common cli utilities
"""

import argparse
import logging

import torch


logger = logging.getLogger(__name__)


class ConfigureLogging(argparse.Action):
    """Argparse action to set logging level
    """

    def __init__(
        self,
        nargs=None,
        const=None,
        type=str.upper,
        choices={'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'},
        default=None,
        help="Set logging level.",
        **kwargs,
    ):
        """Initialise action, enforcing some arguments
        """
        if const is not None:
            raise ValueError("const is not allowed")
        if nargs is not None:
            raise ValueError("nargs is not allowed")
        if type is not str.upper:
            raise ValueError("type must be str.upper")
        if default in choices:
            logging.basicConfig(level=vars(logging)[default])
        super().__init__(type=type, choices=choices, help=help, **kwargs)

    def __call__(self, parser, namespace, values, option_strings):
        """Set logging level
        """
        logging.basicConfig(level=vars(logging)[values])


def build_parser():
    """Configure argument parser with common SiSR package options
    """
    # Determine available devices
    devices = {'cuda:%d' % i for i in range(torch.cuda.device_count())}
    devices.add('cpu')

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--batch_size', type=int, default=16,
        help="How many samples per batch to load.")

    parser.add_argument('--checkpoint', type=str,
        help="Checkpoint file to load.")

    parser.add_argument('--data_dir', type=str, required='/storage/dataset',
        help="Path to directory containing dataset.")

    parser.add_argument('--device', type=str, default='cuda:0',
        choices=devices,
        help="Device to use for tensor computation.")

    parser.add_argument('--input_size', type=int, nargs=2, default=None,
        metavar=('height', 'width'),
        help="Dimensions in pixels of patches extracted from input images. "
             "If omitted, the original image size is used.")

    parser.add_argument('--log_dir', type=str, required=True,
        help="Path to directory for tensorboard logs and checkpoints.")

    parser.add_argument('--logging', action=ConfigureLogging, default='INFO')

    parser.add_argument('--num_workers', type=int, default=1,
        help="How many subprocesses to use for data loading. 0 means that the "
             "data will be loaded in the main process.")

    parser.add_argument('--seed', type=int, default=1234,
        help="Seed for torch random number generators.")

    parser.add_argument('--output_dir', type=str, default='/artifacts',
        help="Path to directory containing dataset.")

    return parser
