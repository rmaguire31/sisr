#!/usr/bin/env python3
"""Tool for training SiSR PyTorch Network
"""

import argparse
import json
import logging

from torch.utils.data import DataLoader

import sisr
import sisr.data


logger = logging.getLogger(__name__)


def build_parser():
    """Build CLI Parser for with options for test.py
    """
    # Inherit package arguments
    parents = sisr.build_parser(),

    parser = argparse.ArgumentParser(
        description="Test SiSR super-resolution network",
        parents=parents)

    return parser


def test(options):
    """Test network
    """
    transform = sisr.data.JointRandomTransform(
        input_size=options.input_size)
    dataset = sisr.data.Dataset(options.data_dir, transform=transform)
    dataloader = DataLoader(dataset,
        num_workers=options.num_workers)

    for iteration, (inputs, targets) in enumerate(dataloader):

        # Copy inputs and targets to the correct device
        inputs = inputs.to(options.device)
        targets = targets.to(options.device)

        logger.info("Iteration: %d, inputs: %r, targets: %r",
                    iteration, inputs.size(), targets.size())
        # TODO<rsm>: run inference
        # TODO<rsm>: compute metrics
        # TODO<rsm>: log


if __name__ == '__main__':

    # Command line interface
    parser = build_parser()
    options = parser.parse_args()

    options_json = json.dumps(vars(options), sort_keys=True, indent=2)
    logger.info("Options: %s", options_json)

    test(options)
