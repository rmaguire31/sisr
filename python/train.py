#!/usr/bin/env python3
"""Tool for testing SiSR PyTorch Network
"""

import argparse
import json
import logging

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import sisr
import sisr.data
import sisr.models
import sisr.loss


logger = logging.getLogger(__name__)


def build_parser():
    """Build CLI parser with options for train.py
    """
    # Inherit package arguments
    parents = sisr.build_parser(),

    parser = argparse.ArgumentParser(
        description="Train SiSR super-resolution network",
        parents=parents)

    parser.add_argument('--adam', action='store_true',
        help="Enable Adam for stochastic optimisation. See "
             "https://arxiv.org/pdf/1412.6980.pdf")

    parser.add_argument('--adam_betas', type=float, nargs=2,
        default=(0.9, 0.999),
        help="Adam coefficients used for computing running averages of "
             "gradient and its square. See https://arxiv.org/pdf/1412.6980.pdf")

    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
        help="Adam term added to the denominator to improve numerical "
             "stability. See https://arxiv.org/pdf/1412.6980.pdf")

    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-4,
        help="Initial learning rate.")

    parser.add_argument('--num_filters', type=int, default=64,
        help="How many filters in each convolutional layer.")

    parser.add_argument('--num_resblocks', type=int, default=8,
        help="How many residual learning blocks to use.")

    parser.add_argument('--scale_factor', type=float, default=4,
        help="Linear upscaling factor of super-resolution network.")

    parser.add_argument('--step_gamma', type=float, default=0,
        help="Decay factor gamma at each scheduled learning rate update.")

    parser.add_argument('--step_size', type=int, default=int(2e5),
        help="Number of epochs between scheduled learning rate updates.")

    parser.add_argument('--upsample', type=str.lower, default='nearest', 
        choices={'nearest', 'shuffle'},
        help="Upsampling method to use in feed forward network.")

    parser.add_argument('--weight_norm', action='store_true',
        help="Enable weight normalisation. See "
             "https://arxiv.org/abs/1602.07868")

    return parser


def train(options):
    """Train network
    """
    # Dataloader
    transform = sisr.data.JointRandomTransform(
        input_size=options.input_size)
    dataset = sisr.data.Dataset(options.data_dir, transform=transform)
    dataloader = DataLoader(dataset,
        num_workers=options.num_workers,
        batch_size=options.batch_size)

    # TODO<rsm>: load network from sisr.model
    from torch import nn
    net = nn.Sequential(
        nn.Conv2d(1, options.num_filters, 3),
        nn.Upsample(scale_factor=options.scale_factor),
        nn.Conv2d(options.num_filters, 1, 3))

    # Optimiser
    if options.adam:
        optimiser = Adam(net.parameters(),
            lr=options.learning_rate,
            betas=options.adam_betas,
            eps=options.adam_epsilon)
    else:
        optimiser = SGD(net.parameters(),
            lr=options.learning_rate)

    # Learning rate scheduler
    scheduler = StepLR(optimiser,
        step_size=options.step_size,
        gamma=options.step_gamma)

    for iteration, (inputs, targets) in enumerate(dataloader):
        
        # Copy inputs and targets to the correct device
        inputs = inputs.to(options.device)
        targets = targets.to(options.device)

        logger.info("Iteration: %d, inputs: %r, targets: %r",
                    iteration, inputs.size(), targets.size())
        # TODO<rsm>: run inference
        # TODO<rsm>: calculate loss
        # TODO<rsm>: backpropagate gradients
        # TODO<rsm>: descend gradients


if __name__ == '__main__':

    # Command line interface
    parser = build_parser()
    options = parser.parse_args()

    options_json = json.dumps(vars(options), sort_keys=True, indent=2)
    logger.info("Options: %s", options_json)

    train(options)
