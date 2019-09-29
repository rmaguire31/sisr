#!/usr/bin/env python3
"""SiSR PyTorch Network training entry-point
"""

import argparse
import json
import logging

import sisr.bin
from sisr.run import Trainer


logger = logging.getLogger(__name__)


def build_parser():
    """Build CLI parser with options for train.py
    """
    # Inherit package arguments
    parents = sisr.bin.build_parser(),

    parser = argparse.ArgumentParser(
        description="Train SiSR super-resolution network",
        parents=parents)

    parser.add_argument(
        '--adam_betas',
        type=float, nargs=2, default=(0.9, 0.999),
        help="Adam coefficients used for computing running averages of "
             "gradient and its square. See "
             "https://arxiv.org/pdf/1412.6980.pdf")

    parser.add_argument(
        '--adam_epsilon',
        type=float, default=1e-8,
        help="Adam term added to the denominator to improve numerical "
             "stability. See https://arxiv.org/pdf/1412.6980.pdf")

    parser.add_argument(
        '--discriminator_adam_betas',
        type=float, nargs=2, default=(0.9, 0.999),
        help="Adam coefficients used for computing running averages of "
             "gradient and its square. "
             "See https://arxiv.org/pdf/1412.6980.pdf")

    parser.add_argument(
        '--discriminator_adam_epsilon',
        type=float, default=1e-8,
        help="Adam term added to the denominator to improve numerical "
             "stability. See https://arxiv.org/pdf/1412.6980.pdf")

    parser.add_argument(
        '--discriminator_lr', '--discriminator_learning_rate',
        type=float, default=1e-4,
        help="Initial learning rate for discriminator.")

    parser.add_argument(
        '--discriminator_optimiser', '--discriminator_optimizer',
        type=str.lower, default='adam', choices={'adam', 'sgd'},
        help="Method for stochastic optimisation. For details on Adam, see "
             "https://arxiv.org/pdf/1412.6980.pdf")

    parser.add_argument(
        '--loss_configuration', '--loss_config', '--loss_cfg',
        type=json.loads, required=True,
        help="JSON string defining the various loss components used in to "
             "the SiSR generator network.")

    parser.add_argument(
        '--lr', '--learning_rate',
        type=float, default=1e-4,
        help="Initial learning rate.")

    parser.add_argument(
        '--num_features',
        type=int, default=64,
        help="How many features are extracted by the convolutional layers.")

    parser.add_argument(
        '--num_resblocks',
        type=int, default=8,
        help="How many residual learning blocks to use.")

    parser.add_argument(
        '--max_epochs',
        type=int, default=10000,
        help="Training is automatically stopped after this many epochs.")

    parser.add_argument(
        '--multiply_resblocks',
        type=float, default=1.0,
        help="Residual block multiply factor.")

    parser.add_argument(
        '--optimiser', '--optimizer',
        type=str.lower, default='adam', choices={'adam', 'sgd'},
        help="Method for stochastic optimisation. For details on Adam, see "
             "https://arxiv.org/pdf/1412.6980.pdf")

    parser.add_argument(
        '--pretrain_epochs',
        type=int, default=5,
        help="Discriminator is disabled for this many epochs.")

    parser.add_argument(
        '--scale_factor',
        type=float, default=4,
        help="Linear upscaling factor of super-resolution network.")

    parser.add_argument(
        '--step_gamma',
        type=float, default=1.0,
        help="Decay factor gamma at each scheduled learning rate update.")

    parser.add_argument(
        '--step_epochs',
        type=int, default=500,
        help="Number of epochs between scheduled learning rate updates.")

    parser.add_argument(
        '--upsample',
        type=str.lower, default='nearest', choices={'nearest', 'shuffle'},
        help="Upsample method to use in feed forward network.")

    parser.add_argument(
        '--weight_norm',
        action='store_true',
        help="Enable weight normalisation. See "
             "https://arxiv.org/abs/1602.07868")

    return parser


def main():
    """Entry point
    """
    # Command line interface
    parser = build_parser()
    options = parser.parse_args()
    Trainer(options).run()


if __name__ == '__main__':
    main()
