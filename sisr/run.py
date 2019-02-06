"""Trainer and Tester classes for SiSR PyTorch super-resolution network
"""

import logging

import torch


logger = logging.getLogger(__name__)


class Runner(nn.Module):
    """Base class for Tester and Trainer
    """

    def __init__(self):
        pass