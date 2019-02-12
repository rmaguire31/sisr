#!/usr/bin/env python3
"""SiSR PyTorch Network training entry-point
"""

import argparse
import logging

import sisr.bin
from sisr.run import Tester


logger = logging.getLogger(__name__)


def build_parser():
    """Build CLI parser with options for train.py
    """
    # Inherit package arguments
    parents = sisr.bin.build_parser(),

    parser = argparse.ArgumentParser(
        description="Test SiSR super-resolution network",
        parents=parents)

    return parser


def main():
    """Entry point
    """
    # Command line interface
    parser = build_parser()
    options = parser.parse_args()
    Tester(options).run()


if __name__ == '__main__':
    main()
