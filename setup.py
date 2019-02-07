"""SiSR package installer
"""

import os
import sisr

from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='sisr',
    version=sisr.__version__,
    description=sisr.__doc__,
    long_description=long_description,
    author='rmaguire31',
    author_email='rmaguire31@gmail.com',
    keywords='deep-learning PyTorch torch machine_learning deep-learning '
        'super-resolution silicon semiconductor inspection SiSR EDSR WDSR',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Artifical Intelligence',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # These classifiers are *not* checked by 'pip install'. See instead
        # 'python_requires' below.
        'Programming Language :: Python :: 3 :: Only'
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.5',
    install_requires=[
        'pillow>=5.4.1',
        'numpy>=1.16.0',
        'torch>=1.0.0',
        'torchvision>=0.2.1',
        'tensorboardx>=1.6',
        'pytorch_ssim',
    ],
    dependency_links=[
        'git+git://github.com/Po-Hsun-Su/pytorch-ssim.git@master=pytorch_ssim'
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'train.py=sisr.bin.train:main',
            'test.py=sisr.bin.test:main',
        ],
    },
)
