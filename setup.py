"""SiSR PyTorch Network for super-resolution in semiconductor device inspection
"""

from setuptools import setup, find_packages


setup(
    name='sisr',
    version='0.1.0.dev0',
    description=__doc__,
    author='rsm',
    author_email='rmaguire31@gmail.com',
    keywords='deep-learning PyTorch torch machine_learning deep-learning '
        'super-resolution silicon semiconductor inspection SiSR EDSR WDSR',
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
        'git+git://github.com/Po-Hsun-Su/pytorch-ssim.git#egg=pytorch_ssim'
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'train.py=sisr.bin.train:main',
            'test.py=sisr.bin.test:main',
        ],
    },
)