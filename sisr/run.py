"""Trainer and Tester classes for SiSR PyTorch super-resolution network
"""

import os
import glob
import logging
import time

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR

from sisr import __version__
import sisr.data
import sisr.loss
import sisr.models


__all__ = 'Tester', 'Trainer'


logger = logging.getLogger(__name__)


class Runner(nn.Module):
    """Base class for Tester and Trainer
    """

    def __init__(self, options):
        super().__init__()

        # Load any existing options
        self.options_file = os.path.join(options.log_dir, 'options.json')
        if os.path.isfile(self.options_file):
            logger.info("Loading existing options from %s", self.options_file)

            with open(options_file) as f:
                options.__dict__.update(json.load(f))

            if options.__version__ != __version__:
                raise ValueError("Loaded options for different sisr version")

        # Save options to file, along with package version number
        logger.info("Saving options to %s", self.options_file)
        options.__version__ = __version__
        os.makedirs(options.log_dir, exist_ok=True)
        with open(options_file, 'w') as f:
            json.dump(options.__dict__, f, indent=2, sort_keys=True)

        # Load components
        self.load_components()

        # Find checkpoint
        if 'checkpoint' in options:
            # Specified checkpoint
            checkpoint_path = os.path.join(options.log_dir, options.checkpoint)
        else:
            # Latest checkpoint
            checkpoint_glob = os.path.join(options.log_dir, '*.pth')
            checkpoint_paths = sorted(glob.glob(checkpoint_glob))
            if checkpoint_paths:
                checkpoint_path = checkpoint_paths[-1]
            else:
                checkpoint_path = None

        # Load checkpoint
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_components(self, options):
        """Loads modules, models, losses, datasets etc.

        This should be called before load_checkpoint
        """
        logger.info("Loading %s with options %r", type(self).__name__,
            json.dumps(options.__dict__, indent=2, sort_keys=True))

        # Load dataset
        logger.info("Loading dataset from %s", options.data_dir)
        transform = sisr.data.JointRandomTransform(
            input_size=options.input_size)
        dataset = sisr.data.Dataset(
            data_dir=options.data_dir,
            transform=transform)

        # Set random seed so we get the same segmentation each time
        torch.manual_seed(options.seed)
        if torch.cuda.isavailable():
            torch.cuda.manual_seed_all(options.seed)

        # Calculate lengths of training, validation and test sets
        lengths = [int(p*len(dataset)) for p in 0.7, 0.2, 0.1]
        lengths[0] = len(datset) - sum(lengths[0:])

        logger.info("Segmenting dataset of length %d into training, validation "
                    "and test sets of lengths %d, %d and %d",
                    len(dataset), *lengths)
        training_set, validation_set, test_set = random_split(dataset, lengths)

        self.training_set = DataLoader(
            batch_size=options.batch_size,
            dataset=training_set,
            drop_last=True,
            num_workers=options.num_workers,
            shuffle=True)
        self.validation_set = DataLoader(
            batch_size=options.batch_size,
            dataset=validation_set,
            drop_last=True,
            num_workers=options.num_workers)
        self.test_set = DataLoader(
            dataset=test_set,
            num_workers=options.num_workers)

        # Load model
        logger.info("Loading model")
        self.model = sisr.models.SiSR(
            num_features=options.num_features,
            num_resblocks=options.num_resblocks,
            scale_factor=options.scale_factor,
            weight_norm=options.weight_norm,
            upscaling=options.upscaling)

    @property
    def checkpoint(self):
        """Builds checkpoint from the state dict of the various components
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        return checkpoint

    def load_checkpoint(self, checkpoint_path):
        """Loads checkpoint into the state dict of the various components
        """
        checkpoint = torch.load(checkpoint_path)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        return checkpoint


class Trainer(Runner):
    """PyTorch trainer class
    """

    def load_components(self, options):
        super().load_components(options)

        # Load optimiser
        if options.optim == 'adam':
            self.optim = Adam(
                params=self.model.parameters(),
                lr=options.lr,
                betas=options.adam_betas,
                eps=options.adam_epsilon)
        elif options.optim == 'sgd':
            self.optim = SGD(
                params=self.model.parameters(),
                lr=options.lr)
        else:
            raise ValueError("Unknown value %r for option 'optim'" %
                options.optim)

        # Load learning rate scheduler
        self.lr_scheduler = StepLR(
            optimiser=self.optim,
            step_size=options.step_size,
            gamma=options.step_gamma)

        # Load content loss
        self.content_loss = sisr.loss.CombinedContentLoss(
            config=options.loss_configuration)

        # Additional loading requirements if adversarial loss is specified
        if 'A' in options.loss_configuration:

            # Load adversarial loss
            self.adversarial_loss = nn.BCELoss()

            # Load discriminator
            self.discriminator_model = sisr.models.Discriminator()
            if 'model_state_dict' in self.checkpoint.get('discriminator', {}):
                self.discriminator_model.load_state_dict(
                    self.checkpoint['discriminator']['model_state_dict'])

            # Load discriminator optimiser
            if options.discriminator_optim == 'adam':
                self.discriminator_optim = Adam(
                    params=self.discriminator_model.parameters(),
                    lr=options.discriminator_lr,
                    betas=options.discriminator_adam_betas,
                    eps=options.discriminator_adam_epsilon)
            elif options.discriminator_optim == 'sgd':
                self.discriminator_optim = SGD(
                    params=self.discriminator_model.parameters(),
                    lr=options.discriminator_lr)
            else:
                raise ValueError("Unknown value %r for option "
                    "'discriminator_optim'" % options.discriminator_optim)
            if 'optim_state_dict' in self.checkpoint.get('discriminator', {}):
                self.discriminator_optim.load_state_dict(
                    self.checkpoint['discriminator']['optim_state_dict'])

            # Load discriminator learning rate scheduler
            self.disciminator_lr_scheduler = StepLR(
                optimiser=self.disciminator_optim,
                step_size=options.step_size,
                gamma=options.step_gamma)
            if 'lr_scheduler_state_dict' in self.checkpoint.get('discriminator', {}):
                self.disciminator_lr_scheduler.load_state_dict(
                    self.checkpoint['lr_scheduler_state_dict'])

    @property
    def checkpoint(self):
        """Builds checkpoint from the state dict of the various components
        """
        checkpoint = super().checkpoint
        checkpoint.update({
            'epoch': self.epoch,
            'optim_state_dict': self.optim.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
        })
        return checkpoint

    def load_checkpoint(self, checkpoint_path):
        """Loads checkpoint into the state dict of the various components
        """
        checkpoint = super().load_checkpoint(checkpoint_path)

        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']
        if 'optim_state_dict' in checkpoint:
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(
                checkpoint['lr_scheduler_state_dict'])
    
