"""Trainer and Tester classes for SiSR PyTorch super-resolution network
"""

import os
import glob
import logging
import time

from PIL import Image
from pytorch_ssim import ssim
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from torchvision.transforms import functional as TF

from sisr import __version__
import sisr.data
import sisr.loss
import sisr.models


__all__ = 'Tester', 'Trainer'


logger = logging.getLogger(__name__)


class Tester(nn.Module):
    """PyTorch tester class
    """

    def __init__(self, options):
        super().__init__()

        # Choose runtime device before we update pre-existing options
        self.device = options.device

        # Save output dir before we update pre-existing options
        self.output_dir = options.output_dir
        os.makedirs(options.output_dir, exist_ok=True)

        # Load any existing options
        self.options_file = os.path.join(options.log_dir, 'options.json')
        if os.path.isfile(self.options_file):
            logger.info("Loading existing options from %s", self.options_file)

            with open(options_file) as f:
                options.__dict__.update(json.load(f))

            if options.__version__ != __version__:
                raise ValueError("Loaded options for different sisr version")

        # Restore some options
        options.device = self.device
        options.output_dir = self.output_dir

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
        if logger.isEnabledFor(logging.INFO):
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

        logger.info("Segmenting dataset of length %d into train, validation "
                    "and test sets of lengths %d, %d and %d",
                    len(dataset), *lengths)
        train_set, validation_set, test_set = random_split(dataset, lengths)

        # Dataloaders
        self.train_loader = DataLoader(
            batch_size=options.batch_size,
            dataset=training_set,
            drop_last=True,
            num_workers=options.num_workers,
            shuffle=True)
        self.validation_loader = DataLoader(
            batch_size=options.batch_size,
            dataset=validation_set,
            drop_last=True,
            num_workers=options.num_workers)
        self.test_loader = DataLoader(
            dataset=test_set,
            num_workers=options.num_workers)

        # Load model
        logger.info("Loading model")
        self.model = sisr.models.Sisr(
            num_features=options.num_features,
            num_resblocks=options.num_resblocks,
            scale_factor=options.scale_factor,
            weight_norm=options.weight_norm,
            upscaling=options.upscaling)
        self.model.to(self.device)

        # Metric file
        self.metric_file = os.path.join(self.output_dir, 'metrics.json')

    @property
    def checkpoint(self):
        """Builds checkpoint from the state dict of the various components
        """
        checkpoint = {}
        checkpoint['model_state_dict'] = self.model.state_dict()
        return checkpoint

    def load_checkpoint(self, checkpoint_path=None, iteration=None):
        """Loads checkpoint into the state dict of the various components
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                log_dir, 'checkpoint-%010d.pth' % iter)
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    def save_checkpoint(self, checkpoint_path=None, iteration=None):
        """Saves checkpoint which can be loaded later
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                log_dir, 'checkpoint-%010d.pth' % iteration)
        torch.save(self.checkpoint, checkpoint_path)
        return checkpoint_path

    def save_metric(self, metric):
        """Logs metric line to metric file and stdout
        """
        json = json.dumps(metric)
        with open(self.metric_file, 'w') as f:
            print(json)
            print(json, file=f)

    init_metric = save_metric

    def save_image(self, input, output, target, filename=None, iteration=None):
        """
        """
        if filename is None:
            filename = os.path.join(output_dir, '%05d.png' % iteration)

        # Copy tensors to CPU
        input = input.cpu()
        output = output.cpu()
        target = target.cpu()

        # Bicubic baseline
        bicubic = TF.to_tensor(
            TF.resize(
                TF.to_pil_image(input), target.size(),
                interpolation=Image.BICUBIC))

        # Save to disk
        save_image((bicubic, output, target), filename)
        return filename

    def test(self):
        """Evaluate model using PSNR and SSIM test metrics
        """
        # Disable training layers
        self.model.train(False)

        # Initialise charts
        self.init_metric({'chart': 'Test SSIM', 'axis': 'example'})
        self.init_metric({'chart': 'Test PSNR', 'axis': 'example'})

        for iteration, (input, target) in self.test_loader:

            # Copy to target device
            input = input.to(self.device)
            target = target.to(self.device)

            # Inference
            output = self.model(input)

            # Compute test metrics
            psnr_metric = -10 * torch.log10(F.mse_loss(output, target))
            ssim_metric = ssim(output, target)

            # Save out
            self.save_image(input, output, target)
            self.save_metric({'chart': 'Test PSNR', 'x': iteration, 'y': psnr_metric})
            self.save_metric({'chart': 'Test SSIM', 'x': iteration, 'y': ssim_metric})


class Trainer(Tester):
    """PyTorch trainer class
    """

    def load_components(self, options):
        super().load_components(options)

        # Load optimiser
        if options.optimiser == 'adam':
            self.optimiser = Adam(
                params=self.model.parameters(),
                lr=options.lr,
                betas=options.adam_betas,
                eps=options.adam_epsilon)
        elif options.optimiser == 'sgd':
            self.optimiser = SGD(
                params=self.model.parameters(),
                lr=options.lr)
        else:
            raise ValueError("Unknown value %r for option 'optimiser'" %
                             options.optimiser)

        # Load learning rate scheduler
        self.lr_scheduler = StepLR(
            optimiser=self.optimiser,
            step_size=options.step_size,
            gamma=options.step_gamma)

        # Load content loss
        self.content_loss = sisr.loss.CombinedContentLoss(
            config=options.loss_configuration)
        self.content_loss.to(self.device)

        # Additional loading requirements if adversarial loss is specified
        if 'A' in options.loss_configuration:

            # Load adversarial loss
            self.adversarial_loss = nn.BCELoss()
            self.adversarial_loss.to(self.device)

            # Load discriminator
            self.discriminator_model = sisr.models.Discriminator()
            self.discriminator_model.to(self.device)

            # Load discriminator optimiser
            if options.discriminator_optimiser == 'adam':
                self.discriminator_optimiser = Adam(
                    params=self.discriminator_model.parameters(),
                    lr=options.discriminator_lr,
                    betas=options.discriminator_adam_betas,
                    eps=options.discriminator_adam_epsilon)
            elif options.discriminator_optimiser == 'sgd':
                self.discriminator_optimiser = SGD(
                    params=self.discriminator_model.parameters(),
                    lr=options.discriminator_lr)
            else:
                raise ValueError("Unknown value %r for option "
                                 "'discriminator_optimiser'" %
                                 options.discriminator_optimiser)

            # Load discriminator learning rate scheduler
            self.disciminator_lr_scheduler = StepLR(
                optimiser=self.disciminator_optim,
                step_size=options.step_size,
                gamma=options.step_gamma)

        # Maximum number of epochs and iterations
        self.max_epochs = options.max_epochs
        self.max_iterations = options.max_iterations

    @property
    def checkpoint(self):
        """Builds checkpoint from the state dict of the various components
        """
        checkpoint = super().checkpoint

        # Basic training components
        checkpoint['epoch'] = self.epoch
        checkpoint['optimiser_state_dict'] = self.optimiser.state_dict()
        checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        # Additional GAN components
        if adversarial_weight != 0:
            checkpoint['discriminator'] = {}
            checkpoint['discriminator']['model_state_dict'] = \
                self.discriminator_model.state_dict()
            checkpoint['discriminator']['optimiser_state_dict'] = \
                self.discriminator_optimiser.state_dict()
            checkpoint['discriminator']['lr_scheduler_state_dict'] = \
                self.discriminator_lr_scheduler.state_dict()

        return checkpoint

    def load_checkpoint(self, checkpoint_path):
        """Loads checkpoint into the state dict of the various components
        """
        checkpoint = super().load_checkpoint(checkpoint_path)

        # Basic training components
        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']
        if 'iteration' in checkpoint:
            self.iteration = checkpoint['iteration']
        if 'optimiser_state_dict' in checkpoint:
            self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(
                checkpoint['lr_scheduler_state_dict'])

        # Additional GAN components
        if 'discriminator' in checkpoint and adversarial_weight != 0
            if 'model_state_dict' in checkpoint['discriminator']:
                self.discriminator_model.load_state_dict(
                    checkpoint['discriminator']['model_state_dict'])
            if 'optimiser_state_dict' in checkpoint['discriminator']:
                self.discriminator_optimiser.load_state_dict(
                    checkpoint['discriminator']['optimiser_state_dict'])
            if 'lr_scheduler_state_dict' in checkpoint['discriminator']:
                self.discriminator_lr_scheduler.load_state_dict(
                    checkpoint['discriminator']['lr_scheduler_state_dict'])

        return checkpoint

        def train(self):
            """
            """
            # Set models to training mode
            self.model.train(True)
            if self.adversarial_weight != 0:
                self.discriminator_model.train(True)

            for iteration, (inputs, targets) in enumerate(self.train_loader):

                # Copy tensors to target device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Inference
                outputs = self.model(inputs)

                # Compute loss
                loss = self.content_loss(outputs, targets)
                if self.adversarial_weight != 0:

                    discriminator_outputs = self.discriminator_model(outputs)
                    discriminator_targets = torch.ones(
                        discriminator_outputs.size()).to(self.device)
                    loss += self.adversarial_weight * self.adversarial_loss(
                        discriminator_outputs,
                        discriminator_targets)

                # Backpropagate gradients
                loss.backward()

                # Update parameters
                self.lr_scheduler.step()
                self.optimiser.step()

