"""Trainer and Tester classes for SiSR PyTorch super-resolution network
"""

import json
import os
import glob
import logging
import tarfile
import warnings

import torch

from PIL import Image
from pytorch_ssim import ssim
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torchvision.utils import save_image
from torchvision.transforms import functional as TF
from tqdm import trange, tqdm

from sisr import __version_info__, __version__
import sisr.data
import sisr.loss
import sisr.models


__all__ = 'Tester', 'Trainer'


logger = logging.getLogger(__name__)


TQDM_WIDTH = 80


class Tester:
    """PyTorch tester class
    """

    def __init__(self, options):
        super().__init__()

        # Runtime options
        accumulation_steps = options.accumulation_steps
        checkpoint = options.checkpoint
        self.device = options.device
        input_size = options.input_size
        num_workers = options.num_workers

        # Save dirs before we update pre-existing options
        data_dir = options.data_dir
        self.log_dir = options.log_dir
        self.output_dir = options.output_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Load any existing options
        options_file = os.path.join(self.log_dir, 'options.json')
        if os.path.isfile(options_file):
            logger.info("Loading existing options from '%s'", options_file)

            with open(options_file) as f:
                options.__dict__.update(json.load(f))

            if '__version_info__' not in options:
                options.__version_info__ = options.__version__.split('.')
            if tuple(options.__version_info__[:2]) != __version_info__[:2]:
                raise ValueError(
                    "Package version %r is incompatible with loaded options "
                    "version %r" % (__version__, options.__version__))

        # Restore some options
        options.accumulation_steps = accumulation_steps
        options.checkpoint = checkpoint
        options.device = self.device
        options.input_size = input_size
        options.num_workers = num_workers

        options.data_dir = data_dir
        options.log_dir = self.log_dir
        options.output_dir = self.output_dir

        # Save options to file, along with package version number
        logger.info("Saving options to '%s'", options_file)
        options.__version__ = __version__
        options.__version_info__ = __version_info__
        with open(options_file, 'w') as f:
            json.dump(options.__dict__, f, indent=2, sort_keys=True)

        # Load components
        self.load_components(options)

        # Find checkpoint
        if options.checkpoint is None:
            # Latest checkpoint
            checkpoint_glob = os.path.join(self.log_dir, '*.pth')
            checkpoint_paths = sorted(glob.glob(checkpoint_glob))
            if checkpoint_paths:
                checkpoint_path = checkpoint_paths[-1]
                checkpoint_iter = None
            else:
                checkpoint_path = None
                checkpoint_iter = None

        # Specified checkpoint
        elif os.path.isfile(options.checkpoint):
            checkpoint_path = options.checkpoint
            checkpoint_iter = None
        elif isinstance(options.checkpoint, int):
            checkpoint_path = None
            checkpoint_iter = options.checkpoint

        # Load checkpoint
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        elif checkpoint_iter is not None:
            self.load_checkpoint(iteration=checkpoint_iter)

        # Avoid polluting log with deprecation warninings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    def load_components(self, options):
        """Loads modules, models, losses, datasets etc.

        This should be called before load_checkpoint
        """
        if logger.isEnabledFor(logging.INFO):
            logger.info("Loading %s with options %s", type(self).__name__,
                        json.dumps(options.__dict__, indent=2, sort_keys=True))

        # Load dataset
        logger.info("Loading dataset from '%s'", options.data_dir)
        transform = sisr.data.JointRandomTransform(
            input_size=options.input_size)
        dataset = sisr.data.Dataset(
            data_dir=options.data_dir,
            transform=transform)

        # Set random seed so we get the same segmentation each time
        torch.manual_seed(options.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(options.seed)

        # Calculate lengths of training, validation and test sets
        lengths = [int(p * len(dataset)) for p in (0.7, 0.2, 0.1)]
        lengths[0] = len(dataset) - sum(lengths[1:])

        logger.info(
            "Segmenting dataset of length %d into train, validation and test "
            "sets of lengths %d, %d and %d respectively.",
            len(dataset), *lengths)
        training_set, validation_set, test_set = data.random_split(
            dataset, lengths)

        # Dataloaders
        self.training_loader = data.DataLoader(
            batch_size=options.batch_size,
            dataset=training_set,
            drop_last=True,
            num_workers=options.num_workers,
            shuffle=True)
        self.validation_loader = data.DataLoader(
            batch_size=options.batch_size,
            dataset=validation_set,
            drop_last=True,
            num_workers=options.num_workers)
        self.test_loader = data.DataLoader(
            dataset=test_set,
            num_workers=options.num_workers)

        # Load model
        logger.info("Loading model.")
        self.model = sisr.models.Sisr(
            num_features=options.num_features,
            num_resblocks=options.num_resblocks,
            multiply_resblocks=options.multiply_resblocks,
            scale_factor=options.scale_factor,
            upsample=options.upsample,
            weight_norm=options.weight_norm)
        self.model.to(self.device)
        logger.info("Model layers: %r", self.model)

        # Metric file
        self.metric_file = os.path.join(self.output_dir, 'metrics.json')
        self._metrics = set()

        # Epoch
        self.epoch = 0

    @property
    def checkpoint(self):
        """Builds checkpoint from the state dict of the various components
        """
        checkpoint = {}
        checkpoint['epoch'] = self.epoch
        checkpoint['model_state_dict'] = self.model.state_dict()
        return checkpoint

    @property
    def img_dir(self):
        img_dir = os.path.join(self.log_dir, 'test_images-%010d' % self.epoch)
        os.makedirs(img_dir, exist_ok=True)
        return img_dir

    @property
    def img_tar(self):
        return os.path.join(
            self.output_dir,
            os.path.basename(self.img_dir) + '.tar.gz')

    def load_checkpoint(self, checkpoint_path=None, iteration=None):
        """Loads checkpoint into the state dict of the various components
        """
        if checkpoint_path is None:
            if iteration is None:
                raise ValueError(
                    "Must specify one of 'checkpoint_path' or 'iteration'")

            checkpoint_path = os.path.join(
                self.log_dir, 'checkpoint-%010d.pth' % iteration)

        logger.info("Loading checkpoint from '%s'", checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError("No such checkpoint: %r", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    def save_checkpoint(self, checkpoint_path=None, iteration=None):
        """Saves checkpoint which can be loaded later
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.log_dir, 'checkpoint-%010d.pth' % iteration)
        logger.info("Saving new checkpoint to '%s'", checkpoint_path)
        torch.save(self.checkpoint, checkpoint_path)
        return checkpoint_path

    def _save_metric(self, metric):
        """Logs metric line to metric file and stdout
        """
        metric = '%-*s' % (TQDM_WIDTH, json.dumps(metric))
        logger.info("JSON data:\n%s", metric)
        with open(self.metric_file, 'a') as f:
            print(metric, file=f)

    def save_metric(self, metric):
        """Logs metric line, ensuring chart is initialised
        """
        if metric['chart'] not in self._metrics:
            self._metrics.add(metric['chart'])

            # Initialise chart
            self._save_metric({k: metric[k] for k in ('chart', 'axis')})

        # Save metric
        self._save_metric({k: metric[k] for k in ('chart', 'x', 'y')})

    def save_image(self, input, output, target, filename=None, iteration=None):
        """Save grid of bicubic scaled input, output and target
        """
        if filename is None:
            filename = os.path.join(self.img_dir, '%05d.png' % iteration)

        # Copy tensors to CPU, removing batch dimension
        input = input.cpu()
        output = output.cpu()
        target = target.cpu()

        # Bicubic baseline
        bicubic = TF.to_pil_image(input)
        bicubic = TF.resize(
            img=bicubic,
            size=target.size()[-2:],
            interpolation=Image.BICUBIC)
        bicubic = TF.to_tensor(bicubic)

        # Save to disk
        save_image([bicubic, output, target], filename)
        return filename

    def test(self):
        """Evaluate model using PSNR and SSIM test metrics
        """
        logger.info("Computing test metrics.")

        # Disable training layers
        self.model.train(False)

        with torch.no_grad():
            for iteration, (inputs, targets) in enumerate(tqdm(
                self.test_loader,
                unit='example',
                desc="Testing model",
                leave=True,
                ncols=TQDM_WIDTH,
            )):
                # Copy to target device
                inputs = inputs.to(self.device).requires_grad_(False)
                targets = targets.to(self.device).requires_grad_(False)

                # Inference
                outputs = self.model(inputs)

                # Compute test metrics
                psnr_metric = -10 * torch.log10(F.mse_loss(outputs, targets))
                ssim_metric = ssim(outputs, targets)

                # Save out
                for input, output, target in zip(inputs, outputs, targets):
                    self.save_image(input, output, target, iteration=iteration)
                    self.save_metric({
                        'chart': "Test PSNR",
                        'axis': "Example",
                        'x': iteration,
                        'y': psnr_metric.item()})
                    self.save_metric({
                        'chart': "Test SSIM",
                        'axis': "Example",
                        'x': iteration,
                        'y': ssim_metric.item()})

        with tarfile.open(self.img_tar, 'w:gz') as tar:
            tar.add(self.img_dir, arcname='.')

    run = test


class Trainer(Tester):
    """PyTorch trainer class
    """

    def load_components(self, options):
        super().load_components(options)

        # Load optimiser
        logger.info("Loading %s optimiser with initial learning rate %g.",
                    options.optimiser, options.lr)
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
        logger.info("Loading learning rate scheduler, to reduce learning rate "
                    "by a factor %g every %d epochs.",
                    options.step_gamma, options.step_epochs)
        self.lr_scheduler = StepLR(
            optimizer=self.optimiser,
            step_size=options.step_epochs,
            gamma=options.step_gamma)

        # Load content loss
        self.content_loss = sisr.loss.CombinedContentLoss(
            config=options.loss_configuration)
        self.content_loss.to(self.device)

        # Additional loading requirements if adversarial loss is specified
        self.adversary_weight = options.loss_configuration.get('A', 0)
        self.discriminator = self.adversary_weight != 0

        if self.discriminator:

            # Load discriminator/adversary loss
            self.discriminator_loss = nn.BCELoss()
            self.discriminator_loss.to(self.device)
            self.adversary_loss = self.discriminator_loss

            # Load discriminator
            logger.info("Loading discriminator.")
            self.discriminator_model = sisr.models.Discriminator()
            self.discriminator_model.to(self.device)
            logger.info("Discriminator layers: %r", self.discriminator_model)

            # Load discriminator optimiser
            logger.info("Loading %s discriminator optimiser with initial "
                        "learning rate %g.",
                        options.optimiser, options.discriminator_lr)
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
            logger.info("Loading learning rate scheduler, to reduce learning "
                        "rate for discriminator by factor %g every %d "
                        "epochs.",
                        options.step_gamma, options.step_epochs)
            self.discriminator_lr_scheduler = StepLR(
                optimizer=self.discriminator_optimiser,
                step_size=options.step_epochs,
                gamma=options.step_gamma)

        # Epoch settings
        self.max_epochs = options.max_epochs
        self.pretrain_epochs = options.pretrain_epochs

        # Accumulation reduces memory usage without impacting training
        self.accumulation_steps = options.accumulation_steps
        self.accumulation_size = options.batch_size // self.accumulation_steps

    @property
    def checkpoint(self):
        """Builds checkpoint from the state dict of the various components
        """
        checkpoint = super().checkpoint

        # Basic training components
        checkpoint['optimiser_state_dict'] = self.optimiser.state_dict()
        checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        # Additional GAN components
        if self.adversary_weight != 0:
            checkpoint['discriminator'] = {}
            checkpoint['discriminator']['model_state_dict'] = \
                self.discriminator_model.state_dict()
            checkpoint['discriminator']['optimiser_state_dict'] = \
                self.discriminator_optimiser.state_dict()
            checkpoint['discriminator']['lr_scheduler_state_dict'] = \
                self.discriminator_lr_scheduler.state_dict()

        return checkpoint

    def load_checkpoint(self, *args, **kwargs):
        """Loads checkpoint into the state dict of the various components
        """
        checkpoint = super().load_checkpoint(*args, **kwargs)

        # Basic training components
        if 'optimiser_state_dict' in checkpoint:
            self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(
                checkpoint['lr_scheduler_state_dict'])

        # Additional GAN components
        if 'discriminator' in checkpoint and self.adversary_weight != 0:
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

    def criteria(
        self,
        outputs,
        targets,
        fake_predictions=None,
    ):
        """Compute all loss criteria
        """
        losses = {}

        # Compute content loss
        losses['generator'] = \
            self.content_loss(outputs, targets) / self.accumulation_steps

        if fake_predictions is not None:

            # GAN targets
            real_targets = torch.ones(fake_predictions.size())
            real_targets = real_targets.to(self.device)

            # Compute adversary loss, fool the discriminator
            losses['adversary'] = self.adversary_loss(
                fake_predictions,
                real_targets) / self.accumulation_steps

            # Update generator loss
            losses['content'] = losses['generator']
            losses['generator'] = \
                losses['content'] + self.adversary_weight * losses['adversary']

        return losses

    def discriminator_criteria(
        self,
        fake_predictions,
        real_predictions,
    ):
        """Compute discriminator loss criteria
        """
        losses = {}

        # Soft GAN targets
        #   Salimans et al. (2016) arXiv:1606.03498
        real_targets = torch.rand(fake_predictions.size()) * 0.5 + 0.7
        fake_targets = torch.rand(fake_predictions.size()) * 0.3
        real_targets = real_targets.to(self.device)
        fake_targets = fake_targets.to(self.device)

        # Compute discriminator loss
        losses['discriminator real'] = self.discriminator_loss(
            real_predictions,
            real_targets) / self.accumulation_steps
        losses['discriminator fake'] = self.discriminator_loss(
            fake_predictions,
            fake_targets) / self.accumulation_steps
        losses['discriminator'] = \
            losses['discriminator real'] + losses['discriminator fake']

        return losses

    def train(self):
        """Train models for one epoch of the training set
        """
        logger.info("Training model and computing mean training loss.")

        # Set models to training mode
        self.model.train(True)
        if self.discriminator:
            self.discriminator_model.train(True)

        # Calculate mean losses across training set
        mean_losses = {}

        # Train for one epoch of training set
        for iteration, (inputs_batch, targets_batch) in enumerate(tqdm(
            self.training_loader,
            unit='minibatch',
            desc="Training model",
            ncols=TQDM_WIDTH,
        )):
            # Accumulate losses for logging purposes
            total_losses = {}

            # Zero gradients
            self.optimiser.zero_grad()

            # Process minibatch in multiple forward passes
            for inputs, targets in zip(
                inputs_batch.split(self.accumulation_size),
                targets_batch.split(self.accumulation_size),
            ):
                # Copy tensors to target device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                if self.discriminator:
                    fake_predictions = self.discriminator_model(outputs)
                else:
                    fake_predictions = None

                # Compute losses
                losses = self.criteria(
                    outputs=outputs,
                    targets=targets,
                    fake_predictions=fake_predictions)
                logger.debug('Losses: %r', losses)

                # Backward pass
                losses['generator'].backward()

                with torch.no_grad():
                    for k in losses:
                        total_losses[k] = total_losses.get(k, 0)
                        total_losses[k] += losses[k]

            # Update parameters
            self.optimiser.step()

            # Train discriminator if enabled
            if self.discriminator:

                # Zero gradients
                self.discriminator_optimiser.zero_grad()

                # Process minibatch in multiple forward passes
                for inputs, targets in zip(
                    inputs_batch.split(self.accumulation_size),
                    targets_batch.split(self.accumulation_size),
                ):
                    # Copy tensors to target device
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # Forward pass
                    with torch.no_grad():
                        outputs = self.model(inputs)
                    fake_predictions = self.discriminator_model(outputs)
                    real_predictions = self.discriminator_model(targets)

                    # Compute losses
                    losses = self.discriminator_criteria(
                        fake_predictions,
                        real_predictions)
                    logger.debug('Losses: %r', losses)

                    # Backward pass
                    losses['discriminator'].backward()

                    with torch.no_grad():
                        for k in losses:
                            total_losses[k] = total_losses.get(k, 0)
                            total_losses[k] += losses[k]

                # Update parameters
                self.discriminator_optimiser.step()

            # Compute running means
            with torch.no_grad():
                for k in losses:
                    mean_losses[k] = mean_losses.get(k, 0)
                    mean_losses[k] *= iteration / (iteration + 1)
                    mean_losses[k] += total_losses[k] / (iteration + 1)

        for k in mean_losses:
            self.save_metric({
                'chart': "Training %s loss" % k,
                'axis': "Epoch",
                'x': self.epoch,
                'y': mean_losses[k].item()})

    def validate(self):
        """Evaluate average loss across validation set
        """
        logger.info("Computing mean validation loss.")

        # Set models to testing mode
        self.model.train(False)
        if self.discriminator:
            self.discriminator_model.train(False)

        # Calculate mean losses across validation set
        with torch.no_grad():
            mean_losses = {}
            for iteration, (inputs_batch, targets_batch) in enumerate(tqdm(
                self.validation_loader,
                unit='minibatch',
                desc="Validating model",
                ncols=TQDM_WIDTH,
            )):
                total_losses = {}

                # Process minibatch in multiple forward passes
                for inputs, targets in zip(
                    inputs_batch.split(self.accumulation_size),
                    targets_batch.split(self.accumulation_size),
                ):
                    # Copy tensors to target device
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # Forward pass
                    outputs = self.model(inputs)
                    if self.discriminator:
                        fake_predictions = self.discriminator_model(outputs)
                        real_predictions = self.discriminator_model(targets)
                    else:
                        fake_predictions = None

                    # Compute losses
                    losses = self.criteria(
                        outputs,
                        targets,
                        fake_predictions)
                    if self.discriminator:
                        discriminator_losses = self.discriminator_criteria(
                            fake_predictions,
                            real_predictions)
                        losses.update(discriminator_losses)

                    for k in losses:
                        total_losses[k] = total_losses.get(k, 0)
                        total_losses[k] += losses[k]

                # Compute running means
                for k in losses:
                    mean_losses[k] = mean_losses.get(k, 0)
                    mean_losses[k] *= iteration / (iteration + 1)
                    mean_losses[k] += total_losses[k] / (iteration + 1)

        for k in mean_losses:
            self.save_metric({
                'chart': "Validation %s loss" % k,
                'axis': "Epoch",
                'x': self.epoch,
                'y': mean_losses[k].item()})

    def run(self):
        """Continue training from loaded epoch
        """
        discriminator = self.discriminator
        for self.epoch in trange(
            self.epoch,
            self.max_epochs,
            initial=self.epoch,
            total=self.max_epochs,
            unit='epoch',
            desc='Running %s' % type(self).__name__,
            ncols=TQDM_WIDTH,
        ):
            # Pretrain without discriminator
            if self.epoch < self.pretrain_epochs:
                self.discriminator = False
            elif self.epoch == self.pretrain_epochs:
                self.discriminator = discriminator

            # Save our progress
            self.save_checkpoint(iteration=self.epoch)

            # Advance learning rate schedule
            self.lr_scheduler.step()
            if discriminator:
                self.discriminator_lr_scheduler.step()

            # Log learning rate
            self.save_metric({
                'chart': "Learning rate",
                'axis': "Epoch",
                'x': self.epoch,
                'y': self.lr_scheduler.get_lr()[0]})
            if discriminator:
                self.save_metric({
                    'chart': "Discriminator learning rate",
                    'axis': "Epoch",
                    'x': self.epoch,
                    'y': self.discriminator_lr_scheduler.get_lr()[0]})

            # Train and validate each epoch
            self.train()
            self.validate()

        # Test final model
        self.test()
