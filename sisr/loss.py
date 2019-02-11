"""PyTorch loss utilities for SiSR super-resolution network
"""

import logging

import torch

from torch import nn
from torch.nn import functional as F

import sisr.models


___all___ = 'CombinedContentLoss',


logger = logging.getLogger(__name__)


class CombinedContentLoss(nn.Module):
    """Computes the combined content loss defined by a configuration dict
    """

    def __init__(self, config={}):
        super().__init__()

        self.feature_names = set()
        self.mse_weight = 0
        self.contextual_weights = {}
        self.perceptual_weights = {}

        for component, weights in config.items():
            logger.debug("Component: %r: %r", component, weights)

            # Sanity check weights
            value_error = ValueError("Argument config should be a dict of "
                                     "dicts or floats, not %s" %
                                     type(weights).__name__)
            if isinstance(weights, dict):
                if not all(
                    isinstance(k, str) and isinstance(v, (int, float))
                    for k, v in weights.items()
                ):
                    raise value_error

                # Keep master set of features
                self.feature_names |= weights.keys()

            elif not isinstance(weights, (int, float)):
                raise value_error

            # Adversarial loss
            if component == 'A':
                # Adversarial loss needs to be computed seperately
                pass

            # MSE loss
            elif component == 'E':
                self.mse_weight = float(weights)

            # Contextual loss
            elif component == 'C':
                self.contextual_weights = {k: float(v)
                                           for k, v in weights.items()}

            # Perceptual loss
            elif component == 'P':
                self.perceptual_weights = {k: float(v)
                                           for k, v in weights.items()}

            else:
                raise NotImplementedError("Unknown content loss component %r" %
                                          component)

        if self.feature_names:
            self.feature_extractor = sisr.models.FeatureExtractor()
            self.feature_extractor.train(False)

            if not self.feature_names.issubset(self.feature_extractor.names):
                unknown = \
                    self.feature_names - set(self.feature_extractor.names)
                raise ValueError("Unknown feature names %r" % unknown)

    def forward(self, outputs, targets):
        loss = 0
        if self.mse_weight:
            loss += self.mse_weight * F.mse_loss(outputs, targets)

        # Extract
        if self.feature_names:
            output_features = self.feature_extractor(outputs,
                                                     self.feature_names)
            target_features = self.feature_extractor(targets,
                                                     self.feature_names)

        for feature_name, weight in self.contextual_weights.items():
            loss -= weight * torch.log(_contextual_similarity(
                output_features[feature_name],
                target_features[feature_name]))

        for feature_name, weight in self.perceptual_weights.items():
            loss += weight * F.mse_loss(
                output_features[feature_name],
                target_features[feature_name])

        return loss


def _contextual_similarity(
    outputs,
    targets,
    eps1=1e-8,
    eps2=1e-5,
    h=1,
    reduction='mean'
):
    """Contextual similarity metric

    References:
        Mechrez et al. (2018) arXiv:1803.02077
    """
    # Apply operation batchwise
    C = []
    for output, target in zip(outputs, targets):
        num_features, *_ = output.size()

        # Two collections of vectors with order num_features
        X = output.view(num_features, -1).t()
        Y = target.view(num_features, -1).t()

        # Normalise w.r.t mean point in feature space
        X = X - X.mean(dim=0)
        Y = Y - Y.mean(dim=0)

        # Pairwise matrix of cosine distances between vectors of X and Y
        dxy = torch.stack([
            1 - torch.mm(X, y.view(-1, 1)).view(-1) / (
                X.norm(p=2, dim=1) * y.norm(p=2) + eps1)
            for y in Y])

        # Normalise w.r.t row minimum
        dxy_ = dxy / (dxy.min(dim=0)[0] + eps2)

        # Normalised cosine distance to a similary
        wxy = torch.exp((1 - dxy_) / h)

        # Normalise w.r.t row sum
        cxy = wxy / wxy.sum(dim=0)

        # Contextual similarity is average of column maximums
        C_ = cxy.max(dim=1)[0].mean()
        C.append(C_)

    # Combine batchwise elements
    C = torch.stack(C)
    if reduction == 'mean':
        C = C.mean()

    return C


def _gram_matrix(outputs):
    """Computes the gram matrix, the inner product of a matrix with itself
    """
    # Apply operation batch wise
    G = []
    for output in outputs:
        # View output in two dimensions
        c, *_ = output.size()
        X = output.view(c, -1)

        # Compute inner product and normalise by size of output
        G_ = torch.mm(X, X.t())

        # Normalise w.r.t number of elements
        G_ /= X.numel()
        G.append(G_)

    return torch.stack(G)
