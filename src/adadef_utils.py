# `adaptive_defense`

# Copyright 2024-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import gc
import typing as t

import numpy as np
import torch
import torchdata
import torchvision
import tqdm
from loguru import logger
from psmi import PSMI

import base
import data
import models


class MedianLossTracker:
    """Class responsible for tracking the median loss and determining if we should
    do normal training or noisy training, and when PSMI should be computed."""

    def __init__(self, num_samples: int, decay_threshold: float):
        """
        Initializes the tracker with a given number of samples to keep track of.

        Args:
            num_samples (int): The maximum number of latest loss samples to keep.
            decay_threshold (float): The decay of the loss between the first complete array
                (containing num_samples elements) and the one that triggers PSMI computation.
        """
        self.num_samples = num_samples
        self.decay_threshold = decay_threshold
        self.psmi_already_computed: bool = False
        self._losses = torch.tensor([], dtype=torch.float32, device="cpu")
        self._initial_median_loss: float = None

    def update(self, loss: torch.Tensor) -> None:
        """
        Updates the loss tracker with a new batch of losses.

        Args:
            loss (torch.Tensor): A 1D tensor of loss values from the current batch.
        """
        # Shuffle the loss tensor
        loss = loss.detach().cpu()
        shuffled_loss = loss[torch.randperm(loss.size(0))]

        # Concatenate the new shuffled losses with the existing losses
        self._losses = torch.cat((shuffled_loss, self._losses))

        # Should we fill the initial loss value?
        if (
            self._initial_median_loss is None
            and self._losses.size(0) >= self.num_samples
        ):
            self._initial_median_loss = (
                self._losses[-self.num_samples :].median().item()
            )

        # Clip the losses to keep only the latest num_samples
        if self._losses.size(0) > self.num_samples:
            self._losses = self._losses[: self.num_samples]

    def median(self) -> float:
        """
        Computes and returns the median of the tracked losses.

        Returns:
            float: The median of the tracked losses.
        """
        if self._losses.numel() == 0:
            raise ValueError("No losses tracked yet.")

        return self._losses.median().item()

    def normal_training(self) -> bool:
        """We do normal training untill psmi is computed."""
        return not self.psmi_already_computed

    def noisy_training(self) -> bool:
        """We do noisy training if PSMI has already been computed."""
        return self.psmi_already_computed

    def compute_psmi_now(self) -> bool:
        """We compute PSMI if it has not been computed yet and the median training
        loss has decreased enough"""

        if (
            self.psmi_already_computed  # PSMI already computed
            or self._initial_median_loss is None  # Not enough samples
            or self.median()
            >= (1 - self.decay_threshold)
            * self._initial_median_loss  # Not deceased enough
        ):
            return False

        self.psmi_already_computed = True
        return True


class NoiseManager:
    def __init__(
        self,
        base_relu_slope: float,
        slope_epoch_increase: float,
        alpha_norm: float,
        relu_quantile_offset: float = 0,
    ) -> None:

        # ReLU slope
        self.base_relu_slope: float = base_relu_slope
        self.current_relu_slope: float = base_relu_slope
        self.relu_quantile_offset: float = relu_quantile_offset
        self.relu_real_offset: t.Optional[float] = None

        # ReLU increase
        self.num_samples_per_epoch: t.Optional[int] = None
        self.slope_epoch_increase: float = slope_epoch_increase
        self.num_samples_since_lase_increase: int = 0

        # Exponential smooting of the norm
        self.alpha_norm: float = alpha_norm

        # PSMI scores
        self.psmi_scores = None
        self.loss_scores = None
        self.logit_gap_scores = None
        self.mean_sample_norm: t.Optional[float] = None

    def noise(self, batch_idx: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """TODO"""
        # Updating sample norm
        batch_size, hidden_dim = features.size()
        mean_batch_norm = torch.mean(torch.norm(features.detach().cpu(), dim=1)).item()
        conservation_coeff = (1 - self.alpha_norm) ** batch_size
        self.mean_sample_norm: float = (
            conservation_coeff * self.mean_sample_norm
            + (1 - conservation_coeff) * mean_batch_norm
        )

        # Updating ReLU slope
        self.num_samples_since_lase_increase += batch_size
        if self.num_samples_since_lase_increase > self.num_samples_per_epoch:
            self.current_relu_slope += self.base_relu_slope * self.slope_epoch_increase
            self.num_samples_since_lase_increase -= self.num_samples_per_epoch

        # Noising
        # We renormalize bu (hidden_dim**0.5) to make sure that the expected squared norm
        # Of the noise for the k-th vector is indeed `sigmas[k]**2`
        batch_psmi = torch.tensor(self.psmi_scores[batch_idx], dtype=base.DTYPE)
        sigmas = (
            torch.relu(-batch_psmi + self.relu_real_offset)
            * self.current_relu_slope
            * self.mean_sample_norm
        )
        noise_result = (
            torch.randn(batch_size, hidden_dim)
            * sigmas.view(-1, 1)
            / (hidden_dim**0.5)
        )

        return noise_result

    def compute_scores(
        self,
        model: models.WideResNet,
        train_data: data.Dataset,
        verbose: bool = True,
    ):
        """TODO"""

        # Pre-processing
        logger.info("Computing PSMI scores")
        logger.debug("Building PSMI data_pipe")
        psmi_train_datapipe = train_data.build_datapipe(add_global_index=True)
        psmi_train_datapipe = psmi_train_datapipe.map(
            torchvision.transforms.v2.ToDtype(base.DTYPE, scale=True),
            input_col=1,
            output_col=1,
        )
        psmi_train_datapipe = psmi_train_datapipe.map(
            torchvision.transforms.v2.Normalize(
                mean=data.CIFAR10_MEAN,
                std=data.CIFAR10_STD,
            ),
            input_col=1,
            output_col=1,
        )
        psmi_train_datapipe = psmi_train_datapipe.batch(
            base.EVAL_BATCH_SIZE, drop_last=False
        ).collate()

        # Init
        logger.debug(f"Forward pass for PSMI")
        model.eval()
        features = []  # List of tensors of shape (bs, 256)
        labels = []  # List of tensors of shape (bs,)
        global_idx = []  # List of tensors of shape (bs,)
        losses = []  # list of tensors of shape (bs,)
        logit_gaps = []  # list of tensors of shape (bs,)
        loss = torch.nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad():
            for batch_global_idx, batch_xs, batch_ys, in tqdm.tqdm(
                psmi_train_datapipe,
                desc="PSMI eval",
                unit="batch",
                leave=True,
                disable=not verbose,
            ):

                # Forward pass
                batch_hidden = model.forward(
                    batch_xs.to(dtype=base.DTYPE, device=base.DEVICE),
                    return_hidden=True,
                )  # Shape (bs, 256)
                batch_pred = model.dense(batch_hidden)  # Shape (bs, 10)
                batch_loss = loss(batch_pred.cpu(), batch_ys)  # Shape (bs,)

                # Logit gap
                batch_pred_logit = batch_pred[torch.arange(len(batch_ys)), batch_ys]
                batch_pred[torch.arange(len(batch_ys)), batch_ys] = float("-inf")
                batch_logit_gaps = (
                    batch_pred_logit - torch.max(batch_pred, dim=1).values
                )

                # Saving
                logit_gaps.append(batch_logit_gaps.detach().cpu())
                losses.append(batch_loss.detach().cpu())
                features.append(batch_hidden.cpu())
                labels.append(batch_ys)
                global_idx.append(batch_global_idx)

                # Cleaning
                del (
                    batch_hidden,
                    batch_pred,
                    batch_loss,
                    batch_pred_logit,
                    batch_logit_gaps,
                )
                gc.collect()

        # Concatenating
        logger.debug("PSMI computation core")
        logit_gaps = np.concatenate(logit_gaps)
        losses = np.concatenate(losses)  # Shape (len(train_data),)
        features = np.concatenate(features)  # Shape (len(train_data), 256)
        labels = np.concatenate(labels)  # Shape (len(train_data),)
        global_idx = np.concatenate(global_idx)  # Shape (len(train_data),)

        estimator = PSMI()
        psmi_mean, _, _ = estimator.fit_transform(features, labels)
        logger.debug(f"Number of estimator used: {estimator.n_estimators}")

        # Saving - PSMI
        logger.debug(f"Saving mean PSMI value")
        max_idx = np.max(global_idx)
        self.psmi_scores = np.empty(max_idx + 1)
        self.psmi_scores[:] = np.nan
        for idx, val in zip(global_idx, psmi_mean):
            self.psmi_scores[idx] = val

        # Saving - Loss
        logger.debug(f"Saving mean Losses value")
        self.loss_scores = np.empty(max_idx + 1)
        self.loss_scores[:] = np.nan
        for idx, val in zip(global_idx, losses):
            self.loss_scores[idx] = val

        # Saving - Logit gaps
        logger.debug(f"Saving mean Logit Gap value")
        self.logit_gap_scores = np.empty(max_idx + 1)
        self.logit_gap_scores[:] = np.nan
        for idx, val in zip(global_idx, logit_gaps):
            self.logit_gap_scores[idx] = val

        # Saving - ReLU commputation
        self.num_samples_per_epoch = psmi_mean.shape[0]
        if self.relu_quantile_offset == 0:
            self.relu_real_offset = float(0)
        else:
            self.relu_real_offset = float(
                np.quantile(psmi_mean, self.relu_quantile_offset)
            )

        # Saving - Mean norm
        logger.debug(f"Saving mean sample norm")
        self.mean_sample_norm = np.mean(np.linalg.norm(features, axis=1))

        logger.debug(f"End of NoiseManager.compute_score")
