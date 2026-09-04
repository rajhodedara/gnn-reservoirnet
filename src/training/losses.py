import torch
import torch.nn as nn
from typing import List

class PinballLoss(nn.Module):
    """
    Quantile regression loss for multiple quantiles.
    
    Attributes:
        quantiles (List[float]): List of quantiles to compute loss for.
    """
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        """
        Initializes the PinballLoss.

        Args:
            quantiles (List[float]): Quantiles to compute loss for.
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the pinball loss.

        Args:
            y_pred (torch.Tensor): Predicted values of shape (batch_size, num_targets, num_quantiles).
            y_true (torch.Tensor): Ground truth values of shape (batch_size, num_targets).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        loss = 0.0
        # Assume y_pred has quantiles along the last dimension
        for i, q in enumerate(self.quantiles):
            pred_q = y_pred[..., i]
            error = y_true - pred_q
            loss += torch.max(q * error, (q - 1) * error).mean()
        return loss / len(self.quantiles)

class ONIConditionedAsymmetricLoss(nn.Module):
    """
    Asymmetric penalty that scales with ONI index.
    Penalizes overestimation of inflows during drought conditions (El Nino).
    """
    def __init__(self, threshold: float = 0.5, scaling_factor: float = 2.0):
        """
        Initializes the ONIConditionedAsymmetricLoss.

        Args:
            threshold (float): ONI threshold above which penalty applies.
            scaling_factor (float): Factor to scale the penalty by.
        """
        super().__init__()
        self.threshold = threshold
        self.scaling_factor = scaling_factor

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, oni: torch.Tensor) -> torch.Tensor:
        """
        Computes the asymmetric loss.

        Args:
            y_pred (torch.Tensor): Predicted median values (or target quantile).
            y_true (torch.Tensor): Ground truth values.
            oni (torch.Tensor): ONI values of shape (batch_size,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        error = y_pred - y_true # Positive if overestimating
        
        # Gamma scales up when ONI > threshold
        gamma = torch.relu(oni - self.threshold) * self.scaling_factor
        
        # Apply penalty only to overestimations (error > 0)
        while gamma.dim() < error.dim():
            gamma = gamma.unsqueeze(-1)
        penalty = gamma * torch.relu(error)
        
        return penalty.mean()

class CombinedLoss(nn.Module):
    """
    Weighted combination of Pinball Loss and ONI Conditioned Asymmetric Loss.
    """
    def __init__(self, alpha: float = 1.0, quantiles: List[float] = [0.1, 0.5, 0.9], threshold: float = 0.5, scaling_factor: float = 2.0):
        """
        Initializes the CombinedLoss.

        Args:
            alpha (float): Weight for the pinball loss.
            quantiles (List[float]): Quantiles for PinballLoss.
            threshold (float): Threshold for Asymmetric Loss.
            scaling_factor (float): Scaling factor for Asymmetric Loss.
        """
        super().__init__()
        self.alpha = alpha
        self.pinball = PinballLoss(quantiles=quantiles)
        self.asymmetric = ONIConditionedAsymmetricLoss(threshold=threshold, scaling_factor=scaling_factor)
        # Assuming the 50th percentile (median) is used for the asymmetric penalty, which is index 1 for [0.1, 0.5, 0.9]
        self.median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, oni: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined loss.

        Args:
            y_pred (torch.Tensor): Predictions of shape (batch_size, num_targets, num_quantiles).
            y_true (torch.Tensor): Targets of shape (batch_size, num_targets).
            oni (torch.Tensor): ONI index of shape (batch_size,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        l_pinball = self.pinball(y_pred, y_true)
        y_pred_median = y_pred[..., self.median_idx]
        l_asym = self.asymmetric(y_pred_median, y_true, oni)
        
        return self.alpha * l_pinball + l_asym
