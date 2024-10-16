from typing import Any


class CovarianceLoss:
    def __call__(self, features) -> Any:
        return self.compute(features)

    def compute(self, z) -> Any:
        # Center the feature vectors by subtracting the mean along the batch dimension
        z_centered = z - z.mean(dim=0, keepdim=True)
        
        # Compute the covariance matrix (normalized by batch size)
        batch_size = z.size(0)
        cov_matrix = (z_centered.T @ z_centered) / (batch_size - 1)
        
        # Compute the off-diagonal elements of the covariance matrix
        cov_loss = cov_matrix.fill_diagonal_(0).pow(2).sum()
        
        return cov_loss
    