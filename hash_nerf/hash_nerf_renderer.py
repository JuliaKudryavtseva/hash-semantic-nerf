import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from typing import Optional

import nerfacc

class HashRenderer(nn.Module):
    """Calculate probability of labels  along ray."""

    @classmethod
    def forward(
        cls,
        embeds: Float[Tensor, "bs num_samples num_classes"],
        weights: Float[Tensor, "bs num_samples 1"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "bs num_classes"]:
        """Calculate semantics along the ray."""
        if ray_indices is not None and num_rays is not None:
            output = nerfacc.accumulate_along_rays(
                weights[..., 0], values=embeds, ray_indices=ray_indices, n_rays=num_rays
            )  
            output+=0.00001
            return output / (torch.linalg.norm(output, dim=-1, keepdim=True)+ 0.00001)          
        return output
