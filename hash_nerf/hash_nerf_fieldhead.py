from typing import Optional

from nerfstudio.field_components.field_heads import FieldHead
from hash_nerf.hash_nerf_fieldheadname import HashFieldHeadNames


class HashFieldHead(FieldHead):
    """Labels output

    Args:
        labels_n_dims: embed dimention 
        in_dim: input dimension. If not defined in constructor, it must be set later.
        activation: output head activation
    """

    def __init__(self, labels_n_dims: int, in_dim: Optional[int] = None, activation=None) -> None:
        super().__init__(in_dim=in_dim, out_dim=labels_n_dims, field_head_name=HashFieldHeadNames.pred_labels, activation=activation)