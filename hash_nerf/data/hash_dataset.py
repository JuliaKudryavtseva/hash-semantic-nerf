"""
SAM dataset.
"""

from typing import Dict
import numpy as np
import torch
import cv2

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

from hash_nerf.data.base_hash_dataparser import HashLabels

class HashDataset(InputDataset):
    """Dataset that returns images and labels of SAM masks.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """
    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["labels_map", "frame_number"]
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        assert "hash_labels" in dataparser_outputs.metadata.keys() and isinstance(self.metadata["hash_labels"], HashLabels)
        self.hash_labels = self.metadata["hash_labels"]
        self.device = torch.device
        
    def get_metadata(self, data: Dict) -> Dict:

        filepath_frame = self.hash_labels.filenames_labels[data["image_idx"]]
        semantics =  torch.from_numpy(np.load(filepath_frame))

        return {"semantics": semantics}

        # map_path = self.hash_labels.filemap
        # labels_map = torch.from_numpy(np.load(map_path)).cuda()

        # filepath_frame = self.hash_labels.filenames_labels[data["image_idx"]]
        # filepath_frame = str(filepath_frame.stem)        
        # frame_number = torch.from_numpy(np.full(labels_map.shape, int(filepath_frame.split("_")[1]))).cuda() # torch.Size(image.shape)

        # return {"labels_map": labels_map, "frame_number": frame_number}
        