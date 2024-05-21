import cupy as cp
import gc
import numpy as np

def cupy_get_IOU(curr_masks, next_masks):
    curr_masks = cp.array(curr_masks)
    next_masks = cp.array(next_masks)

    curr_masks_binary = (curr_masks > 0).astype(cp.int32)
    next_masks_binary = (next_masks > 0).astype(cp.int32)

    num_curr_masks = curr_masks.shape[0]
    num_next_masks = next_masks.shape[0]
    batch_size = 8

    iou_matrix = cp.zeros((num_curr_masks, num_next_masks), dtype=cp.float32)

    for i in range(0, num_curr_masks, batch_size):
        curr_batch = curr_masks_binary[i:i + batch_size, cp.newaxis, :, :]
        intersection_batch = cp.sum(curr_batch & next_masks_binary, axis=(2, 3))
        union_batch = cp.sum(curr_batch | next_masks_binary, axis=(2, 3))

        iou_matrix[i:i + batch_size, :] = intersection_batch / union_batch

    del curr_masks_binary, next_masks_binary, intersection_batch, union_batch, curr_batch
    gc.collect()

    return np.array(iou_matrix.get())