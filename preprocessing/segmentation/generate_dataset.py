import json
import os
from tqdm import tqdm
from PIL import Image
import argparse
import time 

import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from cupy_module import cupy_get_IOU, cupy_get_intern




def remove_duplicates(frame_ann):
    masks = [mask['segmentation'] for mask in frame_ann]
    dup_ious = cupy_get_IOU(masks, masks)
    dup_ious = dup_ious - np.eye(*dup_ious.shape)

    rows_nonzero, col_nonzero = np.nonzero(dup_ious > 0.5)
    unique_masks = np.arange(dup_ious.shape[1]).tolist()

    for row, col in zip(rows_nonzero, col_nonzero):
        if (row in unique_masks) and (col in unique_masks):
            unique_masks.remove(col)

    uni_masks = [mask for ind, mask in enumerate(masks) if ind in unique_masks]
    frame_ann = [val for ind, val in enumerate(frame_ann) if ind in unique_masks]

    del dup_ious
    del masks

    return frame_ann


# NMS algorithm
def NMS_like_algo(masks):
    #------------------ Non-max supression ------------------
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    seg_masks = [mask['segmentation'] for mask in masks]
    intern_matrix, intern_matrix_index = cupy_get_intern(seg_masks, seg_masks)

    dup = []
    for i, mask1 in enumerate(seg_masks):
        for j, mask2 in enumerate(seg_masks):
            if i !=j  and i > 0 and j > 0:

                metrics = intern_matrix[i][j]
                m = i if intern_matrix_index[i][j] == 0 else j

                # teatime 0.5
                if metrics > 0.3:
                    dup.append(m)

    masks = [mask for ind, mask in enumerate(masks) if ind not in np.unique(dup)]
    return masks

def filter_small(masks, ratio=0.01):
    h, w = masks[0]['segmentation'].shape
    
    results = []
    for mask in masks:
        if mask['area'] / (h*w)  > ratio:
            results.append(mask)
    del masks
    return results



# parsing args
def parse_args():

    parser = argparse.ArgumentParser(description ='args for algorithm which makes frame consistant')

    parser.add_argument('--data-path', type=str, default='teatime',  help='Path to the data.')
    parser.add_argument('--exp-name', type=str, default='teatime_large', help='Here you can specify the name of the experiment.')
    return parser.parse_args()


if __name__ == '__main__':

    # ------ init SAM model ------ 
    MODEL_TYPE = "vit_h"
    CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"
    DEVICE = "cuda"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
   
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=16)
    # ---------------------------- 

    args = parse_args()


    video_path = os.path.join('data', args.data_path, 'images') # input
    frames = sorted(os.listdir(video_path), key=lambda x: int(x.split('_')[1].split('.')[0]))

    OUTPUT_NAME = args.exp_name 

    save_path = os.path.join('data', OUTPUT_NAME, 'reverse_segmentation_results')   # output
    os.makedirs(os.path.join(save_path, 'numpy_masks'), exist_ok=True)
    
    print('Experiment name: ', args.exp_name, '\nInput path: ', video_path, 'Output path: ', save_path)

    results = {}

    for image_pil in tqdm(frames):
        # read image
        image_path = os.path.join(video_path, image_pil)
            
        image =Image.open(image_path)
        image = image.convert('RGB')
        image_array = np.array(image)
        
        # SAM segmentation
        masks = mask_generator.generate(image_array)

        # NMS like procedure
        masks = NMS_like_algo(masks)

      
        # save numpy masks in save_path
        path_frame = os.path.join(save_path, 'numpy_masks', image_pil.split('.')[0])
        os.makedirs(path_frame, exist_ok=True)

          
        for inn_mask, mask in enumerate(masks):
            path_mask = os.path.join(save_path, 'numpy_masks', image_pil.split('.')[0], str(inn_mask))
            np.save(path_mask, mask['segmentation'])

            del mask['segmentation']
            mask['segmentation'] =  'numpy_masks'+'/'+image_pil.split('.')[0]+'/'+str(inn_mask)+'.npy'

            results[image_pil] = masks

    with open(os.path.join(save_path, f'{OUTPUT_NAME}.json') , 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)