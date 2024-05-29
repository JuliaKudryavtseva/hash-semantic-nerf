import json
import os
from tqdm import tqdm
from PIL import Image
import argparse
import time 

import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


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

    # get args
    args = parse_args()
    INPUT_NAME = args.data_path
    OUTPUT_NAME = args.exp_name
    

    # ------ Define pathes -------
    # input
    video_path = os.path.join('data', INPUT_NAME, 'images') # input
    frames = sorted(os.listdir(video_path), key=lambda x: int(x.split('_')[1].split('.')[0]))

    # output 
    save_path = os.path.join('data', OUTPUT_NAME, 'segmentation_results')   # output
    os.makedirs(os.path.join(save_path, 'numpy_masks'), exist_ok=True)


    # ------ Start segmentaiton -------
    print('Experiment name: ', args.exp_name, '\nInput path: ', video_path, 'Output path: ', save_path)
    results = {}

    for image_pill in tqdm(frames):
        # read image
        image_path = os.path.join(video_path, image_pill)
            
        image =Image.open(image_path)
        image = image.convert('RGB')
        image_array = np.array(image)
        H, W, _ = image_array.shape

        
        # SAM segmentation
        sam_results = mask_generator.generate(image_array)

        # save numpy masks in save_path
        path_frame = os.path.join(save_path, 'numpy_masks', image_pill.split('.')[0])
        os.makedirs(path_frame, exist_ok=True)

        for inn_mask, mask in enumerate(sam_results):
            path_mask = os.path.join(save_path, 'numpy_masks', image_pill.split('.')[0], str(inn_mask))
            np.save(path_mask, mask['segmentation'])

            del mask['segmentation']
            mask['segmentation'] =  'numpy_masks'+'/'+image_pill.split('.')[0]+'/'+str(inn_mask)+'.npy'
            
            results[image_pill] = sam_results


    with open(os.path.join(save_path, f'{OUTPUT_NAME}.json') , 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)