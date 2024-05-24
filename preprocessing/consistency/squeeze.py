import numpy as np
import os
from tqdm import tqdm
import pickle
import random
import colorsys
from PIL import Image
import matplotlib.pyplot as plt


def get_number_mask(path, frame_name):
    all_pairs = []
    for name in frame_name:
        frame_path = os.path.join(path, name)
        
        frame_number = int(name.split('_')[1])
        all_pairs.extend([int(m.split('.')[0]) for m in os.listdir(frame_path)])

    label_classes = np.unique(np.array(all_pairs))
    return label_classes[-1]


def random_colors(N):
    random_color=np.random.choice(range(255), size=(N, 3))
    return {i: random_color[i] for i in range(N)}


if __name__ == '__main__':

    DATA_PATH = os.environ['DATA_PATH']
    assert len(DATA_PATH) > 0


    inp_path = os.path.join('data', DATA_PATH, 'consistent_masks')
    out_path = os.path.join('data', DATA_PATH, 'consistent_masks', 'ready_masks')
    os.makedirs(out_path, exist_ok=True)

    frame_names = os.listdir(inp_path) 
    frame_names.remove('ready_masks')
    frame_names.remove('logs.txt')
    if 'label_map_colors.pkl' in frame_names:
        frame_names.remove('label_map_colors.pkl')
    

    frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1]))
    
    
    number_masks = get_number_mask(inp_path, frame_names)
    # labels 
    np.save(os.path.join(inp_path, 'number_class.npy'), number_masks)
    print(f'Number classes: {number_masks}')

    # define colors
    colors = random_colors(number_masks)
    with open((os.path.join(inp_path, 'label_map_colors.pkl')), 'wb') as f:
        pickle.dump(colors, f)


    for frame_name in tqdm(frame_names):
        # load masks
        frame_path = os.path.join(inp_path, frame_name)
        frame_masks_name = sorted(os.listdir(frame_path), key=lambda x: int(x.split('.')[0]))
        frame_masks = np.array([np.load(os.path.join(frame_path, mask_name)) for mask_name in frame_masks_name])

        num_masks, H, W = frame_masks.shape

        frame_masks = frame_masks.reshape(num_masks, -1)
        frame_vis = np.zeros((H*W))
        for ind_pixel in range(H*W):

            pix_label = np.unique(frame_masks[:, ind_pixel])
            if pix_label.sum() != 0:
                pix_label=np.sort(pix_label[pix_label != 0])[0]
                frame_vis[ind_pixel] = pix_label

        np.save(os.path.join(out_path, f'{frame_name}.npy'), frame_vis.reshape(H, W))

    

