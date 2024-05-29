import numpy as np
import os
from tqdm import tqdm
import pickle
import random
import colorsys
from PIL import Image
import matplotlib.pyplot as plt
import gc
import cv2
import json


def get_number_mask(path, frame_name):
    all_pairs = []
    for name in frame_name:
        frame_path = os.path.join(path, name)
        
        frame_number = int(name.split('_')[1])
        all_pairs.extend([int(m.split('.')[0]) for m in os.listdir(frame_path)])

    label_classes = np.unique(np.array(all_pairs))
    # weights  
    all_pairs = np.array(all_pairs)
    all_pairs = all_pairs[all_pairs[:, 0].argsort()]
    
    all_classes = np.unique(all_pairs[:, 0]).shape[0]
    occure_once = 0
    occurance = [100]
    for i in np.unique(all_pairs[:, 0]):
        mask = all_pairs[:, 0]==i
        number_occ = np.sum(mask)
        occurance.append(number_occ)
        

    return label_classes[-1] , 1/np.array(occurance) # number classs, weights loss

def random_colors(N):
    random_color=np.random.choice(range(255), size=(N, 3))
    random_color[0] = np.zeros(3)
    return random_color

def get_new_mask(hash_masks, size, relabel_dict):
    H, W = size
    hash_masks = hash_masks.reshape(-1)
    for i in range(H*W):
        old_label = int(hash_masks[i])
        if old_label in relabel_dict.keys():
            hash_masks[i] = relabel_dict[old_label]
        else:
            hash_masks[i] = 0
    return hash_masks.reshape(H, W)

def get_color_img(hash_masks, colors, size):
    result = colors[hash_masks.reshape(-1).astype(int)].reshape( *size, 3)
    return result


if __name__ == '__main__':

    DATA_PATH = os.environ['DATA_PATH']
    assert len(DATA_PATH) > 0

    USER_GUIDE = False


    inp_path = os.path.join('data', DATA_PATH, 'consistent_masks')
    out_path = os.path.join('data', DATA_PATH, 'consistent_masks', 'ready_masks')
    color_out_path = os.path.join('data', DATA_PATH, 'vis')
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(color_out_path, exist_ok=True)

    if USER_GUIDE:
        user_guide_path  = os.path.join('data', DATA_PATH, 'user_guid')


    frame_names = os.listdir(inp_path) 
    frame_names.remove('ready_masks')
    frame_names.remove('logs.txt')
    if "number_class.npy" in frame_names:
        frame_names.remove('number_class.npy')
    if 'label_map_colors.pkl' in frame_names:
        frame_names.remove('label_map_colors.pkl')
    if 'weights.npy' in frame_names:
        frame_names.remove('weights.npy')
    if 'relabel_dict.json' in frame_names:
        frame_names.remove('relabel_dict.json')

    frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[1].split('.')[0]))
    

    print(' === Additional postprocessing === ')
    print(' :} ')

    dataset_labels = []
    dataset = []
    for frame_name in tqdm(frame_names):
        # load masks
        frame_path = os.path.join(inp_path, frame_name)
        hash_masks = np.load(frame_path)

        H, W = hash_masks.shape
        unique_labels, counts = np.unique(hash_masks, return_counts=True)

        # remove reare classes
        occ = counts/(H*W)
        unique_labels = unique_labels[occ > 1e-3].astype(int)

        result = []
        for label in unique_labels:
            mask = (hash_masks==label).astype(dtype=np.uint8) 
            result.append(label*mask)

        hash_masks = np.array(result).sum(0).astype(int)


        if USER_GUIDE and frame_name in os.listdir(user_guide_path):
            user_frame = os.path.join(user_guide_path, frame_name) 
            user_masks = np.load(user_frame)

            for ulabel in np.unique(user_masks)[1:]:
                user_mask = (user_masks == ulabel).astype(int)
                hash_masks = (1 - user_mask) * hash_masks
                hash_masks -= ulabel * user_mask
                print(ulabel, np.unique(hash_masks)[:10])


        dataset.append(hash_masks)
        dataset_labels.extend(np.unique(hash_masks).tolist())
        gc.collect()


    print(' === Relabeling === ')
    unique_labels, counts = np.unique(dataset_labels, return_counts=True)

    if USER_GUIDE:
        counts[:2]=int(np.mean(counts))
        # counts[:new_ulab]= 3

    occ_zeros = int(counts[unique_labels == 0])
    counts = counts[unique_labels != 0]
    unique_labels = unique_labels[unique_labels != 0]

    unique_labels = unique_labels[counts>2]
    counts = counts[counts>2]
    
    
    new_labels = list(range(1, len(unique_labels)+1)) # new labels for dataset
    

    # number masks
    number_masks = new_labels[-1]
    print(f'Number classes: {number_masks} , {counts.shape}')
    np.save(os.path.join(inp_path, 'number_class.npy'), number_masks)

    counts = np.insert(counts, 0, occ_zeros)
    np.save(os.path.join(inp_path, 'weights.npy'), 1/counts)


    # for all dataset
    relabel_dict = {old_lab:new_lab for (old_lab, new_lab) in zip(unique_labels, new_labels)}
    relabel_dict[0] = 0

    
    with open(os.path.join(inp_path, f'relabel_dict.json') , 'w', encoding='utf-8') as f:
        json.dump({str(k): relabel_dict[k] for k in relabel_dict}, f, ensure_ascii=False, indent=4)

        
    # colors for all dataset
    colors = random_colors(len(new_labels)+5)
    for ind, frame_hash_masks in enumerate(tqdm(dataset)):
        hash_masks = get_new_mask(frame_hash_masks, (H, W), relabel_dict)

        np.save(os.path.join(out_path, frame_names[ind]), hash_masks)


        # COLORS
        color_res = get_color_img(hash_masks, colors, (H, W))
        im = Image.fromarray(np.uint8(color_res))
        im.save(os.path.join(color_out_path, frame_names[ind].replace('.npy', '.jpg')))
        
