import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore

from progressbar import progressbar

import cv2
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis
from tqdm import tqdm
import json
import time

from multi_iou import cupy_get_IOU


torch.set_grad_enabled(False)


# default configuration
config = {
    'top_k': 30,
    'mem_every': 10,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 50,
    'max_mid_term_frames': 100,
    'max_long_term_elements': 10000,
}


def read_masks(frame_name, data_info):
  result=[]
  for mask_info in data_info[frame_name]:
    mask = np.load(os.path.join(seg_path, mask_info['segmentation']))
    result.append(mask.astype(int))
  return np.array(result)

def get_labels(counter, num_masks_in_frame):
  return np.arange(counter+1, counter+1+num_masks_in_frame)[..., None, None], counter+num_masks_in_frame


def find_curr_mask_on_frames(processor, inp_path, frame_names, mask):
  '''
  Input: 
    processor (torch model): Xmem model
    inp_path (str): path to video frames
    frame_names (list of str): names of file with .jpg frame 
    mask (np.array): mask of current object

  Output:
    current_masks (list): masks of objects as np array 
  '''

  current_masks = []
  frames_to_propagate = len(frame_names)

  with torch.cuda.amp.autocast(enabled=True):
    for current_frame_index, frame_name in enumerate((frame_names)):
      # read frame              
      frame =Image.open(os.path.join(inp_path, frame_name))
      frame = frame.convert('RGB')
      frame = np.array(frame)

      # convert numpy array to pytorch tensor format
      frame_torch, _ = image_to_torch(frame, device=device)
      if current_frame_index == 0:
        # initialize with the mask
        mask_torch = torch.tensor(mask).to(device)
        # the background mask is not fed into the model
        prediction = processor.step(frame_torch, mask_torch)
      else:
        # propagate only
        prediction = processor.step(frame_torch)

      # argmax, convert to numpy
      prediction = torch_prob_to_numpy_mask(prediction)

      # add for seq. of masks
      current_masks.append(prediction)

  return current_masks




def process_frame_masks(first_frame_masks, network):
  print(' === Process set of masks with Xmem === ')
  final_masks=[]
  for label, init_mask in enumerate(tqdm(first_frame_masks), 1):

    mask = init_mask[None, ...]

    torch.cuda.empty_cache()

    # init mtracking model
    processor = InferenceCore(network, config=config)
    processor.set_all_labels(range(1, num_objects+1)) # consecutive labels

    # get sequence of masks
    current_masks = find_curr_mask_on_frames(processor, inp_path, frame_names, mask)    
    final_masks.append(current_masks)

  final_masks = np.array(final_masks)  # s, 180, h, w 
  final_masks = np.transpose(final_masks, (1, 0, 2, 3)) # 180, s, h, w 
  return final_masks


def make_masks_consist(results, init_masks, frame_names, labels: np.array, TRASHOLD=0.7):
  print('Make current masks consistent')

  for i in tqdm(range(results.shape[0])):
    IOU_matrix = cupy_get_IOU(results[i, ...], init_masks[i])

    sam_masks_ind  = np.argmax(IOU_matrix, axis=1)
    ind_bool_mask = np.max(IOU_matrix, axis=1) > TRASHOLD
    cons_masks = labels*init_masks[i][sam_masks_ind]
    cons_masks = cons_masks[ind_bool_mask] # number_masks, H, W

    # save cons_masks
    for con_ind in range(cons_masks.shape[0]):
      save_index = np.unique(cons_masks[con_ind])[1]
      np.save(
        os.path.join(out_path, frame_names[i].replace('.jpg', ''), f'{save_index}.npy'), 
        cons_masks[con_ind]
        )

    # remove from 
    remain_ind = np.setdiff1d(np.arange(IOU_matrix.shape[1]), sam_masks_ind[ind_bool_mask])
    init_masks[i] = init_masks[i][remain_ind]

  return init_masks


if __name__ == '__main__':

  DATA_PATH = os.environ['DATA_PATH']
  assert len(DATA_PATH) > 0

  # === Paths ===
  inp_path = os.path.join('data', DATA_PATH, 'images')
  out_path = os.path.join('data', DATA_PATH, 'consistent_masks')
  seg_path = os.path.join('data', DATA_PATH, 'segmentation_results')

  os.makedirs(out_path, exist_ok=True)

  # === Variables ===
  device = 'cuda'
  num_objects = 1
  COUNTER = 0

  # images name
  frame_names = sorted(os.listdir(inp_path), key=lambda x: int(x.split('.')[0].split('_')[1]))

  # create output folder
  for frame_name in frame_names:
    os.makedirs(os.path.join(out_path, frame_name.replace('.jpg', '')), exist_ok=True)

  # dataset info
  with open(os.path.join(seg_path, f'{DATA_PATH}.json')) as f:
    data_info = json.load(f)


  print(' === Load Dataset === ')
  # load segmentation masks
  names = sorted(os.listdir(os.path.join(seg_path, 'numpy_masks')), key=lambda x: int(x.split('_')[1]))
  init_masks = [read_masks(frame_name+'.jpg', data_info) for frame_name in tqdm(names)]
  

  # === Init model ===
  network = XMem(config, './saves/XMem.pth').eval().to(device)

  # process masks from first set
  print(' === Start ===\n')

  for i in range(len(init_masks)):
    print(f'Frame: {i}')

    results = process_frame_masks(init_masks[i], network) # num_frames, num_mask, H, W
    # make frame consistant with IOU metrics
    labels, COUNTER = get_labels(COUNTER, results.shape[1])
    init_masks = make_masks_consist(results, init_masks, frame_names, labels)

    print()


  
  print(' === Finish ===\n')

  
  # for i in tqdm(range(180)):
  #   # np.save(os.path.join(out_path, f'masks_frame_{str(i+1).zfill(5)}.npy'), np.array(results[i]))

  #   frame_masks = 255*np.array(del_results[i]).mean(0)
  #   plt.imshow(frame_masks)
  #   plt.savefig(f"trash/frame_{str(i+1).zfill(5)}.jpg")
