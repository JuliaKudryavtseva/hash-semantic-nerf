import os
import gc
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
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 100000,
}


def read_masks(frame_name, data_info):
  result=[]
  for mask_info in data_info[frame_name]:
    mask = np.load(os.path.join(seg_path, mask_info['segmentation']))
    result.append(mask.astype(int))
  return np.array(result)
  




def get_labels(counter, num_masks_in_frame):
  return np.arange(counter+1, counter+1+num_masks_in_frame)[..., None, None], counter+num_masks_in_frame


def find_curr_mask_on_frames(processor, start_frame_ind, end_frame_ind, inp_path, frame_names, mask, num_objects):
  '''
  Input: 
    processor (torch model): Xmem model
    inp_path (str): path to video frames
    frame_names (list of str): names of file with .jpg frame 
    mask (np.array): mask of current object

  Output:
    current_masks (list): masks of objects as np array 
  '''


  frame_names = frame_names[start_frame_ind:end_frame_ind]

  current_masks = []
  frames_to_propagate = len(frame_names)

  with torch.cuda.amp.autocast(enabled=True):
    for current_frame_index, frame_name in enumerate(tqdm(frame_names)):
      # read frame              
      frame =Image.open(os.path.join(inp_path, frame_name))
      frame = frame.convert('RGB')
      frame = np.array(frame)

      # convert numpy array to pytorch tensor format

      frame_torch, _ = image_to_torch(frame, device=device)
      if current_frame_index == 0:
        # initialize with the mask
        mask_torch = index_numpy_to_one_hot_torch(mask[0], num_objects+1).to(device)

        # the background mask is not fed into the model
        prediction = processor.step(frame_torch, mask_torch[1:])
      else:
        # propagate only
        prediction = processor.step(frame_torch)

      # argmax, convert to numpy
      prediction = torch_prob_to_numpy_mask(prediction)

      # add for seq. of masks
      current_masks.append(prediction)

  return current_masks




def process_frame_masks(first_frame_masks, start_frame_ind,  end_frame_ind, network, num_objects):
  print(' === Process set of masks with Xmem === ')
  final_masks=[]
  for label, init_mask in enumerate(first_frame_masks, 1):
    mask = init_mask[None, ...]

    torch.cuda.empty_cache()

    # init mtracking model
    processor = InferenceCore(network, config=config)
    processor.set_all_labels(range(1, num_objects+1)) # consecutive labels

    # get sequence of masks
    current_masks = find_curr_mask_on_frames(processor, start_frame_ind, end_frame_ind, inp_path, frame_names, mask, num_objects)    
    final_masks.append(current_masks)

    del processor
    del mask
    del current_masks
    gc.collect()

  final_masks = np.array(final_masks)  # s, 180, h, w 
  final_masks = np.transpose(final_masks, (1, 0, 2, 3)) # 180, s, h, w 
  return final_masks # 180, h, w 


def make_masks_consist(results, init_masks, curr_frame_ind, frame_names, labels: np.array, TRASHOLD=0.7):
  print('Make current masks consistent')

  for i in tqdm(range(results.shape[0])):

    n_xmm, n_sam = results[i, ...].shape[0], init_masks[curr_frame_ind+i].shape[0]

    if n_xmm > 0 and n_sam > 0:
      IOU_matrix = cupy_get_IOU(results[i, ...], init_masks[curr_frame_ind+i])

      sam_masks_ind  = np.argmax(IOU_matrix, axis=1)
      ind_bool_mask = np.max(IOU_matrix, axis=1) > TRASHOLD
      cons_masks = labels*init_masks[curr_frame_ind+i][sam_masks_ind]
      cons_masks = cons_masks[ind_bool_mask] # number_masks, H, W

      # save cons_masks
      for con_ind in range(cons_masks.shape[0]):
        save_index = np.unique(cons_masks[con_ind])[1]
        np.save(
          os.path.join(out_path, frame_names[curr_frame_ind+i].replace('.jpg', ''), f'{save_index}.npy'), 
          cons_masks[con_ind]
          )

      # remove from 
      remain_ind = np.setdiff1d(np.arange(IOU_matrix.shape[1]), sam_masks_ind[ind_bool_mask])
      init_masks[curr_frame_ind+i] = init_masks[curr_frame_ind+i][remain_ind]

  return init_masks


def save_masks(process_masks, result, out_path, frame_names, COUNTER, num_objects, start):
  print(' === post-processing masks === ')
  n_frames, _, H, W = result.shape

  for frame_number in tqdm(range(n_frames)):
    # processsing
    # TO SPEEED UP
    frame_masks = result[frame_number, 0, ...]
    frame_masks = index_numpy_to_one_hot_torch(frame_masks, num_objects+1)
    frame_masks = frame_masks[1:] * np.arange(1+COUNTER, num_objects+1+COUNTER)[:, None, None] # num_class, H, W
    frame_masks = frame_masks.sum(0)

    # saving
    process_masks[os.path.join(out_path, frame_names[start+frame_number].replace('.jpg', '.npy'))].append(frame_masks)
    del frame_masks
    
  gc.collect() # save memory
  COUNTER += num_objects
  return COUNTER, process_masks



def post_processing(masks: list, size: tuple, take: int):
    H, W = size
    init_masks = np.concatenate(masks)
    init_masks = init_masks.reshape(-1, H * W )

    save = np.zeros(H * W)

    for i in range(H * W):
      # values = np.unique(init_masks[:, i])
      # if len(values) > 1:
      #       save[i] = values[1]
      #   else:
      #       save[i] = values[0]

        values = np.unique(init_masks[:, i][take:])
        if len(values) > 1:
            save[i] = values[1]
        else:
            save[i] = values[0]
    save = save.reshape(H, W)
    return save


def squeezig(results, window, size):
  print(" === Squeezing === ")
  
  take = 0
  for ind, path in enumerate(tqdm(results), 1):
    ready_mask = post_processing(results[path], size, take)
    np.save(path, ready_mask)
    # if ind % window == 0:
    #   take += window


if __name__ == '__main__':

  DATA_PATH = os.environ['DATA_PATH']
  assert len(DATA_PATH) > 0

  # === Paths ===
  inp_path = os.path.join('data', DATA_PATH, 'images')
  out_path = os.path.join('data', DATA_PATH, 'window60_consistent_masks')
  seg_path = os.path.join('data', DATA_PATH, 'segmentation_results')
  logs_path = os.path.join(out_path, 'logs.txt')

  os.makedirs(out_path, exist_ok=True)

  # === Variables ===
  device = 'cuda'
  COUNTER = 0
  window = 60 # 60 for dozer
  print(f'Input: {inp_path} , Output {out_path}')
  print(f' === window {window} === ')
  set_masks = {}
  
  # images name
  frame_names = sorted(os.listdir(inp_path), key=lambda x: int(x.split('.')[0].split('_')[1]))


  # # create output folder
  for frame_name in frame_names:
    set_masks[os.path.join(out_path, frame_name.replace('.jpg', '.npy'))] = []
  #   os.makedirs(os.path.join(out_path, frame_name.replace('.jpg', '')), exist_ok=True)

  # dataset info
  with open(os.path.join(seg_path, f'{DATA_PATH}.json')) as f:
    data_info = json.load(f)


  print(' === Load Dataset === ')
  # load segmentation masks
  names = sorted(os.listdir(os.path.join(seg_path, 'numpy_masks')), key=lambda x: int(x.split('_')[1]))
  init_masks = [read_masks(frame_name+'.jpg', data_info) for frame_name in tqdm(names)]
  _, H, W = init_masks[0].shape
  
  

  # === Init model ===
  network = XMem(config, './saves/XMem.pth').eval().to(device)

  
  # process masks from first set
  START_TIME = time.time()
  print('\n === Start ===\n')
  with open(logs_path, 'a') as logs:
    logs.write('=== Start ===\n')

  stop_ind = window
  for i in range(len(init_masks)):
    print(f'Frame: {i}')
    torch.cuda.empty_cache()
    gc.collect()

    num_objects = init_masks[i].shape[0]
    if num_objects > 0:
      # process masks
      frame_masks = init_masks[i] * np.arange(1, num_objects+1)[:, None, None]
      frame_masks = frame_masks.max(0)[None, ...]
      # consistancy
      results = process_frame_masks(frame_masks, i, stop_ind, network, num_objects=num_objects) # num_frames, 1, H, W

      # saving
      COUNTER, set_masks = save_masks(set_masks, results, out_path, frame_names, COUNTER, num_objects, i)

      if (i+1) % window == 0:
        stop_ind+=window

      # logs
      with open(logs_path, 'a') as logs:
        logs.write(f'Frame: {i}\n')

      with open(logs_path, 'a') as logs:
        logs.write(f'{i}: {stop_ind}\n')
    
    print()

    
  # final processig
  squeezig(set_masks, window, size=(H, W))
  print()

  
      
  
  print(' === Finish ===\n')
  FINISH_TIME = time.time()
  time_spent = FINISH_TIME-START_TIME
  time_spent_min = time_spent/60
  time_spent_hour = time_spent_min/60

  with open(logs_path, 'a') as logs:
      logs.write(f'TIME in {time_spent} s., in {time_spent_min} min., in {time_spent_hour} h.')
      logs.write(f'\n === Finish ===\n')