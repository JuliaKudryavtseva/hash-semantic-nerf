import torch
import open_clip
import time
from tqdm import tqdm
import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gzip
import cv2


def vis(mask_bbox, mask):
    # ======================================================    
    y_min, x_min, box_height, box_width = mask_bbox          
    fig, ax = plt.subplots()
    ax.imshow(mask)
    ax.axis('off')
    rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    plt.savefig('clip.jpg')
    # ======================================================

def vis_img(img):
    plt.imshow(img)
    plt.savefig('img.jpg')

def load_npy_gz(filename):
    # Open the .npy.gz file
    with gzip.open(filename, 'rb') as f:
        # Load the numpy array
        array = np.load(f)
    return array


def get_mask(mask):

    thresh_gray = (255*mask).astype(dtype=np.uint8) 
    thresh_gray = cv2.dilate(thresh_gray, np.ones((5, 5)), iterations = 1)

    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(thresh_gray)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(mask, [cntsSorted[-1]], -1, 1, thickness=cv2.FILLED)

    return mask


# parsing args
def parse_args():

    parser = argparse.ArgumentParser(description ='args for algorithm which makes frame consistant')

    parser.add_argument('--text-prompt', type=str, default='teddy bear',  help='Text prompt for masks predicting masks.')
    return parser.parse_args()

if __name__ == '__main__':

    DATA_PATH = os.environ['DATA_PATH']
    assert len(DATA_PATH) > 0

    args = parse_args()

    TEXT_PROMPT = args.text_prompt


    nerf_path = os.path.join('data', 'nerf_prediction', DATA_PATH, 'raw-pred_labels')
    database_path = os.path.join('data', DATA_PATH, 'base_emb_database', 'database', 'dataset_emb.pt')
    out_path = os.path.join('data', DATA_PATH, 'base_emb_database', 'predictions', TEXT_PROMPT.replace(' ', '_'))
    vis_path = os.path.join('data', DATA_PATH, 'base_emb_database', 'vis', TEXT_PROMPT.replace(' ', '_'))

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(vis_path, exist_ok=True)

    # Varubels
    device = 'cpu'


    # nerf_frames = sorted(os.listdir(nerf_path), lambda x: int(x.split('.')[0].split('_')[1]))
    nerf_frames = os.listdir(nerf_path)



    # --- Init model ---
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    clip_model.to(device)

    # get text embeddings
    text_tokens = open_clip.tokenize(TEXT_PROMPT)
    text_features = clip_model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # get image class embeddings
    database = torch.load(database_path, map_location='cpu').detach().numpy() # [num_class, emb_size]


    # get similarity
    similarity = text_features.detach().cpu().numpy() @ database.T
    similarity = similarity[0]


    # maske predictions
    label1 = np.argmax(similarity) + 1

    # get similar labels
    similarity_emb = database @ database[label1-1].T   # [num_class, emb_size] *[emb_size, 1]
    similarity_emb[label1-1] = 0
    label2 = np.argmax(similarity_emb)+1


    for frame_name in tqdm(nerf_frames):

        # load nerf predicions
        frame_path = os.path.join(nerf_path, frame_name)
        nerf_masks = load_npy_gz(frame_path)
        nerf_masks = nerf_masks[:, :, 0]

        
        mask1 = (nerf_masks == label1).astype(int)
        mask2 = (nerf_masks == label2).astype(int)
        mask = np.clip(mask1+mask2, 0, 1)

    
        # save
        np.save(os.path.join(out_path, frame_name.replace('.gz', '')), mask)
        
        # show
        plt.imshow(mask)
        plt.savefig(os.path.join(vis_path, frame_name.replace('.npy.gz', '.jpg')))
        

