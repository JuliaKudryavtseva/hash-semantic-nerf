import torch
import open_clip
import time
from tqdm import tqdm
import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2


def get_mask_cropped(image_array, boxes, mask):
    y, x, h, w = boxes
    x, y, w, h = int(x), int(y), int(w), int(h)
    img = mask[..., None]*image_array
    img = img[y:y+h, x:x+w, :]
    return img, mask[y:y+h, x:x+w]


def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    box = np.array([rmin, cmin, (rmax-rmin), (cmax-cmin)])
    return box # y, x, h, w

def fill_zeros(image, mask):
    H, W, C = image.shape
    obj = []
    for i in range(3):
        ch = (1-mask) * image[:, :, i]
        ch =(1-mask) * int(ch.mean())
        ch = mask*image[:, :, i] + ch
        obj.append(ch)

    return np.array(obj).transpose(1, 2, 0)



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

def vis_img(img, percent_zero):
   
    plt.imshow(img)
    plt.title(percent_zero)
    plt.savefig('img.jpg')



def get_mask(mask):

    thresh_gray = (255*mask).astype(dtype=np.uint8) 
    thresh_gray = cv2.dilate(thresh_gray, np.ones((5, 5)), iterations = 1)

    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(thresh_gray)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cv2.drawContours(mask, [cntsSorted[-1]], -1, 1, thickness=cv2.FILLED)

    return mask
    

if __name__ == '__main__':

    DATA_PATH = os.environ['DATA_PATH']
    assert len(DATA_PATH) > 0
    
    image_path = f'data/{DATA_PATH}/images'
    mask_path = f'data/{DATA_PATH}/consistent_masks'
    user_path = os.path.join('data', DATA_PATH, 'user_guid' )
    out_path = os.path.join('data', DATA_PATH, 'base_emb_database', 'embeddings' )


    os.makedirs(out_path, exist_ok=True)


    # === Varuabels ===
    device = 'cpu'
    USER_GUIDE=True
    precise = False

    # relabel dict
    with open(os.path.join(mask_path, 'relabel_dict.json')) as f:
        relabel_dict = json.load(f)

    for new_label in relabel_dict.values():
        if new_label != 0:
            os.makedirs(os.path.join(out_path, str(new_label)), exist_ok=True)
        
    relabel_dict = {int(key):relabel_dict[key] for key in relabel_dict.keys()}
    
    

    # --- Init model ---
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    # clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    clip_model.to(device)

    # load seg dataset info
    frames = sorted(os.listdir(image_path), key=lambda x: int(x.split('_')[1].split('.')[0]))

    START = time.time()
    
    for frame_name in tqdm(frames):
        # load image
        image = Image.open(os.path.join(image_path, frame_name))
        image = np.array(image)

        # load mask
        cons_masks = np.load(os.path.join(mask_path, frame_name.replace('.jpg', '.npy')))
        H, W = cons_masks.shape

        # user guidence
        if USER_GUIDE and frame_name.replace('.jpg', '.npy') in os.listdir(user_path):

            user_masks = np.load(os.path.join(user_path, frame_name.replace('.jpg', '.npy')))
            for ulabel in np.unique(user_masks)[1:]:
                user_mask = (user_masks == ulabel).astype(int)
                cons_masks = (1 - user_mask) * cons_masks
                cons_masks -= ulabel * user_mask
        

        frame_labels = np.unique(cons_masks).astype(int)
        frame_labels = frame_labels[frame_labels!=0]

        for class_label in frame_labels:

            if class_label in relabel_dict.keys():
                # get cropped images
                mask = (cons_masks==class_label).astype(int)

                mask_bbox = bounding_box(mask)
                if precise:
                    mask = get_mask(mask)    

                cropped_image, cropped_mask= get_mask_cropped(image, mask_bbox, mask)
                
                if np.sum(cropped_mask.shape) > 0:

                    # controle 
                    h, w = cropped_mask.shape
                    percent_zeros = np.sum(1-cropped_mask)/(h * w)
                    size = (h * w) / (H * W)

                    
                    if percent_zeros < 0.9:
                        cropped_image = Image.fromarray(cropped_image.astype('uint8'))
                        
                        try:
                            # clip encoding
                            clip_image_input = preprocess(cropped_image).unsqueeze(0)
                        
                            clip_image_input = clip_image_input.to(device)
                            image_features = clip_model.encode_image(clip_image_input).float()
                            image_features /= image_features.norm(dim=-1, keepdim=True)

                            # save results
                            save_name = frame_name.replace('.jpg', '')
                        
                            torch.save(
                                image_features, 
                                os.path.join(out_path, str(relabel_dict[class_label]), f'{save_name}.pt')
                                )


                        except ZeroDivisionError:
                            print('BUG SKIP')
            
    FINAL= time.time()
    spent_time = FINAL-START
    spent_time_min = spent_time/60
    spent_time_h = spent_time_min/60

    print(f"Time: {spent_time} s. , {spent_time_min} min., {spent_time_h} h.")



           
