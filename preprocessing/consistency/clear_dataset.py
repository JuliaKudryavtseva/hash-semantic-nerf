import numpy as np 
import os
from tqdm import tqdm


def get_mask_occ_once(path, frame_name):
    # how many masks in dataset occure once
    all_pairs = []
    for name in frame_name:
        frame_path = os.path.join(path, name)
        frame_number = int(name.split('_')[1])
        all_pairs.extend(
            [(int(m.split('.')[0]), frame_number) for m in os.listdir(frame_path)]
            )

    all_pairs = np.array(all_pairs)
    all_pairs = all_pairs[all_pairs[:, 0].argsort()]
    
    all_classes = np.unique(all_pairs[:, 0]).shape[0]
    occure_once = 0
    result = []
    for i in np.unique(all_pairs[:, 0]):
        mask = all_pairs[:, 0]==i
        number_occ = np.sum(mask)
        if number_occ == 1:
            occure_once += 1
            mask_number, frame_ind = all_pairs[mask][0]
            result.append([f'frame_{str(frame_ind).zfill(5)}', f'{mask_number}.npy'])
    return result


def get_relabel(path, frame_names):
    all_pairs = []
    for name in frame_names:
        frame_path = os.path.join(path, name)
        frame_number = int(name.split('_')[1])
        all_pairs.extend([int(m.split('.')[0]) for m in os.listdir(frame_path)])

    label_classes = np.unique(np.array(all_pairs))
    relabel_map = {}
    new_classes = np.arange(1, len(label_classes)+1, 1,  dtype=int)
    for ind, i in enumerate(label_classes):
        relabel_map[i] = new_classes[ind]
    
    return relabel_map

if __name__ == '__main__':
    path = 'data/teatime/consistent_masks'
    frame_names = os.listdir(path)
    frame_names.remove('logs.txt')
    if "label_map_colors.pkl" in frame_names:
        frame_names.remove('label_map_colors.pkl')
    if "ready_masks" in frame_names:
        frame_names.remove('ready_masks')
    
    # DELETE 
    for pair in get_mask_occ_once(path, frame_names):
        frame_name, mask_name = pair
        os.remove(os.path.join(path, frame_name, mask_name))

        
    # # RENAME
    frame_names =  sorted(frame_names, key=lambda x: int(x.split('_')[1]))
    relabel_map = get_relabel(path, frame_names)
    for name in tqdm(frame_names):
        frame_path = os.path.join(path, name)
        frame_labels = sorted(os.listdir(frame_path), key=lambda x: int(x.split('.')[0]))
        for old_label in frame_labels:

            new_label = relabel_map[int(old_label.split('.')[0])]

            old_name = os.path.join(frame_path, f"{old_label}")
            new_name = os.path.join(frame_path, f"{new_label}.npy")
            

            # relabel
            os.rename(old_name, new_name)

            mask = np.load(new_name)
            mask = int(new_label) * (mask > 0).astype(float)
            np.save(new_name, mask)