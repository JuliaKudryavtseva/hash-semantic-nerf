import os
import torch 
from tqdm import tqdm

def load_emb(path, name, size=512):
    path = os.path.join(path, name)

    class_emb = []

    for emb in os.listdir(path):
        emb_path = os.path.join(path, emb)
        class_emb.append(
            torch.load(emb_path, map_location=torch.device('cpu'))
        )
    if len(class_emb) > 0:
        class_emb = torch.cat(class_emb).mean(0)
    else:
        class_emb = torch.zeros(size)

    return class_emb.unsqueeze(0)



if __name__ == '__main__':

    DATA_PATH = os.environ['DATA_PATH']

    assert len(DATA_PATH) > 0
    # === Pathes ===
    inp_path = os.path.join('data', DATA_PATH, 'base_emb_database', 'embeddings' )
    out_path = os.path.join('data', DATA_PATH, 'base_emb_database', 'database' )
    os.makedirs(out_path, exist_ok=True)


    class_path = os.listdir(inp_path)
    if 'predictions' in class_path:
        class_path.remove('predictions')

    if 'zero_database' in class_path:
        class_path.remove('zero_database')

    classes_path = sorted(class_path, key=lambda x: int(x))
    # average embeddings
    dataset_emb = torch.cat([load_emb(inp_path, class_label) for class_label in tqdm(classes_path)], dim=0)
    
    # save
    torch.save(dataset_emb, os.path.join(out_path, f'dataset_emb.pt'))
