# Hash-Semantic-NeRF


Training     
:-------------------------:
![](https://github.com/JuliaKudryavtseva/hash-semantic-nerf/blob/main/assets/training.png)  
## Installation
0. Install Nerfstudio dependencies including "tinycudann" to install dependencies 
1. Clone this git repo
```
git clone https://github.com/JuliaKudryavtseva/hash-semantic-nerf.git
```
2. Create folder "data", put the dataset in LERF format there and create the variable with dataset name:
```
mkdir data
DATA_PATH=teatime

data/teatime
├─ images               (folder for images)
│  ├─ frame_00001.jpg   (image 1)
│  ├─ frame_00002.jpg   (image 2)
│  ├─ frame_00003.jpg   (image 3)    
│  ...
├─ transforms.json
...

```

## Preprocessing
### Segmentation 
```
# create image
docker build -t kudryavtseva.prepro_seg -f preprocessing/segmentation/Dockerfile  .

# run segmentation phase
docker run --rm --gpus device=0 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/data:/segment-anything/data \
            --name kudryavtseva.prepro_seg \
            kudryavtseva.prepro_seg
```




### Consistancy
```
# create image
docker build -t kudryavtseva.prepro_const -f preprocessing/consistency/Dockerfile  .

# run consistancy phase
docker run -it --rm --gpus device=0 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/XMem:/XMem \
            --name kudryavtseva.prepro_const_dozer \
            kudryavtseva.prepro_const
```


### CLIP embeddings of images
```
# create image
docker build -t kudryavtseva.clip_enc -f preprocessing/embedding/Dockerfile  .

# create container
docker run -it --rm --gpus device=0 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/data:/clip_emb/data \
            -v $PWD/preprocessing/embedding:/clip_emb \
            --name kudryavtseva.clip_enc \
            kudryavtseva.clip_enc

# create database with image-langaude embeddings
python3 create_database.py && python3 create_dataset_emb.py

```

### Generate masks

```
python3 make_pred.py --text-prompt "plate"

```




## Hash-NeRF

```
# docker image
docker build --tag kudryavtseva/nerfstudio:version1 -f Dockerfile .

# docker container
docker run -it --rm --gpus device=0 \
            --memory=100gb --shm-size=100gb \
            -p 7087:7087 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD:/workspace \
            -v /home/j.kudryavtseva/.cache/:/home/user/.cache \
            --name kudryavtseva.hash_nerf \
            -u root gbobrovskikh.nerfstudio:dev   

# register model
pip install -e . 
ns-install-cli


# train NeRF
ns-train hash-nerf --data data/$DATA_PATH --vis viewer --viewer.websocket-port=7087

# view NeRF
ns-viewer --load-config $PATH_TO_CONFIG --viewer.websocket-port=7087


# render predicitons

==============
LABEL_DIM:
--------------
teatime: 402
dozer: 1211
kitchen: 340
==============

ns-render dataset --load-config outputs/waldo_kitchen/hash-nerf/2024-05-28_125744/config.yml \
            --rendered-output-names raw-pred_labels \
            --colormap-options.colormap-min -1  \
            --colormap-options.colormap-max $LABEL_DIM \
            --split test


ns-render dataset --load-config outputs/waldo_kitchen/hash-nerf/2024-05-28_125744/config.yml \
            --rendered-output-names pred_labels \
            --split test

# mesh
ns-export pointcloud --load-config outputs/dozer_nerfgun_waldo/hash-nerf/2024-05-27_211212/config.yml \ 
            --output-dir exports/dozer/ \
            --num-points 10000  \
            --remove-outliers True \
            --normal-method open3d \
            --use-bounding-box True \
            --bounding-box-min -1 -1 -1 --bounding-box-max 1 1 1

```

