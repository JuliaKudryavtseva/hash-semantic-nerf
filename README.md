# hash-semantic-nerf

## preprocessing


### segmentation 
```
docker rmi -f kudryavtseva.prepro_seg && docker build -t kudryavtseva.prepro_seg -f preprocessing/segmentation/Dockerfile  .


docker run --rm --gpus device=4 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/data:/segment-anything/data \
            --name kudryavtseva.prepro_seg \
            kudryavtseva.prepro_seg
```




### consistancy
```
docker build -t kudryavtseva.prepro_const -f preprocessing/consistency/Dockerfile  .


docker run -it --rm --gpus device=4 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/XMem:/XMem \
            --name kudryavtseva.prepro_const_dozer \
            kudryavtseva.prepro_const
```


### clip encoding
```
docker rmi -f kudryavtseva.clip_enc && docker build -t kudryavtseva.clip_enc -f preprocessing/embedding/Dockerfile  .


docker run -it --rm --gpus device=5 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/data:/clip_emb/data \
            -v $PWD/preprocessing/embedding:/clip_emb \
            --name kudryavtseva.clip_enc \
            kudryavtseva.clip_enc

python3 create_database.py && python3 create_dataset_emb.py

python3 make_pred.py

```




## Hash-NeRF
Create image

```
# docker image
docker build --tag kudryavtseva/nerfstudio:version1 -f Dockerfile .

# docker container
docker run -it --rm --gpus device=5 \
            --memory=100gb --shm-size=100gb \
            -p 7087:7087 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD:/workspace \
            -v /home/j.kudryavtseva/.cache/:/home/user/.cache \
            --name kudryavtseva.hash_nerf_waldo \
            -u root gbobrovskikh.nerfstudio:dev   

# register model
pip install -e . 
ns-install-cli


# train NeRF
ns-train hash-nerf --data data/$DATA_PATH --vis viewer --viewer.websocket-port=7087

# view NeRF
ns-viewer --load-config $PATH_TO_CONFIG --viewer.websocket-port=7087

ns-viewer --load-config outputs/teatime/hash-nerf/2024-05-28_075026/config.yml --viewer.websocket-port=7087

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

# mesh
ns-export pointcloud --load-config outputs/dozer_nerfgun_waldo/hash-nerf/2024-05-27_211212/config.yml \ 
            --output-dir exports/dozer/ \
            --num-points 10000  \
            --remove-outliers True \
            --normal-method open3d \
            --use-bounding-box True \
            --bounding-box-min -1 -1 -1 --bounding-box-max 1 1 1

```

