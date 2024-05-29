# hash-semantic-nerf

## preprocessing
```
# segmentation 

docker rmi -f kudryavtseva.prepro_seg && docker build -t kudryavtseva.prepro_seg -f preprocessing/segmentation/Dockerfile  .


docker run --rm --gpus device=4 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/data:/segment-anything/data \
            --name kudryavtseva.prepro_seg \
            kudryavtseva.prepro_seg





# consistancy

docker rmi -f kudryavtseva.prepro_const && docker build -t kudryavtseva.prepro_const -f preprocessing/consistency/Dockerfile  .


docker run -it --rm --gpus device=4 \
            --memory=480gb --shm-size=480gb \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/XMem:/XMem \
            --name kudryavtseva.prepro_const_dozer \
            kudryavtseva.prepro_const



docker rmi -f kudryavtseva.prepro_squeeze && docker build -t kudryavtseva.prepro_squeeze -f preprocessing/consistency/Dockerfile  .

docker run  --rm --gpus device=4 \
            --memory=400gb --shm-size=400gb \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/data:/XMem/data \
            --name kudryavtseva.prepro_get_masks \
            kudryavtseva.prepro_squeeze



# clip encoding

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
docker build -t kudryavtseva.preprocessing -f preprocessing/Dockerfile  .


docker run -it --rm --gpus device=5 \
            -v $PWD/preprocessing:/workspace \
            --name kudryavtseva.preprocessing \
            kudryavtseva.preprocessing



### Hash-NeRF
Create image

```
docker build --tag kudryavtseva/nerfstudio:version1 -f Dockerfile .
```

Docker container 

```

docker run -it --rm --gpus device=2  --memory=100gb --shm-size=100gb -p 7087:7087 -e "DATA_PATH=$DATA_PATH" -v $PWD:/workspace -v /home/j.kudryavtseva/.cache/:/home/user/.cache --name kudryavtseva.hash_nerf_waldo -u root gbobrovskikh.nerfstudio:dev   


pip install -e . 
ns-install-cli




# train NeRF

ns-train hash-nerf --data data/$DATA_PATH --vis viewer --viewer.websocket-port=7087




teatime: 402
dozer: 1211
kitchen: 340


ns-render dataset --load-config outputs/waldo_kitchen/hash-nerf/2024-05-28_125744/config.yml --rendered-output-names raw-pred_labels --colormap-options.colormap-min -1  --colormap-options.colormap-max 340 --split test









# render sam-features # render rgb

ns-render dataset --load-config outputs/$DATA_PATH/sam-nerf/2024-04-08_142627/config.yml --rendered-output-names raw-sam_features  --colormap-options.colormap-min -1 --split test

ns-render dataset --load-config outputs/$DATA_PATH/sam-nerf/2024-04-08_142627/config.yml --rendered-output-names raw-sam_features  --colormap-options.colormap-min -1 --split val

ns-render dataset --load-config outputs/teatime/hash-nerf/2024-05-25_085136/config.yml --rendered-output-names pred_labels --split test




ns-render dataset --load-config outputs/$DATA_PATH/sam-nerf/2024-04-08_142627/config.yml --rendered-output-names rgb --split val





# view pre-trained

ns-viewer --load-config outputs/teatime/hash-nerf/2024-05-28_075026/config.yml --viewer.websocket-port=7087


```
