# hash-semantic-nerf

## preprocessing
```
# segmentation 

docker rmi -f kudryavtseva.prepro_seg && docker build -t kudryavtseva.prepro_seg -f preprocessing/segmentation/Dockerfile  .

docker run --rm --gpus device=5 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/data:/segment-anything/data \
            --name kudryavtseva.prepro_seg \
            kudryavtseva.prepro_seg


# consistancy



docker rmi -f kudryavtseva.squeeze && docker build -t kudryavtseva.squeeze -f preprocessing/consistency/Dockerfile  .


docker run --rm --gpus device=1 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/data:/XMem/data \
            --name kudryavtseva.squeeze \
            kudryavtseva.squeeze 



docker rmi -f kudryavtseva.prepro_const && docker build -t kudryavtseva.prepro_const -f preprocessing/consistency/Dockerfile  .


docker run --rm --gpus device=1 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/data:/XMem/data \
            --name kudryavtseva.prepro_const \
            kudryavtseva.prepro_const 





# clip encoding
docker build -t kudryavtseva.clip_enc -f preprocessing/embedding/Dockerfile  .


docker run -it --rm --gpus device=5 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/data:/clip_emb/data \
            -v $PWD/preprocessing/embedding:/clip_emb \
            --name kudryavtseva.clip_enc \
            kudryavtseva.clip_enc

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
 
docker run -it --rm --gpus device=5  --memory=100gb --shm-size=100gb -p 7087:7087 -e "DATA_PATH=$DATA_PATH" -v $PWD:/workspace -v /home/j.kudryavtseva/.cache/:/home/user/.cache --name kudryavtseva.hash_nerf -u root gbobrovskikh.nerfstudio:dev   


pip install -e . 
ns-install-cli




# train NeRF

ns-train hash-nerf --data data/$DATA_PATH --vis viewer --viewer.websocket-port=7087


# render sam-features # render rgb

ns-render dataset --load-config outputs/$DATA_PATH/sam-nerf/2024-04-08_142627/config.yml --rendered-output-names raw-sam_features  --colormap-options.colormap-min -1 --split test

ns-render dataset --load-config outputs/$DATA_PATH/sam-nerf/2024-04-08_142627/config.yml --rendered-output-names raw-sam_features  --colormap-options.colormap-min -1 --split val

ns-render dataset --load-config outputs/$DATA_PATH/sam-nerf/2024-04-04_144538/config.yml --rendered-output-names depth --split test
ns-render dataset --load-config outputs/$DATA_PATH/sam-nerf/2024-04-08_142627/config.yml --rendered-output-names rgb --split val





# view pre-trained

ns-viewer --load-config outputs/teatime/hash-nerf/2024-05-24_182651/config.yml --viewer.websocket-port=7087


```
