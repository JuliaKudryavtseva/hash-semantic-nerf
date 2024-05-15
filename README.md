# hash-semantic-nerf

## preprocessing
```
# segmentation 

docker build -t kudryavtseva.prepro_seg -f preprocessing/segment/Dockerfile  .

docker run --rm --gpus device=5 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/data:/segment-anything/data \
            --name kudryavtseva.prepro_seg \
            kudryavtseva.prepro_seg


# consistancy

docker build -t kudryavtseva.prepro_const -f preprocessing/consistency/Dockerfile  .

docker run --rm --gpus device=1 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/data:/XMem/data \
            --name kudryavtseva.prepro_const \
            kudryavtseva.prepro_const



docker run --rm --gpus device=1 \
            -e "DATA_PATH=$DATA_PATH" \
            -v $PWD/preprocessing/consistency/XMem:/XMem \
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


# nerf


docker run -it --name kudryavtseva.nerf_default --memory=50gb --shm-size=50gb  --rm --gpus device=5 -v $PWD:/workspace -v /home/j.kudryavtseva/cache/:/home/user/.cache/ -p 7087:7087 gbobrovskikh.nerfstudio:dev



ns-train nerfacto --data  teatime --viewer.websocket-port=7087

ns-viewer --load-config outputs/teatime/nerfacto/2024-04-20_164423/config.yml --viewer.websocket-port=7087


