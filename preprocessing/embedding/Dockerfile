FROM ubuntu:22.04

# install app dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install torch numpy pillow tqdm open_clip_torch opencv-python
RUN pip install matplotlib

WORKDIR /clip_emb
