FROM ubuntu:22.04

# install app dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 
# RUN apt install nvidia-cuda-toolkit -y
# RUN pip install cupy-cuda11x

RUN pip install unzip numpy torch torchvision tqdm opencv-python matplotlib

# wget
RUN apt update && apt install -y wget
# git 
RUN apt update && apt install -y git


RUN git clone https://github.com/facebookresearch/segment-anything.git
WORKDIR /segment-anything/
RUN mkdir weights && cd weights && wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  && cd ..


COPY preprocessing/segment/segment.py  segment.py

CMD python3 segment.py --data-path $DATA_PATH --exp-name $DATA_PATH