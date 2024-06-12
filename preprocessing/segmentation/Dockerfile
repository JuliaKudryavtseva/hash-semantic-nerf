FROM ubuntu:22.04

# install app dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install numpy tqdm opencv-python matplotlib
RUN pip install torch torchvision
# RUN apt update && apt install zip

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 

# multiprocess libs
RUN apt install nvidia-cuda-toolkit -y
RUN pip install cupy-cuda11x

# wget
RUN apt update && apt install wget
# git 
RUN apt update && apt install -y git


RUN  git clone https://github.com/facebookresearch/segment-anything.git

WORKDIR /segment-anything/

RUN mkdir weights && cd weights && wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  && cd ..

COPY preprocessing/segmentation/generate_dataset.py  generate_dataset.py
COPY preprocessing/segmentation/cupy_module.py  cupy_module.py


CMD python3 generate_dataset.py --data-path $DATA_PATH --exp-name $DATA_PATH