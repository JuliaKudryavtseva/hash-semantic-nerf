FROM ubuntu:22.04

# install python
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 

# install dependancy
RUN pip install torch matplotlib opencv-python


# multiprocess libs
RUN apt update && apt install nvidia-cuda-toolkit -y
RUN pip install cupy-cuda11x

# wget
RUN apt update && apt install -y wget
# git 
RUN apt update && apt install -y git


# install XMem
RUN git clone https://github.com/hkchengrex/XMem.git
WORKDIR /XMem
RUN pip install -r requirements.txt
RUN wget -P ./saves/ https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth

# copy files
COPY preprocessing/consistency/masks_consist.py masks_consist.py
COPY preprocessing/consistency/multi_iou.py multi_iou.py

# run
CMD python3 masks_consist.py