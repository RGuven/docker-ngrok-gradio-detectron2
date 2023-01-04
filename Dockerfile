FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y gcc libglib2.0-0

#Install Ninja for fast build to detectron2
#Download the latest version of Ninja from GitHub:
RUN apt-get update \
        && apt-get install wget unzip zip -y
RUN wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip
RUN gunzip /usr/local/bin/ninja.gz
RUN chmod a+x /usr/local/bin/ninja

#nvidia image is using python 3.6.9 but detectron2 should have above 3.7, so upgrade the python.
RUN apt-get -y install software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && apt -y install python3.10
# now you have python3.8.10 and python3.10.9 after this cmd

RUN rm -rf /usr/bin/python
RUN rm -rf /usr/bin/python3

RUN ln -sv /usr/bin/python3.10 /usr/bin/python
RUN ln -sv /usr/bin/python3.8 /usr/bin/python3
#Now we have 2 version of python which is python -> 3.10.9 and python3 -> 3.8.10

#upgrade git
RUN apt-get -y upgrade git

#install pip
RUN apt install -y python3-pip

#install pycocotools
RUN pip3 install pycocotools 

#to solve Pillow error
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*

#install torch
RUN pip3 install torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113

#install detectron2
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

WORKDIR /workspace

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8888

ENTRYPOINT python3 gradio_web_ui.py