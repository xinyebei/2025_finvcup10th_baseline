# TODOs: 国内无法访问docker hub
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

#更换阿里源
RUN sed -i 's/https:\/\/mirrors.aliyun.com/http:\/\/mirrors.cloud.aliyuncs.com/g' /etc/apt/sources.list

#设置时区
ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

# 修改shell为bash
RUN rm /bin/sh && ln -s /usr/bin/bash /bin/sh


#安装一些必要的库
RUN apt-get update
RUN apt-get install -y \
    sudo \
    git \
    cmake \
    vim \
    wget \
    curl \
    tmux \
    zip \
    unzip \
    ffmpeg \
    build-essential \
    libfreeimage-dev 


# 设置环境变量 url
ENV HOME /algorithm
ENV HF_ENDPOINT https://hf-mirror.com


# 切换工作空间到对应目录
WORKDIR ${HOME}   

COPY ./requirements.txt ${HOME}/requirements.txt


# pip 更换为清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 安装依赖
RUN pip install -r requirements.txt

# 清理 APT 缓存
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -Rf /tmp/*
# 清理pip缓存
RUN rm -rf /root/.cache/pip
# 清理 CONDA 缓存
RUN find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    conda clean -afy
