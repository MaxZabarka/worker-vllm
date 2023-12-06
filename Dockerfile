# Base image
# The following docker base image is recommended by VLLM: 
# FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel
# FROM nvcr.io/nvidia/pytorch:22.12-py3
# FROM nvidia/cuda:11.7.1-devel-ubi8
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set the working directory
WORKDIR /

# Update and upgrade the system packages (Worker Template)
ARG DEBIAN_FRONTEND=noninteractive
# RUN pip uninstall torch -y
# RUN pip install torch==2.0.1 -f https://download.pytorch.org/whl/cu118
COPY builder/setup.sh /setup.sh
RUN chmod +x /setup.sh && \
    /setup.sh && \
    rm /setup.sh

RUN apt-get update
RUN apt-get install python3.10 -y 
RUN apt-get install python3-pip -y
RUN apt-get install git -y

# Install fast api
RUN pip install fastapi==0.99.1

ENV bust_cach="123fdsfsdf"
RUN git clone https://github.com/MaxZabarka/vllm

ENV PIP_DEFAULT_TIMEOUT=100
RUN --mount=type=cache,target=/root/.cache/pip \
  pip install -e vllm --verbose
 
# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt


# Quick temporary updates
RUN pip install git+https://github.com/runpod/runpod-python@a1#egg=runpod --compile

# Prepare the models inside the docker image
ARG HUGGING_FACE_HUB_TOKEN=
ENV HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Prepare argument for the model and tokenizer
# ARG MODEL_NAME=""
# ENV MODEL_NAME=$MODEL_NAME
# ARG MODEL_REVISION="main"
# ENV MODEL_REVISION=$MODEL_REVISION
# ARG MODEL_BASE_PATH="/runpod-volume/"
# ENV MODEL_BASE_PATH=$MODEL_BASE_PATH
# ARG TOKENIZER=
# ENV TOKENIZER=$TOKENIZER
# ARG STREAMING=
# ENV STREAMING=$STREAMING

ENV HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
ENV TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"

# Add src files (Worker Template)
ADD src .  
# Download the models
RUN mkdir -p /model

# Set environment variables
# ENV MODEL_NAME=$MODEL_NAME \
#     MODEL_REVISION=$MODEL_REVISION \
#     MODEL_BASE_PATH=$MODEL_BASE_PATH \
#     HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Run the Python script to download the model
# RUN python -u /download_model.py

ENV bust_cache123="asdfsdf"
RUN git -C /vllm pull

RUN safefilename=
# Start the handler
RUN mkdir -p /runpod-volume
CMD STREAMING=$STREAMING MODEL_NAME=$MODEL_NAME MODEL_BASE_PATH=$MODEL_BASE_PATH TOKENIZER=$TOKENIZER python3 -u /handler.py 
# &> /runpod-volume/handler_$(echo "$MODEL_NAME" | tr -d '\n\t\r' | sed 's/[^a-zA-Z0-9_.-]/_/g').log
