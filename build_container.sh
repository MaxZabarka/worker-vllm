# sudo DOCKER_BUILDKIT=1 docker build . \
#     --platform linux/amd64 \
#     --build-arg MODEL_NAME=TheBloke/Spicyboros-70B-2.2-AWQ
    # --build-arg STREAMING=True
    # --build-arg HUGGING_FACE_HUB_TOKEN=your_hugging_face_token_here \
    # --build-arg TOKENIZER=hf-internal-testing/llama-tokenizer \

tag=$(cat /proc/sys/kernel/random/uuid)  
sudo DOCKER_BUILDKIT=1 docker build . --platform linux/amd64 --build-arg HUGGING_FACE_HUB_TOKEN=your_hugging_face_token_here -t maxzabarka/runpod-chat-vllm -t maxzabarka/runpod-chat-vllm:$tag

sudo docker push maxzabarka/runpod-chat-vllm:$tag  
echo "maxzabarka/runpod-chat-vllm:$tag" | xclip -sel clip
echo "Copied to clipboard: maxzabarka/runpod-chat-vllm:$tag"

# sudo docker run --gpus all -it runpod-vllm-worker 

