# Base: NVCR PyTorch image with CUDA 12.4, cuDNN 9, PyTorch 2.5
# Pinned for reproducibility — update tag to upgrade CUDA/PyTorch together.
FROM nvcr.io/nvidia/pytorch:26.06-py3

WORKDIR /workspace

# Install flash-attn separately (needs CUDA headers from base image)
RUN pip install flash-attn==2.6.3 --no-build-isolation

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional extras — uncomment to include in the image
# RUN pip install wandb lm-eval torchao

COPY . .

ENV PYTHONPATH=/workspace
ENV TOKENIZERS_PARALLELISM=false

CMD ["python", "train.py", "--config", "configs/3b.yaml"]
