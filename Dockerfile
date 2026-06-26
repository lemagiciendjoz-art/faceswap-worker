# RunPod Serverless worker — FaceFusion 3.3.2 préinstallé + modèles cuits.
# Reproduit la séquence d'install de pod_install_facefusion_if_missing(),
# mais UNE SEULE FOIS au build -> cold start quasi nul à l'exécution.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FACEFUSION_WORKDIR=/app
ENV PIP_NO_CACHE_DIR=1

# [1/6] deps système (cf. apt-get install ffmpeg python3-pip git curl wget)
RUN apt-get update -qq && \
    apt-get install -y -qq ffmpeg python3-pip git curl wget && \
    rm -rf /var/lib/apt/lists/*

# [2/6] deps pip GPU + clients (onnxruntime-gpu, cuDNN) + handler deps
RUN pip install -q \
      gdown onnxruntime-gpu nvidia-cudnn-cu12 \
      runpod \
      google-api-python-client google-auth google-auth-httplib2

# [3/6] cuDNN ld.so.conf + ldconfig (identique au sidecar)
RUN CUDNN_LIB=$(python3 -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')") && \
    echo "$CUDNN_LIB" > /etc/ld.so.conf.d/cudnn.conf && ldconfig

# [4/6] git clone FaceFusion 3.3.2 -> /app
RUN git clone --quiet --depth 1 --branch 3.3.2 \
      https://github.com/facefusion/facefusion.git /app

WORKDIR /app

# [5/6] requirements FaceFusion
RUN pip install -q -r requirements.txt

# [6/6] install.py FaceFusion (onnxruntime cuda)
RUN python install.py --onnxruntime cuda --skip-conda

# --- Modèles cuits dans l'image (évite tout download au cold start) ---
# inswapper_128 (swapper) + gfpgan_1.4 (enhancer) + détecteurs requis.
# force-download récupère l'ensemble des assets dans /app/.assets.
RUN python facefusion.py force-download || true

# Handler serverless
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
