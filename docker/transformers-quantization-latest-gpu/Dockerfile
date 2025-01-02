FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# Use login shell to read variables from `~/.profile` (to pass dynamic created variables between RUN commands)
SHELL ["sh", "-lc"]

# The following `ARG` are mainly used to specify the versions explicitly & directly in this docker file, and not meant
# to be used as arguments for docker build (so far).

ARG PYTORCH='2.5.1'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu118'

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; echo "export VERSION='$VERSION'" >> ~/.profile
RUN echo torch=$VERSION
# `torchvision` and `torchaudio` should be installed along with `torch`, especially for nightly build.
# Currently, let's just use their latest releases (when `torch` is installed with a release version)
RUN python3 -m pip install --no-cache-dir -U $VERSION torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-torch]

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/accelerate@main#egg=accelerate

# needed in bnb and awq
RUN python3 -m pip install --no-cache-dir einops

# Add bitsandbytes for mixed int8 testing
RUN python3 -m pip install --no-cache-dir bitsandbytes

# Add auto-gptq for gtpq quantization testing, installed from source for pytorch==2.5.1 compatibility
# TORCH_CUDA_ARCH_LIST="7.5+PTX" is added to make the package compile for Tesla T4 gpus available for the CI.
RUN pip install gekko
RUN git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ && TORCH_CUDA_ARCH_LIST="7.5+PTX" python3 setup.py install

# Add optimum for gptq quantization testing
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/optimum@main#egg=optimum

# Add PEFT
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/peft@main#egg=peft

# Add aqlm for quantization testing
RUN python3 -m pip install --no-cache-dir aqlm[gpu]==1.0.2

# Add vptq for quantization testing
RUN python3 -m pip install --no-cache-dir vptq

# Add hqq for quantization testing
RUN python3 -m pip install --no-cache-dir hqq

# For GGUF tests
RUN python3 -m pip install --no-cache-dir gguf

# Add autoawq for quantization testing
# >=v0.2.7 needed for compatibility with transformers > 4.46
RUN python3 -m pip install --no-cache-dir https://github.com/casper-hansen/AutoAWQ/releases/download/v0.2.7.post2/autoawq-0.2.7.post2-py3-none-any.whl

# Add quanto for quantization testing
RUN python3 -m pip install --no-cache-dir optimum-quanto

# Add eetq for quantization testing
RUN python3 -m pip install git+https://github.com/NetEase-FuXi/EETQ.git

# Add flute-kernel and fast_hadamard_transform for quantization testing
RUN python3 -m pip install --no-cache-dir flute-kernel==0.3.0 -i https://flute-ai.github.io/whl/cu118
RUN python3 -m pip install --no-cache-dir fast_hadamard_transform==1.0.4.post1

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop
