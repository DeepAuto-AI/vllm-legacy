# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/source/dev/dockerfile/dockerfile.rst and
# docs/source/assets/dev/dockerfile-stages-dependency.png

#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip git

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-12.4/compat/

WORKDIR /workspace

# install build and runtime dependencies
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-cuda.txt

# install development dependencies
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
#################### BASE BUILD IMAGE ####################


#################### WHEEL BUILD IMAGE ####################
FROM dev AS build

# install build dependencies
COPY requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-build.txt

# install compiler cache to speed up compilation leveraging local or remote caching
RUN apt-get update -y && apt-get install -y ccache

# files and directories related to build wheels
COPY csrc csrc
COPY setup.py setup.py
COPY cmake cmake
COPY CMakeLists.txt CMakeLists.txt
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
COPY pyproject.toml pyproject.toml
COPY vllm vllm

# max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads
# make sure punica kernels are built (for LoRA)
ENV VLLM_INSTALL_PUNICA_KERNELS=1

ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    python3 setup.py bdist_wheel --dist-dir=dist

# check the size of the wheel, we cannot upload wheels larger than 100MB
COPY .buildkite/check-wheel-size.py check-wheel-size.py
RUN python3 check-wheel-size.py dist

#################### EXTENSION Build IMAGE ####################

#################### vLLM installation IMAGE ####################
# image with vLLM installed
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS vllm-base
WORKDIR /vllm-workspace

RUN apt-get update -y \
    && apt-get install -y python3-pip git vim

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-12.4/compat/

# install vllm wheel first, so that torch etc will be installed
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/pip \
    pip install dist/*.whl --verbose
#################### vLLM installation IMAGE ####################


#################### TEST IMAGE ####################
# image to run unit testing suite
# note that this uses vllm installed by `pip`
FROM vllm-base AS test

# copy pytorch extensions separately to avoid having to rebuild
WORKDIR /vllm-workspace
# when python code changes
# ADD is used to preserve directory structure
ADD . /vllm-workspace/

# install development dependencies (for testing)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-dev.txt

# doc requires source code
# we hide them inside `test_docs/` , so that this source code
# will not be imported by other tests
RUN mkdir test_docs
RUN mv docs test_docs/
RUN mv vllm test_docs/

#################### TEST IMAGE ####################

#################### OPENAI API SERVER ####################
# openai api server alternative
FROM vllm-base AS vllm-openai

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate hf_transfer modelscope

ENV VLLM_USAGE_SOURCE production-docker-image

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
#################### OPENAI API SERVER ####################

#################### HiP Attention ####################
# Copy hip
COPY --from=hip . /workspace/hip/
# NOTE: We upgrade pip to support PEP 660 (pyproject.toml with editable installs)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip \
    && pip install -e "/workspace/hip[openai]" \
    && pip install -e "/workspace/hip[no_build_iso]" --no-build-isolation
#################### HiP Attention ####################
