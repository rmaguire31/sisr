FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel

RUN apt-get update \
    && apt-get install -y \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml environment.yml
RUN conda env update -n base -f environment.yml

COPY python /python
WORKDIR /python

ENTRYPOINT ["/bin/bash", "-c"]
