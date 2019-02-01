FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel

COPY environment.yml environment.yml
RUN conda env update -n base -f environment.yml

COPY python /python
WORKDIR /python

ENTRYPOINT ["/bin/bash", "-c"]
