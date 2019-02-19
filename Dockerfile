FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel

# Dependencies are updated less often and are slow to build
COPY environment.yml environment.yml
RUN conda env update -n base -f environment.yml

# This package is updated more often
COPY . /sisr
RUN pip install /sisr

