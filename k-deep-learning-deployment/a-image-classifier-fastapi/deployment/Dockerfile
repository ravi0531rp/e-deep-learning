FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

WORKDIR /deploy
COPY ./requirements.txt /deploy/requirements.txt

ARG DEBIAN_FRONTEND=noninteractive


RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils && \
    apt install -y locales locales-all gunicorn && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./ /deploy

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV NUM_WORKERS 4

EXPOSE 80
EXPOSE 5001
EXPOSE 8082

RUN python3 -m pip install gunicorn==20.1.0

CMD gunicorn -k uvicorn.workers.UvicornWorker -w $NUM_WORKERS -b 0.0.0.0:5001 --timeout 0 app:app