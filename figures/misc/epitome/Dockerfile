# start with docker pull tensorflow/tensorflow:2.6.1
FROM python:3.7-slim

# install git and other tools
RUN apt-get update && apt-get install -y git

RUN pip install tensorflow==2.10.0 tensorflow-io==0.27.0
RUN git clone https://github.com/YosefLab/epitome.git

# bring directory into container
COPY . /app

# change working directroy to epitome
WORKDIR /app
#


