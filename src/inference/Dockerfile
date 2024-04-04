# Use ubuntu 22.04 as the base image
# FROM ubuntu:22.04
FROM --platform=linux/amd64 nvidia/cuda:12.3.2-base-ubuntu22.04
# Install gcc to compile any extensions
RUN apt-get update && apt-get install -y \
    gcc \
    python3.10 \
    python3-pip


# Copy the necessary files into the container at /app
COPY dataloader.py /opt/program/dataloader.py
COPY deepseq.py /opt/program/deepseq.py
COPY inference.py /opt/program/inference.py
COPY main.py /opt/program/main.py
COPY requirements.txt /opt/program/requirements.txt
COPY entrypoint.sh /opt/program/entrypoint.sh
# COPY serve /opt/program/serve


# Set the working directory to /app
WORKDIR /opt/program

# ONLY FOR TESTING: Copy pretrained model
# COPY data/pretrained_weight.pth /opt/ml/model/pretrained_weight.pth


# Make entrypoint.sh executable
RUN chmod +x entrypoint.sh

# Install FastAPI, Uvicorn, and other required packages
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV MODEL_PATH="/opt/ml/model"

# Expose the port FastAPI will run on
EXPOSE 8080

# Set the entry point to use the entry script
ENTRYPOINT ["./entrypoint.sh"]

# Local testings
# docker build -t tf-binding-inference .
# docker run -p 8080:8080 tf-binding-inference serve
# curl -X POST -F "file=@data/dataset.tfrecord" http://localhost:8080/invocations --output downloaded_file.tsv