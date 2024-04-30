# Use ubuntu 22.04 as the base image
# FROM ubuntu:22.04
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    gcc \
    python3.10 \
    python3-pip


# Copy the necessary files into the container at /app
COPY dataloader.py /usr/local/bin/dataloader.py
COPY deepseq.py /usr/local/bin/deepseq.py
COPY inference.py /usr/local/bin/inference.py
COPY main.py /usr/local/bin/main.py
COPY requirements.txt /usr/local/bin/requirements.txt
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# COPY serve /usr/local/bin/serve


# Set the working directory to /app
WORKDIR /usr/local/bin

# ONLY FOR TESTING: Copy pretrained model
# COPY data/pretrained_weight.pth /opt/ml/model/pretrained_weight.pth


# change permissions of the entrypoint script
RUN chmod +x /usr/local/bin/entrypoint.sh

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

CMD ["serve"]
# Local testings
# docker build -t tf-binding-inference .
# docker run -p 8080:8080 tf-binding-inference serve
# curl -X POST -F "file=@data/dataset.tfrecord" http://localhost:8080/invocations --output downloaded_file.tsv